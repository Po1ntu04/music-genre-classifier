from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class SongEmbeddingDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], label_to_index: dict[str, int]) -> None:
        self.rows = rows
        self.label_to_index = label_to_index

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        payload = torch.load(row["song_embedding_path"], map_location="cpu")
        embedding = payload["embedding"].to(torch.float32)
        label = self.label_to_index[row["genre"]]
        return embedding, label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a song-level genre classifier on frozen MERT embeddings.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the YAML config file.")
    parser.add_argument(
        "--model-type",
        choices=["linear", "mlp"],
        default=None,
        help="Override classifier type from config.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count from config.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size from config.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override learning rate from config.")
    parser.add_argument(
        "--embedding-suffix",
        type=str,
        default="",
        help="Optional song embedding suffix, for example layer_08 -> train_song_embeddings_layer_08.csv.",
    )
    return parser.parse_args()


def collate_batch(batch: list[tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor]:
    embeddings, labels = zip(*batch)
    return torch.stack(list(embeddings), dim=0), torch.tensor(labels, dtype=torch.long)


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, list[int], list[int]]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    all_labels: list[int] = []
    all_predictions: list[int] = []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for embeddings, labels in tqdm(
            dataloader,
            desc="train" if is_train else "val",
            leave=False,
            unit="batch",
        ):
            embeddings = embeddings.to(device)
            labels = labels.to(device)

            logits = model(embeddings)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            predictions = logits.argmax(dim=1)
            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    average_loss = total_loss / max(len(dataloader.dataset), 1)
    return average_loss, all_labels, all_predictions


def main() -> None:
    args = parse_args()

    from src.models.classifier_head import build_classifier
    from src.train.metrics import plot_loss_curves
    from src.utils.config import load_config
    from src.utils.io import read_csv_rows
    from src.utils.runtime import resolve_device
    from src.utils.seed import set_seed

    config = load_config(args.config)
    set_seed(config["project"]["seed"])
    device = resolve_device(config["project"]["device"])

    embedding_suffix = args.embedding_suffix.strip()
    embedding_suffix_fragment = f"_{embedding_suffix}" if embedding_suffix else ""

    model_type = args.model_type or config["training"]["model_type"]
    epochs = args.epochs or config["training"]["epochs"]
    batch_size = args.batch_size or config["training"]["batch_size"]
    learning_rate = args.learning_rate or config["training"]["learning_rate"]

    train_rows = read_csv_rows(config["paths"]["splits_dir"] / f"train_song_embeddings{embedding_suffix_fragment}.csv")
    val_rows = read_csv_rows(config["paths"]["splits_dir"] / f"val_song_embeddings{embedding_suffix_fragment}.csv")
    class_names = sorted({row["genre"] for row in train_rows + val_rows})
    label_to_index = {name: idx for idx, name in enumerate(class_names)}

    if not train_rows or not val_rows:
        raise RuntimeError("Train/val song embeddings are missing. Run extraction and aggregation first.")

    sample_embedding = torch.load(train_rows[0]["song_embedding_path"], map_location="cpu")["embedding"]
    input_dim = int(sample_embedding.numel())

    train_dataset = SongEmbeddingDataset(train_rows, label_to_index)
    val_dataset = SongEmbeddingDataset(val_rows, label_to_index)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    model = build_classifier(
        model_type=model_type,
        input_dim=input_dim,
        num_classes=len(class_names),
        hidden_dim=config["training"]["hidden_dim"],
        dropout=config["training"]["dropout"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=config["training"]["weight_decay"],
    )

    checkpoints_dir = config["paths"]["outputs_dir"] / "checkpoints"
    logs_dir = config["paths"]["outputs_dir"] / "logs"
    figures_dir = config["paths"]["outputs_dir"] / "figures"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    best_metric = float("-inf")
    best_epoch = 0
    best_val_accuracy = 0.0
    best_val_macro_f1 = 0.0
    output_tag = model_type if not embedding_suffix else f"{model_type}_{embedding_suffix}"
    best_checkpoint_path = checkpoints_dir / f"best_{output_tag}.pt"
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "train_macro_f1": [],
        "val_macro_f1": [],
    }

    for epoch in range(1, epochs + 1):
        train_loss, train_labels, train_predictions = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_labels, val_predictions = run_epoch(model, val_loader, criterion, device, optimizer=None)

        train_accuracy = accuracy_score(train_labels, train_predictions)
        val_accuracy = accuracy_score(val_labels, val_predictions)
        train_macro_f1 = f1_score(train_labels, train_predictions, average="macro")
        val_macro_f1 = f1_score(val_labels, val_predictions, average="macro")

        history["epoch"].append(epoch)
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["train_accuracy"].append(float(train_accuracy))
        history["val_accuracy"].append(float(val_accuracy))
        history["train_macro_f1"].append(float(train_macro_f1))
        history["val_macro_f1"].append(float(val_macro_f1))

        monitor_metric = val_macro_f1 if config["training"]["checkpoint_metric"] == "macro_f1" else -val_loss
        if monitor_metric > best_metric:
            best_metric = monitor_metric
            best_epoch = epoch
            best_val_accuracy = float(val_accuracy)
            best_val_macro_f1 = float(val_macro_f1)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_type": model_type,
                    "input_dim": input_dim,
                    "num_classes": len(class_names),
                    "class_names": class_names,
                    "hidden_dim": config["training"]["hidden_dim"],
                    "dropout": config["training"]["dropout"],
                    "epoch": epoch,
                    "val_accuracy": float(val_accuracy),
                    "val_macro_f1": float(val_macro_f1),
                    "embedding_suffix": embedding_suffix,
                    "config_path": str(args.config.resolve()),
                },
                best_checkpoint_path,
            )

        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"train_acc={train_accuracy:.4f} val_acc={val_accuracy:.4f} "
            f"train_f1={train_macro_f1:.4f} val_f1={val_macro_f1:.4f}"
        )

    history_path = logs_dir / f"{output_tag}_history.json"
    with history_path.open("w", encoding="utf-8") as file:
        json.dump(history, file, ensure_ascii=False, indent=2)

    plot_loss_curves(history, figures_dir / f"{output_tag}_loss_curve.png")
    print(f"Best checkpoint: {best_checkpoint_path}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val accuracy: {best_val_accuracy:.4f}")
    print(f"Best val macro F1: {best_val_macro_f1:.4f}")


if __name__ == "__main__":
    main()
