from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class SongAllLayerEmbeddingDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        label_to_index: dict[str, int],
        layer_indices: list[int] | None = None,
    ) -> None:
        grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            grouped[(row["genre"], row["song_id"])].append(row)

        self.items = []
        self.label_to_index = label_to_index
        self.layer_indices = layer_indices

        for (genre, song_id), song_rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
            song_rows = sorted(song_rows, key=lambda item: int(item["segment_index"]))
            self.items.append(
                {
                    "song_id": song_id,
                    "genre": genre,
                    "embedding_paths": [row["embedding_path"] for row in song_rows],
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        item = self.items[index]
        segment_embeddings = []
        for path in item["embedding_paths"]:
            payload = torch.load(path, map_location="cpu")
            available_indices = [int(value) for value in payload["layer_indices"]]
            if self.layer_indices is None:
                layer_positions = list(range(len(available_indices)))
            else:
                layer_positions = [available_indices.index(layer_index) for layer_index in self.layer_indices]
            embedding = payload["layer_embeddings"][layer_positions].to(torch.float32)
            segment_embeddings.append(embedding)
        stacked = torch.stack(segment_embeddings, dim=0)
        label = self.label_to_index[item["genre"]]
        return stacked, label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a layer-weighted song classifier from all segment layers.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the YAML config file.")
    parser.add_argument(
        "--aggregation-type",
        choices=["mean", "mean_std", "attention"],
        default=None,
        help="How to aggregate the 6 segment embeddings after layer weighting.",
    )
    parser.add_argument(
        "--model-type",
        choices=["linear", "mlp"],
        default=None,
        help="Classifier head applied after segment aggregation.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count from config.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size from config.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override learning rate from config.")
    parser.add_argument("--label-smoothing", type=float, default=None, help="Override label smoothing value.")
    parser.add_argument("--chunk-dropout-prob", type=float, default=None, help="Override chunk dropout probability.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed from config.")
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
) -> tuple[float, list[int], list[int], list[float] | None]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    all_labels: list[int] = []
    all_predictions: list[int] = []
    last_layer_weights: list[float] | None = None

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

            logits, extras = model(embeddings)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            predictions = logits.argmax(dim=1)
            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())
            if "layer_weights" in extras:
                last_layer_weights = [float(value) for value in extras["layer_weights"].detach().cpu().tolist()]

    average_loss = total_loss / max(len(dataloader.dataset), 1)
    return average_loss, all_labels, all_predictions, last_layer_weights


def main() -> None:
    args = parse_args()

    from src.models.layer_weighted_segment_aggregator import build_layer_weighted_segment_classifier
    from src.train.metrics import plot_loss_curves
    from src.utils.config import load_config
    from src.utils.io import read_csv_rows
    from src.utils.runtime import resolve_device
    from src.utils.seed import set_seed

    config = load_config(args.config)
    seed = args.seed if args.seed is not None else config["project"]["seed"]
    set_seed(seed)
    device = resolve_device(config["project"]["device"])

    aggregation_type = args.aggregation_type or config["segment_training"]["aggregation_type"]
    model_type = args.model_type or config["segment_training"]["model_type"]
    epochs = args.epochs or config["segment_training"]["epochs"]
    batch_size = args.batch_size or config["segment_training"]["batch_size"]
    learning_rate = args.learning_rate or config["segment_training"]["learning_rate"]
    label_smoothing = (
        args.label_smoothing
        if args.label_smoothing is not None
        else config["segment_training"]["label_smoothing"]
    )
    chunk_dropout_prob = (
        args.chunk_dropout_prob
        if args.chunk_dropout_prob is not None
        else config["segment_training"]["chunk_dropout_prob"]
    )
    layer_indices = [int(index) for index in config["feature_extraction"]["layer_indices"]]

    train_rows = read_csv_rows(config["paths"]["splits_dir"] / "train_segment_embeddings.csv")
    val_rows = read_csv_rows(config["paths"]["splits_dir"] / "val_segment_embeddings.csv")
    class_names = sorted({row["genre"] for row in train_rows + val_rows})
    label_to_index = {name: idx for idx, name in enumerate(class_names)}

    train_dataset = SongAllLayerEmbeddingDataset(train_rows, label_to_index, layer_indices=layer_indices)
    val_dataset = SongAllLayerEmbeddingDataset(val_rows, label_to_index, layer_indices=layer_indices)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    sample_embeddings, _ = train_dataset[0]
    num_layers = int(sample_embeddings.shape[1])
    input_dim = int(sample_embeddings.shape[-1])

    model = build_layer_weighted_segment_classifier(
        num_layers=num_layers,
        input_dim=input_dim,
        num_classes=len(class_names),
        aggregation_type=aggregation_type,
        model_type=model_type,
        hidden_dim=config["segment_training"]["hidden_dim"],
        dropout=config["segment_training"]["dropout"],
        chunk_dropout_prob=chunk_dropout_prob,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=config["segment_training"]["weight_decay"],
    )

    checkpoints_dir = config["paths"]["outputs_dir"] / "checkpoints"
    logs_dir = config["paths"]["outputs_dir"] / "logs"
    figures_dir = config["paths"]["outputs_dir"] / "figures"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    output_tag = f"layer_weighted_{aggregation_type}_{model_type}_all_layers"
    if label_smoothing > 0:
        output_tag += f"_ls{int(round(label_smoothing * 100)):02d}"
    if chunk_dropout_prob > 0:
        output_tag += f"_cd{int(round(chunk_dropout_prob * 100)):02d}"
    output_tag += f"_seed{seed}"

    best_metric = float("-inf")
    best_epoch = 0
    best_val_accuracy = 0.0
    best_val_macro_f1 = 0.0
    best_layer_weights: list[float] = []
    best_checkpoint_path = checkpoints_dir / f"best_{output_tag}.pt"
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "train_macro_f1": [],
        "val_macro_f1": [],
        "layer_weights": [],
    }

    for epoch in range(1, epochs + 1):
        train_loss, train_labels, train_predictions, train_layer_weights = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
        )
        val_loss, val_labels, val_predictions, val_layer_weights = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            optimizer=None,
        )

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
        history["layer_weights"].append(val_layer_weights or train_layer_weights or [])

        monitor_metric = val_macro_f1 if config["segment_training"]["checkpoint_metric"] == "macro_f1" else -val_loss
        if monitor_metric > best_metric:
            best_metric = monitor_metric
            best_epoch = epoch
            best_val_accuracy = float(val_accuracy)
            best_val_macro_f1 = float(val_macro_f1)
            best_layer_weights = list(val_layer_weights or train_layer_weights or [])
            torch.save(
                {
                    "model_family": "layer_weighted_segment_aggregator",
                    "model_state_dict": model.state_dict(),
                    "aggregation_type": aggregation_type,
                    "model_type": model_type,
                    "input_dim": input_dim,
                    "num_classes": len(class_names),
                    "class_names": class_names,
                    "hidden_dim": config["segment_training"]["hidden_dim"],
                    "dropout": config["segment_training"]["dropout"],
                    "chunk_dropout_prob": chunk_dropout_prob,
                    "label_smoothing": float(label_smoothing),
                    "seed": int(seed),
                    "epoch": epoch,
                    "val_accuracy": float(val_accuracy),
                    "val_macro_f1": float(val_macro_f1),
                    "layer_indices": layer_indices,
                    "num_layers": num_layers,
                    "best_layer_weights": best_layer_weights,
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
    print(f"Best layer weights: {[round(value, 4) for value in best_layer_weights]}")


if __name__ == "__main__":
    main()
