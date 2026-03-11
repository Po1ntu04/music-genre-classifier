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


class SongSegmentEmbeddingDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        label_to_index: dict[str, int],
        embedding_suffix: str,
    ) -> None:
        grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            grouped[(row["genre"], row["song_id"])].append(row)

        self.items = []
        self.label_to_index = label_to_index
        self.layer_index = None
        if embedding_suffix.startswith("layer_"):
            self.layer_index = int(embedding_suffix.split("_")[-1])

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
            if self.layer_index is None:
                embedding = payload["embedding"]
            else:
                layer_indices = [int(value) for value in payload["layer_indices"]]
                layer_position = layer_indices.index(self.layer_index)
                embedding = payload["layer_embeddings"][layer_position]
            segment_embeddings.append(embedding.to(torch.float32))
        stacked = torch.stack(segment_embeddings, dim=0)
        label = self.label_to_index[item["genre"]]
        return stacked, label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a song classifier directly from segment embeddings.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the YAML config file.")
    parser.add_argument(
        "--embedding-suffix",
        type=str,
        default=None,
        help="Segment embedding suffix to select a transformer layer, for example layer_06.",
    )
    parser.add_argument(
        "--aggregation-type",
        choices=["mean", "mean_std", "attention"],
        default=None,
        help="How to aggregate the 6 segment embeddings into a song representation.",
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
    parser.add_argument("--warmup-ratio", type=float, default=None, help="Override warmup ratio from config.")
    parser.add_argument(
        "--scheduler-type",
        choices=["none", "linear_warmup", "linear_warmup_cosine"],
        default=None,
        help="Learning-rate schedule applied during training.",
    )
    parser.add_argument("--time-mask-prob", type=float, default=None, help="Probability of masking song segments.")
    parser.add_argument("--time-mask-max-width", type=int, default=None, help="Maximum contiguous masked segments.")
    parser.add_argument(
        "--freq-mask-prob",
        type=float,
        default=None,
        help="Probability of masking a contiguous feature band.",
    )
    parser.add_argument(
        "--freq-mask-max-width-ratio",
        type=float,
        default=None,
        help="Maximum masked feature width as a ratio of the embedding dimension.",
    )
    parser.add_argument(
        "--gaussian-noise-prob",
        type=float,
        default=None,
        help="Probability of adding Gaussian noise to a song embedding tensor.",
    )
    parser.add_argument(
        "--gaussian-noise-std",
        type=float,
        default=None,
        help="Standard deviation of Gaussian noise added to embeddings.",
    )
    parser.add_argument(
        "--random-loudness-prob",
        type=float,
        default=None,
        help="Probability of scaling an embedding tensor with a random gain.",
    )
    parser.add_argument(
        "--random-loudness-db-range",
        type=float,
        default=None,
        help="Maximum absolute gain in dB for random loudness augmentation.",
    )
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
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    augmenter: object | None = None,
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
            if is_train and augmenter is not None:
                embeddings = augmenter(embeddings)

            logits, _ = model(embeddings)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            total_loss += loss.item() * labels.size(0)
            predictions = logits.argmax(dim=1)
            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    average_loss = total_loss / max(len(dataloader.dataset), 1)
    return average_loss, all_labels, all_predictions


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    warmup_ratio: float,
    total_steps: int,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    if scheduler_type == "none" or total_steps <= 0:
        return None

    warmup_steps = int(total_steps * max(warmup_ratio, 0.0))
    warmup_steps = min(warmup_steps, max(total_steps - 1, 0))

    if scheduler_type == "linear_warmup":
        if warmup_steps <= 0:
            return None
        return torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps,
        )

    if scheduler_type == "linear_warmup_cosine":
        if warmup_steps <= 0:
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine_steps = max(total_steps - warmup_steps, 1)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_steps)
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_steps],
        )

    raise ValueError(f"Unsupported scheduler_type: {scheduler_type}")


def main() -> None:
    args = parse_args()

    from src.models.segment_aggregator import build_segment_classifier
    from src.train.metrics import plot_loss_curves
    from src.utils.config import load_config
    from src.utils.io import read_csv_rows
    from src.utils.runtime import resolve_device
    from src.utils.seed import set_seed
    from src.utils.segment_augment import SegmentEmbeddingAugmenter

    config = load_config(args.config)
    seed = args.seed if args.seed is not None else config["project"]["seed"]
    set_seed(seed)
    device = resolve_device(config["project"]["device"])

    embedding_suffix = args.embedding_suffix or config["segment_training"]["embedding_suffix"]
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
    scheduler_type = args.scheduler_type or config["segment_training"].get("scheduler_type", "none")
    warmup_ratio = (
        args.warmup_ratio if args.warmup_ratio is not None else config["segment_training"].get("warmup_ratio", 0.0)
    )
    chunk_dropout_prob = (
        args.chunk_dropout_prob
        if args.chunk_dropout_prob is not None
        else config["segment_training"]["chunk_dropout_prob"]
    )
    time_mask_prob = (
        args.time_mask_prob if args.time_mask_prob is not None else config["segment_training"].get("time_mask_prob", 0.0)
    )
    time_mask_max_width = (
        args.time_mask_max_width
        if args.time_mask_max_width is not None
        else config["segment_training"].get("time_mask_max_width", 1)
    )
    freq_mask_prob = (
        args.freq_mask_prob if args.freq_mask_prob is not None else config["segment_training"].get("freq_mask_prob", 0.0)
    )
    freq_mask_max_width_ratio = (
        args.freq_mask_max_width_ratio
        if args.freq_mask_max_width_ratio is not None
        else config["segment_training"].get("freq_mask_max_width_ratio", 0.0)
    )
    gaussian_noise_prob = (
        args.gaussian_noise_prob
        if args.gaussian_noise_prob is not None
        else config["segment_training"].get("gaussian_noise_prob", 0.0)
    )
    gaussian_noise_std = (
        args.gaussian_noise_std
        if args.gaussian_noise_std is not None
        else config["segment_training"].get("gaussian_noise_std", 0.0)
    )
    random_loudness_prob = (
        args.random_loudness_prob
        if args.random_loudness_prob is not None
        else config["segment_training"].get("random_loudness_prob", 0.0)
    )
    random_loudness_db_range = (
        args.random_loudness_db_range
        if args.random_loudness_db_range is not None
        else config["segment_training"].get("random_loudness_db_range", 0.0)
    )

    train_rows = read_csv_rows(config["paths"]["splits_dir"] / "train_segment_embeddings.csv")
    val_rows = read_csv_rows(config["paths"]["splits_dir"] / "val_segment_embeddings.csv")
    class_names = sorted({row["genre"] for row in train_rows + val_rows})
    label_to_index = {name: idx for idx, name in enumerate(class_names)}

    train_dataset = SongSegmentEmbeddingDataset(train_rows, label_to_index, embedding_suffix=embedding_suffix)
    val_dataset = SongSegmentEmbeddingDataset(val_rows, label_to_index, embedding_suffix=embedding_suffix)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    sample_embeddings, _ = train_dataset[0]
    input_dim = int(sample_embeddings.shape[-1])

    model = build_segment_classifier(
        aggregation_type=aggregation_type,
        input_dim=input_dim,
        num_classes=len(class_names),
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
    total_steps = epochs * max(len(train_loader), 1)
    scheduler = build_scheduler(
        optimizer=optimizer,
        scheduler_type=scheduler_type,
        warmup_ratio=warmup_ratio,
        total_steps=total_steps,
    )
    augmenter = SegmentEmbeddingAugmenter(
        time_mask_prob=time_mask_prob,
        time_mask_max_width=time_mask_max_width,
        freq_mask_prob=freq_mask_prob,
        freq_mask_max_width_ratio=freq_mask_max_width_ratio,
        gaussian_noise_prob=gaussian_noise_prob,
        gaussian_noise_std=gaussian_noise_std,
        random_loudness_prob=random_loudness_prob,
        random_loudness_db_range=random_loudness_db_range,
    )

    checkpoints_dir = config["paths"]["outputs_dir"] / "checkpoints"
    logs_dir = config["paths"]["outputs_dir"] / "logs"
    figures_dir = config["paths"]["outputs_dir"] / "figures"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    output_tag = f"segment_{aggregation_type}_{model_type}_{embedding_suffix}"
    if label_smoothing > 0:
        output_tag += f"_ls{int(round(label_smoothing * 100)):02d}"
    if chunk_dropout_prob > 0:
        output_tag += f"_cd{int(round(chunk_dropout_prob * 100)):02d}"
    if scheduler_type != "none":
        output_tag += f"_{scheduler_type}"
        if warmup_ratio > 0:
            output_tag += f"_wu{int(round(warmup_ratio * 100)):02d}"
    if time_mask_prob > 0:
        output_tag += f"_tm{int(round(time_mask_prob * 100)):02d}"
    if freq_mask_prob > 0:
        output_tag += f"_fm{int(round(freq_mask_prob * 100)):02d}"
    if gaussian_noise_prob > 0 and gaussian_noise_std > 0:
        output_tag += f"_gn{int(round(gaussian_noise_prob * 100)):02d}"
    if random_loudness_prob > 0 and random_loudness_db_range > 0:
        output_tag += f"_rl{int(round(random_loudness_prob * 100)):02d}"
    output_tag += f"_seed{seed}"

    best_metric = float("-inf")
    best_epoch = 0
    best_val_accuracy = 0.0
    best_val_macro_f1 = 0.0
    best_checkpoint_path = checkpoints_dir / f"best_{output_tag}.pt"
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "train_macro_f1": [],
        "val_macro_f1": [],
        "learning_rate": [],
    }

    for epoch in range(1, epochs + 1):
        train_loss, train_labels, train_predictions = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            scheduler=scheduler,
            augmenter=augmenter if augmenter.is_enabled() else None,
        )
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
        history["learning_rate"].append(float(optimizer.param_groups[0]["lr"]))

        monitor_metric = val_macro_f1 if config["segment_training"]["checkpoint_metric"] == "macro_f1" else -val_loss
        if monitor_metric > best_metric:
            best_metric = monitor_metric
            best_epoch = epoch
            best_val_accuracy = float(val_accuracy)
            best_val_macro_f1 = float(val_macro_f1)
            torch.save(
                {
                    "model_family": "segment_aggregator",
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
                    "scheduler_type": scheduler_type,
                    "warmup_ratio": float(warmup_ratio),
                    "time_mask_prob": float(time_mask_prob),
                    "time_mask_max_width": int(time_mask_max_width),
                    "freq_mask_prob": float(freq_mask_prob),
                    "freq_mask_max_width_ratio": float(freq_mask_max_width_ratio),
                    "gaussian_noise_prob": float(gaussian_noise_prob),
                    "gaussian_noise_std": float(gaussian_noise_std),
                    "random_loudness_prob": float(random_loudness_prob),
                    "random_loudness_db_range": float(random_loudness_db_range),
                    "embedding_suffix": embedding_suffix,
                    "seed": int(seed),
                    "epoch": epoch,
                    "val_accuracy": float(val_accuracy),
                    "val_macro_f1": float(val_macro_f1),
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
