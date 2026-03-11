from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score
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
    parser = argparse.ArgumentParser(description="Evaluate a trained classifier on the test split.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the YAML config file.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the checkpoint produced by training.")
    parser.add_argument(
        "--embedding-suffix",
        type=str,
        default=None,
        help="Optional override for the test song embedding suffix. Defaults to the value stored in the checkpoint.",
    )
    return parser.parse_args()


def collate_batch(batch: list[tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor]:
    embeddings, labels = zip(*batch)
    return torch.stack(list(embeddings), dim=0), torch.tensor(labels, dtype=torch.long)


def main() -> None:
    args = parse_args()

    from src.models.classifier_head import build_classifier
    from src.train.metrics import build_classification_report, plot_confusion_matrix
    from src.utils.config import load_config
    from src.utils.io import read_csv_rows
    from src.utils.runtime import resolve_device
    from src.utils.seed import set_seed

    config = load_config(args.config)
    set_seed(config["project"]["seed"])
    device = resolve_device(config["project"]["device"])

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    embedding_suffix = checkpoint.get("embedding_suffix", "") if args.embedding_suffix is None else args.embedding_suffix
    embedding_suffix_fragment = f"_{embedding_suffix}" if embedding_suffix else ""
    class_names = checkpoint["class_names"]
    label_to_index = {name: idx for idx, name in enumerate(class_names)}

    test_rows = read_csv_rows(config["paths"]["splits_dir"] / f"test_song_embeddings{embedding_suffix_fragment}.csv")
    if not test_rows:
        raise RuntimeError("Test song embeddings are missing. Run extraction and aggregation first.")

    dataset = SongEmbeddingDataset(test_rows, label_to_index)
    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        collate_fn=collate_batch,
    )

    model = build_classifier(
        model_type=checkpoint["model_type"],
        input_dim=checkpoint["input_dim"],
        num_classes=checkpoint["num_classes"],
        hidden_dim=checkpoint["hidden_dim"],
        dropout=checkpoint["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_labels: list[int] = []
    all_predictions: list[int] = []

    with torch.no_grad():
        for embeddings, labels in tqdm(dataloader, desc="test", unit="batch"):
            embeddings = embeddings.to(device)
            logits = model(embeddings)
            predictions = logits.argmax(dim=1).cpu().tolist()
            all_predictions.extend(predictions)
            all_labels.extend(labels.tolist())

    accuracy = float(accuracy_score(all_labels, all_predictions))
    macro_f1 = float(f1_score(all_labels, all_predictions, average="macro"))
    report_text, report_dict = build_classification_report(all_labels, all_predictions, class_names)

    figures_dir = config["paths"]["outputs_dir"] / "figures"
    reports_dir = config["paths"]["outputs_dir"] / "reports"
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    output_tag = checkpoint["model_type"] if not embedding_suffix else f"{checkpoint['model_type']}_{embedding_suffix}"
    confusion_matrix_path = figures_dir / f"{output_tag}_confusion_matrix.png"
    matrix = plot_confusion_matrix(
        all_labels,
        all_predictions,
        class_names,
        confusion_matrix_path,
    )

    report_path = reports_dir / f"{output_tag}_classification_report.txt"
    with report_path.open("w", encoding="utf-8") as file:
        file.write(report_text)

    metrics_path = reports_dir / f"{output_tag}_test_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "classification_report": report_dict,
                "confusion_matrix": matrix.tolist(),
                "checkpoint": str(args.checkpoint.resolve()),
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test macro F1: {macro_f1:.4f}")
    print(f"Classification report saved to: {report_path}")
    print(f"Confusion matrix saved to: {confusion_matrix_path}")


if __name__ == "__main__":
    main()
