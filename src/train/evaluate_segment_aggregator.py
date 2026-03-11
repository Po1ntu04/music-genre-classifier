from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.train.train_segment_aggregator import SongSegmentEmbeddingDataset, collate_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a segment-aggregator classifier on the test split.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the YAML config file.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the checkpoint produced by training.")
    parser.add_argument(
        "--eval-mode",
        choices=["embeddings", "audio"],
        default="embeddings",
        help="Evaluate from cached segment embeddings or from processed audio files.",
    )
    parser.add_argument(
        "--model-source",
        choices=["offline", "online"],
        default="offline",
        help="MERT source when eval-mode=audio.",
    )
    parser.add_argument(
        "--tta-mode",
        choices=["none", "dual", "random", "dual_random"],
        default="none",
        help="Inference-time view strategy when eval-mode=audio.",
    )
    parser.add_argument(
        "--tta-random-views",
        type=int,
        default=2,
        help="Number of extra random views when tta-mode includes random crops.",
    )
    parser.add_argument(
        "--tta-seed",
        type=int,
        default=None,
        help="Random seed used by TTA random crops. Defaults to project seed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from src.models.inference_utils import predict_probabilities_from_audio
    from src.models.segment_aggregator import build_segment_classifier
    from src.train.metrics import build_classification_report, plot_confusion_matrix
    from src.utils.config import load_config
    from src.utils.io import read_csv_rows
    from src.utils.runtime import resolve_device
    from src.utils.seed import set_seed

    config = load_config(args.config)
    set_seed(config["project"]["seed"])
    device = resolve_device(config["project"]["device"])

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    class_names = checkpoint["class_names"]
    label_to_index = {name: idx for idx, name in enumerate(class_names)}

    all_labels: list[int] = []
    all_predictions: list[int] = []

    if args.eval_mode == "embeddings":
        test_rows = read_csv_rows(config["paths"]["splits_dir"] / "test_segment_embeddings.csv")
        dataset = SongSegmentEmbeddingDataset(
            test_rows,
            label_to_index=label_to_index,
            embedding_suffix=checkpoint["embedding_suffix"],
        )
        dataloader = DataLoader(
            dataset,
            batch_size=config["segment_training"]["batch_size"],
            shuffle=False,
            collate_fn=collate_batch,
        )

        model = build_segment_classifier(
            aggregation_type=checkpoint["aggregation_type"],
            input_dim=checkpoint["input_dim"],
            num_classes=checkpoint["num_classes"],
            model_type=checkpoint["model_type"],
            hidden_dim=checkpoint["hidden_dim"],
            dropout=checkpoint["dropout"],
            chunk_dropout_prob=0.0,
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        with torch.no_grad():
            for embeddings, labels in tqdm(dataloader, desc="test", unit="batch"):
                embeddings = embeddings.to(device)
                logits, _ = model(embeddings)
                predictions = logits.argmax(dim=1).cpu().tolist()
                all_predictions.extend(predictions)
                all_labels.extend(labels.tolist())
    else:
        test_rows = read_csv_rows(config["paths"]["splits_dir"] / "test.csv")
        for row in tqdm(test_rows, desc="test-audio", unit="song"):
            result = predict_probabilities_from_audio(
                audio_path=row["processed_path"],
                config=config,
                checkpoint_path=args.checkpoint,
                model_source=args.model_source,
                tta_mode=args.tta_mode,
                random_views=args.tta_random_views,
                tta_seed=args.tta_seed,
            )
            prediction = int(result["song_probabilities"].argmax().item())
            all_predictions.append(prediction)
            all_labels.append(label_to_index[row["genre"]])

    accuracy = float(accuracy_score(all_labels, all_predictions))
    macro_f1 = float(f1_score(all_labels, all_predictions, average="macro"))
    report_text, report_dict = build_classification_report(all_labels, all_predictions, class_names)

    figures_dir = config["paths"]["outputs_dir"] / "figures"
    reports_dir = config["paths"]["outputs_dir"] / "reports"
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    output_tag = args.checkpoint.stem.replace("best_", "")
    if args.eval_mode == "audio":
        output_tag += f"_audio_{args.tta_mode}"
        if args.tta_mode in {"random", "dual_random"}:
            output_tag += f"_rv{args.tta_random_views}"
    confusion_matrix_path = figures_dir / f"{output_tag}_confusion_matrix.png"
    matrix = plot_confusion_matrix(all_labels, all_predictions, class_names, confusion_matrix_path)

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
                "eval_mode": args.eval_mode,
                "tta_mode": args.tta_mode if args.eval_mode == "audio" else "none",
                "tta_random_views": args.tta_random_views if args.eval_mode == "audio" else 0,
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
