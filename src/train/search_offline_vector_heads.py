from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search offline vector classification heads on frozen song vectors.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the YAML config file.")
    parser.add_argument(
        "--layer-suffixes",
        nargs="+",
        default=["layer_04", "layer_06", "layer_07", "layer_08", "layer_10"],
        help="Candidate transformer layers to evaluate.",
    )
    parser.add_argument(
        "--aggregations",
        nargs="+",
        default=["mean", "mean_std"],
        choices=["mean", "mean_std"],
        help="Song-level vector aggregation choices built from segment embeddings.",
    )
    return parser.parse_args()


def build_song_vectors(
    rows: list[dict[str, str]],
    layer_index: int,
    aggregation: str,
) -> tuple[np.ndarray, list[str], list[str]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["genre"], row["song_id"])].append(row)

    features = []
    labels = []
    song_ids = []
    for (genre, song_id), song_rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        segment_embeddings = []
        for row in sorted(song_rows, key=lambda item: int(item["segment_index"])):
            payload = torch.load(row["embedding_path"], map_location="cpu")
            layer_indices = [int(value) for value in payload["layer_indices"]]
            layer_position = layer_indices.index(layer_index)
            segment_embeddings.append(payload["layer_embeddings"][layer_position].to(torch.float32).numpy())
        stacked = np.stack(segment_embeddings, axis=0)
        mean_vector = stacked.mean(axis=0)
        if aggregation == "mean":
            feature = mean_vector
        elif aggregation == "mean_std":
            std_vector = stacked.std(axis=0, ddof=0)
            feature = np.concatenate([mean_vector, std_vector], axis=0)
        else:
            raise ValueError(f"Unsupported aggregation: {aggregation}")
        features.append(feature.astype(np.float32))
        labels.append(genre)
        song_ids.append(song_id)
    return np.stack(features, axis=0), labels, song_ids


def make_pipeline(head_name: str, hyperparameter: float) -> Pipeline:
    if head_name == "logreg":
        classifier = LogisticRegression(
            C=hyperparameter,
            solver="lbfgs",
            multi_class="multinomial",
            max_iter=5000,
            random_state=42,
        )
    elif head_name == "linear_svm":
        classifier = LinearSVC(C=hyperparameter, random_state=42, max_iter=10000)
    elif head_name == "ridge":
        classifier = RidgeClassifier(alpha=hyperparameter)
    else:
        raise ValueError(f"Unsupported head: {head_name}")

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    )


def main() -> None:
    args = parse_args()

    from src.train.metrics import build_classification_report, plot_confusion_matrix
    from src.utils.config import load_config
    from src.utils.io import read_csv_rows

    config = load_config(args.config)
    train_rows = read_csv_rows(config["paths"]["splits_dir"] / "train_segment_embeddings.csv")
    val_rows = read_csv_rows(config["paths"]["splits_dir"] / "val_segment_embeddings.csv")
    test_rows = read_csv_rows(config["paths"]["splits_dir"] / "test_segment_embeddings.csv")
    class_names = sorted({row["genre"] for row in train_rows + val_rows + test_rows})
    label_to_index = {name: index for index, name in enumerate(class_names)}

    head_grid: dict[str, list[float]] = {
        "logreg": [0.1, 1.0, 10.0, 100.0],
        "linear_svm": [0.1, 1.0, 10.0, 100.0],
        "ridge": [0.01, 0.1, 1.0, 10.0],
    }

    experiment_rows: list[dict[str, object]] = []
    best_by_head: dict[str, dict[str, object]] = {}
    best_overall: dict[str, object] | None = None

    for layer_suffix in args.layer_suffixes:
        layer_index = int(layer_suffix.split("_")[-1])
        for aggregation in args.aggregations:
            train_x, train_labels, _ = build_song_vectors(train_rows, layer_index=layer_index, aggregation=aggregation)
            val_x, val_labels, _ = build_song_vectors(val_rows, layer_index=layer_index, aggregation=aggregation)
            test_x, test_labels, _ = build_song_vectors(test_rows, layer_index=layer_index, aggregation=aggregation)

            train_y = np.array([label_to_index[label] for label in train_labels], dtype=np.int64)
            val_y = np.array([label_to_index[label] for label in val_labels], dtype=np.int64)
            test_y = np.array([label_to_index[label] for label in test_labels], dtype=np.int64)

            for head_name, hyperparameters in head_grid.items():
                for hyperparameter in hyperparameters:
                    pipeline = make_pipeline(head_name=head_name, hyperparameter=hyperparameter)
                    pipeline.fit(train_x, train_y)
                    val_predictions = pipeline.predict(val_x)
                    test_predictions = pipeline.predict(test_x)

                    row = {
                        "head_name": head_name,
                        "hyperparameter": float(hyperparameter),
                        "layer_suffix": layer_suffix,
                        "aggregation": aggregation,
                        "val_accuracy": float(accuracy_score(val_y, val_predictions)),
                        "val_macro_f1": float(f1_score(val_y, val_predictions, average="macro")),
                        "test_accuracy": float(accuracy_score(test_y, test_predictions)),
                        "test_macro_f1": float(f1_score(test_y, test_predictions, average="macro")),
                    }
                    experiment_rows.append(row)

                    current_best = best_by_head.get(head_name)
                    if current_best is None or row["val_macro_f1"] > current_best["val_macro_f1"]:
                        best_by_head[head_name] = row
                    if best_overall is None or row["val_macro_f1"] > best_overall["val_macro_f1"]:
                        best_overall = row

    experiment_rows.sort(
        key=lambda item: (
            item["head_name"],
            -float(item["val_macro_f1"]),
            -float(item["test_macro_f1"]),
            item["layer_suffix"],
            item["aggregation"],
        )
    )

    reports_dir = config["paths"]["outputs_dir"] / "reports"
    figures_dir = config["paths"]["outputs_dir"] / "figures"
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = reports_dir / "offline_vector_head_search_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "head_name",
                "hyperparameter",
                "layer_suffix",
                "aggregation",
                "val_accuracy",
                "val_macro_f1",
                "test_accuracy",
                "test_macro_f1",
            ],
        )
        writer.writeheader()
        writer.writerows(experiment_rows)

    assert best_overall is not None
    best_layer_index = int(str(best_overall["layer_suffix"]).split("_")[-1])
    best_train_x, best_train_labels, _ = build_song_vectors(
        train_rows, layer_index=best_layer_index, aggregation=str(best_overall["aggregation"])
    )
    best_val_x, best_val_labels, _ = build_song_vectors(
        val_rows, layer_index=best_layer_index, aggregation=str(best_overall["aggregation"])
    )
    best_test_x, best_test_labels, _ = build_song_vectors(
        test_rows, layer_index=best_layer_index, aggregation=str(best_overall["aggregation"])
    )
    combined_x = np.concatenate([best_train_x, best_val_x], axis=0)
    combined_y = np.array(
        [label_to_index[label] for label in best_train_labels + best_val_labels],
        dtype=np.int64,
    )
    test_y = np.array([label_to_index[label] for label in best_test_labels], dtype=np.int64)
    best_pipeline = make_pipeline(
        head_name=str(best_overall["head_name"]),
        hyperparameter=float(best_overall["hyperparameter"]),
    )
    best_pipeline.fit(combined_x, combined_y)
    best_test_predictions = best_pipeline.predict(best_test_x)

    report_text, report_dict = build_classification_report(
        test_y.tolist(),
        best_test_predictions.tolist(),
        class_names,
    )
    best_tag = (
        f"offline_{best_overall['head_name']}_{best_overall['layer_suffix']}_{best_overall['aggregation']}"
        f"_h{str(best_overall['hyperparameter']).replace('.', 'p')}"
    )
    report_path = reports_dir / f"{best_tag}_classification_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    matrix = plot_confusion_matrix(
        test_y.tolist(),
        best_test_predictions.tolist(),
        class_names,
        figures_dir / f"{best_tag}_confusion_matrix.png",
    )
    metrics_path = reports_dir / f"{best_tag}_test_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "accuracy": float(accuracy_score(test_y, best_test_predictions)),
                "macro_f1": float(f1_score(test_y, best_test_predictions, average="macro")),
                "classification_report": report_dict,
                "confusion_matrix": matrix.tolist(),
                "selection_record": best_overall,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    summary_lines = [
        "# Offline Vector Head Search Summary",
        "",
        "本阶段在冻结的歌曲向量上搜索轻量分类头，比较 `logistic regression`、`linear SVM` 和 `ridge classifier`。",
        "",
        "## Search Space",
        "",
        f"- Layers: `{', '.join(args.layer_suffixes)}`",
        f"- Aggregations: `{', '.join(args.aggregations)}`",
        "- Heads: `logreg`, `linear_svm`, `ridge`",
        "",
        "## Best By Head",
        "",
        "| Head | Layer | Aggregation | Hyperparameter | Val F1 | Test F1 |",
        "|---|---|---|---:|---:|---:|",
    ]
    for head_name in ["logreg", "linear_svm", "ridge"]:
        row = best_by_head[head_name]
        summary_lines.append(
            f"| {head_name} | {row['layer_suffix']} | {row['aggregation']} | {row['hyperparameter']:.2f} | "
            f"{row['val_macro_f1']:.4f} | {row['test_macro_f1']:.4f} |"
        )
    summary_lines.extend(
        [
            "",
            "## Overall Winner",
            "",
            f"- Head: `{best_overall['head_name']}`",
            f"- Layer: `{best_overall['layer_suffix']}`",
            f"- Aggregation: `{best_overall['aggregation']}`",
            f"- Hyperparameter: `{best_overall['hyperparameter']}`",
            f"- Validation macro F1: `{best_overall['val_macro_f1']:.4f}`",
            f"- Test macro F1 after retraining on train+val: `{float(f1_score(test_y, best_test_predictions, average='macro')):.4f}`",
            "",
            "## Interpretation",
            "",
            "- 这个搜索回答的是：在固定冻结特征上，传统线性分类头是否能追上当前的分段 MLP 正式模型。",
            "- 如果最优传统头仍落后于当前正式模型，那么后续应优先保留分段聚合路线，把经典线性头当作强基线而不是正式替代方案。",
        ]
    )
    summary_path = reports_dir / "offline_vector_head_search_summary.md"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"Search results saved to: {csv_path}")
    print(f"Summary saved to: {summary_path}")
    print(f"Best overall: {best_overall}")


if __name__ == "__main__":
    main()
