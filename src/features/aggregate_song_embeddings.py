from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate segment-level embeddings into song-level embeddings.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the YAML config file.")
    parser.add_argument("--split", choices=["train", "val", "test"], required=True, help="Dataset split to process.")
    parser.add_argument(
        "--layer-index",
        type=int,
        default=None,
        help="Optional transformer layer index to aggregate instead of the default embedding.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing song embeddings.")
    parser.add_argument("--max-songs", type=int, default=None, help="Optional limit for debugging.")
    return parser.parse_args()


def aggregate_embeddings(embeddings: list[torch.Tensor], pooling: str) -> torch.Tensor:
    stacked = torch.stack(embeddings, dim=0)
    if pooling != "mean":
        raise ValueError(f"Unsupported song pooling: {pooling}")
    return stacked.mean(dim=0)


def resolve_embedding_variant(layer_index: int | None) -> str | None:
    if layer_index is None:
        return None
    return f"layer_{layer_index:02d}"


def select_segment_embedding(payload: dict[str, object], layer_index: int | None) -> torch.Tensor:
    if layer_index is None:
        return payload["embedding"]

    layer_indices = [int(value) for value in payload.get("layer_indices", [])]
    if not layer_indices:
        raise KeyError("This segment embedding file does not contain per-layer embeddings.")
    if layer_index not in layer_indices:
        raise ValueError(f"Requested layer {layer_index} is not available. Found: {layer_indices}")

    layer_position = layer_indices.index(layer_index)
    layer_embeddings = payload["layer_embeddings"]
    return layer_embeddings[layer_position]


def main() -> None:
    args = parse_args()

    from src.data.dataset_utils import expected_song_embedding_path
    from src.utils.config import load_config
    from src.utils.io import read_csv_rows, write_csv_rows
    from src.utils.seed import set_seed

    config = load_config(args.config)
    set_seed(config["project"]["seed"])

    rows = read_csv_rows(config["paths"]["splits_dir"] / f"{args.split}_segment_embeddings.csv")
    grouped_rows: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped_rows[(row["genre"], row["song_id"])].append(row)

    grouped_items = sorted(grouped_rows.items(), key=lambda item: (item[0][0], item[0][1]))
    if args.max_songs is not None:
        grouped_items = grouped_items[: args.max_songs]

    expected_segments = config["audio"]["segments_per_song"]
    save_extension = config["feature_extraction"]["save_extension"]
    variant = resolve_embedding_variant(args.layer_index)
    output_rows: list[dict[str, str | int]] = []

    for (genre, song_id), song_rows in tqdm(grouped_items, desc=f"aggregate-{args.split}", unit="song"):
        song_rows = sorted(song_rows, key=lambda item: int(item["segment_index"]))
        if len(song_rows) != expected_segments:
            raise ValueError(f"{song_id} has {len(song_rows)} segment embeddings, expected {expected_segments}")

        output_path = expected_song_embedding_path(
            config["paths"]["song_embeddings_dir"],
            args.split,
            genre,
            song_id,
            extension=save_extension,
            variant=variant,
        )
        if output_path.exists() and not args.overwrite:
            output_rows.append(
                {
                    "song_id": song_id,
                    "genre": genre,
                    "split": args.split,
                    "num_segments": len(song_rows),
                    "embedding_variant": variant or "default",
                    "song_embedding_path": str(output_path.resolve()),
                }
            )
            continue

        embeddings = [
            select_segment_embedding(torch.load(row["embedding_path"], map_location="cpu"), args.layer_index).to(
                torch.float32
            )
            for row in song_rows
        ]
        song_embedding = aggregate_embeddings(embeddings, config["model"]["song_pooling"]).to(torch.float32)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "embedding": song_embedding,
                "song_id": song_id,
                "genre": genre,
                "split": args.split,
                "num_segments": len(song_rows),
                "model_id": config["model"]["hf_model_id"],
                "song_pooling": config["model"]["song_pooling"],
                "embedding_variant": variant or "default",
                "layer_index": args.layer_index,
            },
            output_path,
        )
        output_rows.append(
            {
                "song_id": song_id,
                "genre": genre,
                "split": args.split,
                "num_segments": len(song_rows),
                "embedding_variant": variant or "default",
                "song_embedding_path": str(output_path.resolve()),
            }
        )

    suffix = "" if variant is None else f"_{variant}"
    output_index = config["paths"]["splits_dir"] / f"{args.split}_song_embeddings{suffix}.csv"
    write_csv_rows(
        output_index,
        output_rows,
        fieldnames=["song_id", "genre", "split", "num_segments", "embedding_variant", "song_embedding_path"],
    )
    print(f"Song embedding index written to: {output_index}")


if __name__ == "__main__":
    main()
