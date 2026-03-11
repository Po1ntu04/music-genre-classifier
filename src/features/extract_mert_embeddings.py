from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frozen MERT embeddings for 5-second audio segments.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the YAML config file.")
    parser.add_argument("--split", choices=["train", "val", "test"], required=True, help="Dataset split to process.")
    parser.add_argument(
        "--model-source",
        choices=["offline", "online"],
        default="offline",
        help="Choose a local model directory or Hugging Face download.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing embeddings.")
    parser.add_argument("--max-songs", type=int, default=None, help="Optional limit for debugging.")
    return parser.parse_args()


def pool_hidden_states(hidden_states: torch.Tensor, pooling: str) -> torch.Tensor:
    if pooling != "mean":
        raise ValueError(f"Unsupported segment pooling: {pooling}")
    return hidden_states.mean(dim=1)


def main() -> None:
    args = parse_args()

    from src.data.dataset_utils import expected_segment_embedding_path
    from src.utils.audio import load_wav
    from src.utils.config import load_config
    from src.utils.io import read_csv_rows, write_csv_rows
    from src.utils.runtime import resolve_device
    from src.utils.seed import set_seed

    config = load_config(args.config)
    set_seed(config["project"]["seed"])
    device = resolve_device(config["project"]["device"])

    hf_cache_dir = config["paths"]["hf_cache_dir"]
    hf_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_cache_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(hf_cache_dir / "transformers")
    os.environ["HF_MODULES_CACHE"] = str(hf_cache_dir / "modules")

    from transformers import AutoFeatureExtractor, AutoModel

    segment_rows = read_csv_rows(config["paths"]["splits_dir"] / f"{args.split}_segments.csv")
    grouped_rows: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in segment_rows:
        grouped_rows[(row["genre"], row["song_id"])].append(row)

    grouped_items = sorted(grouped_rows.items(), key=lambda item: (item[0][0], item[0][1]))
    if args.max_songs is not None:
        grouped_items = grouped_items[: args.max_songs]

    model_path_or_id = (
        str(config["paths"]["local_model_dir"])
        if args.model_source == "offline"
        else config["model"]["hf_model_id"]
    )
    local_files_only = args.model_source == "offline"

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_path_or_id,
        local_files_only=local_files_only,
        trust_remote_code=config["model"]["trust_remote_code"],
    )
    model = AutoModel.from_pretrained(
        model_path_or_id,
        local_files_only=local_files_only,
        trust_remote_code=config["model"]["trust_remote_code"],
    )
    model.eval()
    model.requires_grad_(False)
    model.to(device)

    save_extension = config["feature_extraction"]["save_extension"]
    batch_size = config["feature_extraction"]["batch_size"]
    expected_segments = config["audio"]["segments_per_song"]
    layer_indices = [int(index) for index in config["feature_extraction"].get("layer_indices", [])]
    output_rows: list[dict[str, str | int]] = []

    with torch.no_grad():
        for (genre, song_id), song_rows in tqdm(grouped_items, desc=f"embed-{args.split}", unit="song"):
            song_rows = sorted(song_rows, key=lambda item: int(item["segment_index"]))
            if len(song_rows) != expected_segments:
                raise ValueError(f"{song_id} has {len(song_rows)} segments, expected {expected_segments}")

            output_paths = [
                expected_segment_embedding_path(
                    config["paths"]["segment_embeddings_dir"],
                    args.split,
                    genre,
                    song_id,
                    int(row["segment_index"]),
                    extension=save_extension,
                )
                for row in song_rows
            ]
            if all(path.exists() for path in output_paths) and not args.overwrite:
                for row, output_path in zip(song_rows, output_paths):
                    output_rows.append(
                        {
                            "song_id": song_id,
                            "genre": genre,
                            "split": args.split,
                            "segment_index": int(row["segment_index"]),
                            "segment_path": row["segment_path"],
                            "embedding_path": str(output_path.resolve()),
                        }
                    )
                continue

            audio_arrays = []
            for row in song_rows:
                waveform, sample_rate = load_wav(row["segment_path"])
                if sample_rate != config["audio"]["sample_rate"]:
                    raise ValueError(f"Unexpected sample rate {sample_rate} for {row['segment_path']}")
                audio_arrays.append(waveform.squeeze(0).numpy())

            pooled_batches = []
            for start_idx in range(0, len(audio_arrays), batch_size):
                batch_audio = audio_arrays[start_idx : start_idx + batch_size]
                inputs = feature_extractor(
                    batch_audio,
                    sampling_rate=config["audio"]["sample_rate"],
                    return_tensors="pt",
                    padding=True,
                )
                inputs = {key: value.to(device) for key, value in inputs.items()}
                outputs = model(
                    **inputs,
                    output_hidden_states=config["model"].get("output_hidden_states", True),
                )
                pooled_last_hidden = pool_hidden_states(
                    outputs.last_hidden_state,
                    config["model"]["segment_pooling"],
                ).cpu()
                if outputs.hidden_states is None or not layer_indices:
                    pooled_layers = pooled_last_hidden.unsqueeze(1)
                    active_layer_indices = [len(outputs.hidden_states or []) - 1] if outputs.hidden_states else [-1]
                else:
                    pooled_layers = torch.stack(
                        [
                            pool_hidden_states(outputs.hidden_states[layer_index], config["model"]["segment_pooling"])
                            for layer_index in layer_indices
                        ],
                        dim=1,
                    ).cpu()
                    active_layer_indices = layer_indices
                pooled_batches.append((pooled_last_hidden, pooled_layers))

            pooled = torch.cat([batch[0] for batch in pooled_batches], dim=0)
            pooled_by_layer = torch.cat([batch[1] for batch in pooled_batches], dim=0)

            for row, embedding, layer_embeddings, output_path in zip(song_rows, pooled, pooled_by_layer, output_paths):
                output_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "embedding": embedding.to(torch.float32),
                        "layer_embeddings": layer_embeddings.to(torch.float32),
                        "layer_indices": active_layer_indices,
                        "song_id": song_id,
                        "genre": genre,
                        "split": args.split,
                        "segment_index": int(row["segment_index"]),
                        "source_segment_path": row["segment_path"],
                        "model_id": model_path_or_id,
                    },
                    output_path,
                )
                output_rows.append(
                    {
                        "song_id": song_id,
                        "genre": genre,
                        "split": args.split,
                        "segment_index": int(row["segment_index"]),
                        "segment_path": row["segment_path"],
                        "embedding_path": str(output_path.resolve()),
                    }
                )

    output_index = config["paths"]["splits_dir"] / f"{args.split}_segment_embeddings.csv"
    write_csv_rows(
        output_index,
        output_rows,
        fieldnames=["song_id", "genre", "split", "segment_index", "segment_path", "embedding_path"],
    )
    print(f"Embedding index written to: {output_index}")
    print(f"Device used: {device}")
    print(f"Model source: {args.model_source} -> {model_path_or_id}")


if __name__ == "__main__":
    main()
