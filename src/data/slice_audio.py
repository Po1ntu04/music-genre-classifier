from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Slice standardized songs into fixed 5-second segments.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the YAML config file.")
    parser.add_argument("--split", choices=["train", "val", "test"], required=True, help="Dataset split to process.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing segment files.")
    parser.add_argument("--max-items", type=int, default=None, help="Optional limit for debugging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from tqdm import tqdm

    from src.data.dataset_utils import expected_segment_path
    from src.utils.audio import load_wav, pad_or_trim, save_wav
    from src.utils.config import load_config
    from src.utils.io import ensure_dir, read_csv_rows, write_csv_rows
    from src.utils.seed import set_seed

    config = load_config(args.config)
    set_seed(config["project"]["seed"])
    sample_rate = config["audio"]["sample_rate"]
    song_num_samples = sample_rate * config["audio"]["song_duration_sec"]
    segment_num_samples = sample_rate * config["audio"]["segment_duration_sec"]
    segments_per_song = config["audio"]["segments_per_song"]

    split_csv = config["paths"]["splits_dir"] / f"{args.split}.csv"
    rows = read_csv_rows(split_csv)
    if args.max_items is not None:
        rows = rows[: args.max_items]

    segment_rows: list[dict[str, str | int]] = []
    for row in tqdm(rows, desc=f"slice-{args.split}", unit="song"):
        processed_path = Path(row["processed_path"])
        waveform, current_sr = load_wav(processed_path)
        if current_sr != sample_rate:
            raise ValueError(f"Expected {sample_rate} Hz audio, got {current_sr} for {processed_path}")

        waveform = pad_or_trim(waveform, song_num_samples)

        for segment_idx in range(segments_per_song):
            start = segment_idx * segment_num_samples
            end = start + segment_num_samples
            segment_waveform = waveform[:, start:end]
            segment_path = expected_segment_path(
                config["paths"]["segments_dir"],
                args.split,
                row["genre"],
                row["song_id"],
                segment_idx,
            )
            ensure_dir(segment_path.parent)
            if not (segment_path.exists() and not args.overwrite):
                save_wav(segment_path, segment_waveform, sample_rate)
            segment_rows.append(
                {
                    "song_id": row["song_id"],
                    "genre": row["genre"],
                    "split": args.split,
                    "segment_index": segment_idx,
                    "segment_path": str(segment_path.resolve()),
                    "sample_rate": sample_rate,
                    "segment_duration_sec": config["audio"]["segment_duration_sec"],
                    "num_samples": segment_num_samples,
                }
            )

    segment_metadata_path = config["paths"]["splits_dir"] / f"{args.split}_segments.csv"
    write_csv_rows(
        segment_metadata_path,
        segment_rows,
        fieldnames=[
            "song_id",
            "genre",
            "split",
            "segment_index",
            "segment_path",
            "sample_rate",
            "segment_duration_sec",
            "num_samples",
        ],
    )
    print(f"Finished slicing split: {args.split}")
    print(f"Segment metadata written to: {segment_metadata_path}")


if __name__ == "__main__":
    main()
