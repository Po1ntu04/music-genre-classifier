from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate song-level metadata from split CSV files.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the YAML config file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from src.data.dataset_utils import expected_segment_dir
    from src.utils.config import load_config
    from src.utils.io import read_csv_rows, write_csv_rows
    from src.utils.seed import set_seed

    config = load_config(args.config)
    set_seed(config["project"]["seed"])

    rows: list[dict[str, str | int]] = []
    for split_name in ("train", "val", "test"):
        split_rows = read_csv_rows(config["paths"]["splits_dir"] / f"{split_name}.csv")
        for row in split_rows:
            rows.append(
                {
                    "song_id": row["song_id"],
                    "genre": row["genre"],
                    "split": split_name,
                    "raw_path": row["raw_path"],
                    "processed_path": row["processed_path"],
                    "segment_dir": str(
                        expected_segment_dir(
                            config["paths"]["segments_dir"],
                            split_name,
                            row["genre"],
                            row["song_id"],
                        ).resolve()
                    ),
                    "sample_rate": config["audio"]["sample_rate"],
                    "song_duration_sec": config["audio"]["song_duration_sec"],
                    "segment_duration_sec": config["audio"]["segment_duration_sec"],
                    "segments_per_song": config["audio"]["segments_per_song"],
                }
            )

    output_path = config["paths"]["splits_dir"] / "song_metadata.csv"
    write_csv_rows(
        output_path,
        rows,
        fieldnames=[
            "song_id",
            "genre",
            "split",
            "raw_path",
            "processed_path",
            "segment_dir",
            "sample_rate",
            "song_duration_sec",
            "segment_duration_sec",
            "segments_per_song",
        ],
    )
    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
