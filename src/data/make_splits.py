from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create song-level train/val/test splits from the raw inventory.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the YAML config file.")
    parser.add_argument(
        "--inventory",
        type=Path,
        default=None,
        help="Optional override for raw_inventory.csv. Defaults to data/splits/raw_inventory.csv.",
    )
    return parser.parse_args()


def compute_counts(total: int, train_ratio: float, val_ratio: float) -> tuple[int, int, int]:
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    test_count = total - train_count - val_count

    if total >= 3:
        if train_count == 0:
            train_count = 1
            test_count -= 1
        if val_count == 0:
            val_count = 1
            test_count -= 1
        if test_count == 0:
            test_count = 1
            train_count -= 1

    return train_count, val_count, test_count


def main() -> None:
    args = parse_args()

    from src.data.dataset_utils import expected_processed_path
    from src.utils.config import load_config
    from src.utils.io import read_csv_rows, write_csv_rows
    from src.utils.seed import set_seed

    config = load_config(args.config)
    set_seed(config["project"]["seed"])
    splits_dir = config["paths"]["splits_dir"]
    processed_audio_dir = config["paths"]["processed_audio_dir"]
    inventory_path = args.inventory.resolve() if args.inventory else splits_dir / "raw_inventory.csv"

    rows = read_csv_rows(inventory_path)
    valid_rows = [row for row in rows if row["is_valid_audio"].lower() == "true"]
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in valid_rows:
        grouped[row["genre"]].append(row)

    rng = random.Random(config["project"]["seed"])
    split_rows = {"train": [], "val": [], "test": []}

    for genre, genre_rows in sorted(grouped.items()):
        shuffled_rows = genre_rows[:]
        rng.shuffle(shuffled_rows)
        train_count, val_count, _ = compute_counts(
            len(shuffled_rows),
            config["split"]["train_ratio"],
            config["split"]["val_ratio"],
        )

        train_rows = shuffled_rows[:train_count]
        val_rows = shuffled_rows[train_count : train_count + val_count]
        test_rows = shuffled_rows[train_count + val_count :]

        for split_name, current_rows in (
            ("train", train_rows),
            ("val", val_rows),
            ("test", test_rows),
        ):
            for row in sorted(current_rows, key=lambda item: item["song_id"]):
                split_rows[split_name].append(
                    {
                        "song_id": row["song_id"],
                        "genre": row["genre"],
                        "split": split_name,
                        "raw_path": row["source_path"],
                        "raw_sample_rate": row["sample_rate"],
                        "raw_duration_sec": row["duration_sec"],
                        "processed_path": str(
                            expected_processed_path(
                                processed_audio_dir=processed_audio_dir,
                                split=split_name,
                                genre=row["genre"],
                                song_id=row["song_id"],
                            ).resolve()
                        ),
                    }
                )

    fieldnames = [
        "song_id",
        "genre",
        "split",
        "raw_path",
        "raw_sample_rate",
        "raw_duration_sec",
        "processed_path",
    ]
    for split_name, current_rows in split_rows.items():
        output_path = splits_dir / f"{split_name}.csv"
        write_csv_rows(output_path, current_rows, fieldnames=fieldnames)
        print(f"Wrote {len(current_rows)} rows to {output_path}")

    all_song_ids = [row["song_id"] for rows_ in split_rows.values() for row in rows_]
    if len(all_song_ids) != len(set(all_song_ids)):
        raise RuntimeError("Duplicate song_id detected across splits.")


if __name__ == "__main__":
    main()
