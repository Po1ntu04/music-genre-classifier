from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan raw GTZAN audio files and generate an inventory CSV.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the YAML config file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional override for the inventory CSV path. Defaults to data/splits/raw_inventory.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from src.data.dataset_utils import iter_genre_dirs, is_allowed_audio, make_song_id
    from src.utils.audio import load_wav
    from src.utils.config import load_config
    from src.utils.io import ensure_dir, write_csv_rows, write_json
    from src.utils.seed import set_seed

    config = load_config(args.config)
    set_seed(config["project"]["seed"])
    raw_audio_dir = config["paths"]["raw_audio_dir"]
    splits_dir = config["paths"]["splits_dir"]
    output_path = args.output.resolve() if args.output else splits_dir / "raw_inventory.csv"

    allowed_extensions = config["audio"]["allowed_extensions"]
    ignore_patterns = config["audio"]["ignore_patterns"]

    rows: list[dict[str, str | int | float | bool]] = []
    summary = {
        "num_genres": 0,
        "num_valid_audio": 0,
        "num_invalid_files": 0,
        "genres": {},
    }

    for genre_dir in iter_genre_dirs(raw_audio_dir):
        summary["num_genres"] += 1
        genre_valid = 0
        genre_invalid = 0
        for file_path in sorted((path for path in genre_dir.iterdir() if path.is_file()), key=lambda p: p.name):
            is_valid_audio, note = is_allowed_audio(file_path, allowed_extensions, ignore_patterns)
            sample_rate = ""
            num_channels = ""
            num_samples = ""
            duration_sec = ""
            if is_valid_audio:
                try:
                    waveform, sample_rate = load_wav(file_path)
                    num_channels = int(waveform.shape[0])
                    num_samples = int(waveform.shape[1])
                    duration_sec = round(num_samples / float(sample_rate), 6)
                    genre_valid += 1
                    summary["num_valid_audio"] += 1
                except Exception as exc:
                    is_valid_audio = False
                    note = f"load_failed:{type(exc).__name__}"
                    genre_invalid += 1
                    summary["num_invalid_files"] += 1
            else:
                genre_invalid += 1
                summary["num_invalid_files"] += 1
            rows.append(
                {
                    "song_id": make_song_id(file_path),
                    "genre": genre_dir.name,
                    "source_path": str(file_path.resolve()),
                    "extension": file_path.suffix.lower(),
                    "sample_rate": sample_rate,
                    "num_channels": num_channels,
                    "num_samples": num_samples,
                    "duration_sec": duration_sec,
                    "is_valid_audio": is_valid_audio,
                    "note": note,
                }
            )
        summary["genres"][genre_dir.name] = {"valid_audio": genre_valid, "invalid_files": genre_invalid}

    ensure_dir(output_path.parent)
    write_csv_rows(
        output_path,
        rows,
        fieldnames=[
            "song_id",
            "genre",
            "source_path",
            "extension",
            "sample_rate",
            "num_channels",
            "num_samples",
            "duration_sec",
            "is_valid_audio",
            "note",
        ],
    )
    write_json(output_path.parent / "dataset_summary.json", summary)

    valid_count = sum(1 for row in rows if row["is_valid_audio"])
    invalid_count = len(rows) - valid_count
    print(f"Scanned {len(rows)} files from {raw_audio_dir}")
    print(f"Valid audio files: {valid_count}")
    print(f"Excluded files: {invalid_count}")
    print(f"Inventory written to: {output_path}")


if __name__ == "__main__":
    main()
