from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resample raw audio to 24 kHz mono and save standardized WAV files.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the YAML config file.")
    parser.add_argument("--split", choices=["train", "val", "test"], required=True, help="Dataset split to process.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing processed files.")
    parser.add_argument("--max-items", type=int, default=None, help="Optional limit for debugging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from tqdm import tqdm

    from src.utils.audio import load_wav, mix_down_to_mono, pad_or_trim, resample_waveform, save_wav
    from src.utils.config import load_config
    from src.utils.io import ensure_dir, read_csv_rows
    from src.utils.seed import set_seed

    config = load_config(args.config)
    set_seed(config["project"]["seed"])
    sample_rate = config["audio"]["sample_rate"]
    mono = config["audio"]["mono"]
    target_num_samples = sample_rate * config["audio"]["song_duration_sec"]
    split_csv = config["paths"]["splits_dir"] / f"{args.split}.csv"
    rows = read_csv_rows(split_csv)
    if args.max_items is not None:
        rows = rows[: args.max_items]

    processed_count = 0
    skipped_count = 0
    for row in tqdm(rows, desc=f"preprocess-{args.split}", unit="song"):
        source_path = Path(row["raw_path"])
        target_path = Path(row["processed_path"])
        ensure_dir(target_path.parent)

        if target_path.exists() and not args.overwrite:
            skipped_count += 1
            continue

        waveform, original_sr = load_wav(source_path)
        if mono:
            waveform = mix_down_to_mono(waveform)

        if original_sr != sample_rate:
            waveform = resample_waveform(waveform, orig_sr=original_sr, target_sr=sample_rate)

        waveform = pad_or_trim(waveform, target_num_samples)
        save_wav(target_path, waveform, sample_rate)
        processed_count += 1

    print(f"Finished preprocessing split: {args.split}")
    print(f"Processed: {processed_count}, skipped: {skipped_count}")


if __name__ == "__main__":
    main()
