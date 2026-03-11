from __future__ import annotations

import argparse
import csv
import random
import shutil
import zipfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_input = script_dir / "datasets_10classes" / "Data" / "genres_original"
    default_output = script_dir / "datasets_10classes" / "Data" / "genres_split"

    parser = argparse.ArgumentParser(
        description=(
            "Split genres_original into train/test with an 8:2 ratio. "
            "Training samples are stored as one zip file per genre, while "
            "test samples stay uncompressed."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=default_input,
        help="Path to genres_original.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help="Directory used to store the split result.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train split ratio for each genre.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for deterministic splitting.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output-dir if it already exists.",
    )
    return parser.parse_args()


def validate_args(input_dir: Path, output_dir: Path, train_ratio: float, force: bool) -> None:
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train-ratio must be between 0 and 1, got {train_ratio}")

    if output_dir.exists():
        if not force:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}\n"
                "Re-run with --force to overwrite it."
            )
        shutil.rmtree(output_dir)


def collect_class_dirs(input_dir: Path) -> list[Path]:
    class_dirs = [path for path in input_dir.iterdir() if path.is_dir()]
    if not class_dirs:
        raise RuntimeError(f"No class directories found under: {input_dir}")
    return sorted(class_dirs, key=lambda path: path.name)


def split_files(files: list[Path], train_ratio: float, rng: random.Random) -> tuple[list[Path], list[Path]]:
    shuffled = files[:]
    rng.shuffle(shuffled)

    train_count = int(len(shuffled) * train_ratio)
    if len(shuffled) > 1:
        train_count = max(1, min(train_count, len(shuffled) - 1))

    train_files = sorted(shuffled[:train_count], key=lambda path: path.name)
    test_files = sorted(shuffled[train_count:], key=lambda path: path.name)
    return train_files, test_files


def copy_test_files(test_files: list[Path], class_name: str, test_root: Path) -> None:
    target_dir = test_root / class_name
    target_dir.mkdir(parents=True, exist_ok=True)

    for src_file in test_files:
        shutil.copy2(src_file, target_dir / src_file.name)


def zip_train_files(train_files: list[Path], class_name: str, train_zip_root: Path) -> Path:
    train_zip_root.mkdir(parents=True, exist_ok=True)
    zip_path = train_zip_root / f"{class_name}.zip"

    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        for src_file in train_files:
            arcname = Path(class_name) / src_file.name
            zip_file.write(src_file, arcname=arcname)

    return zip_path


def write_manifest(rows: list[dict[str, str]], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["genre", "split", "filename"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    validate_args(input_dir, output_dir, args.train_ratio, args.force)

    rng = random.Random(args.seed)
    class_dirs = collect_class_dirs(input_dir)
    test_root = output_dir / "test"
    train_zip_root = output_dir / "train_zip"
    manifest_rows: list[dict[str, str]] = []
    summary_lines: list[str] = []

    for class_dir in class_dirs:
        audio_files = sorted(path for path in class_dir.iterdir() if path.is_file())
        if not audio_files:
            continue

        train_files, test_files = split_files(audio_files, args.train_ratio, rng)
        copy_test_files(test_files, class_dir.name, test_root)
        zip_path = zip_train_files(train_files, class_dir.name, train_zip_root)

        for file_path in train_files:
            manifest_rows.append(
                {"genre": class_dir.name, "split": "train", "filename": file_path.name}
            )
        for file_path in test_files:
            manifest_rows.append(
                {"genre": class_dir.name, "split": "test", "filename": file_path.name}
            )

        summary_lines.append(
            f"{class_dir.name}: train={len(train_files)}, test={len(test_files)}, zip={zip_path.name}"
        )

    write_manifest(manifest_rows, output_dir / "split_manifest.csv")

    summary_path = output_dir / "README.txt"
    summary_path.write_text(
        "\n".join(
            [
                f"Input: {input_dir}",
                f"Output: {output_dir}",
                f"Train ratio: {args.train_ratio}",
                f"Seed: {args.seed}",
                "",
                "Per-class summary:",
                *summary_lines,
            ]
        ),
        encoding="utf-8",
    )

    print(f"Split complete. Output written to: {output_dir}")
    for line in summary_lines:
        print(line)


if __name__ == "__main__":
    main()
