from __future__ import annotations

import fnmatch
from pathlib import Path


def iter_genre_dirs(raw_audio_dir: Path) -> list[Path]:
    return sorted((path for path in raw_audio_dir.iterdir() if path.is_dir()), key=lambda p: p.name)


def build_label_mapping(raw_audio_dir: Path) -> dict[str, int]:
    return {genre_dir.name: idx for idx, genre_dir in enumerate(iter_genre_dirs(raw_audio_dir))}


def matches_any_pattern(filename: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(filename, pattern) for pattern in patterns)


def is_allowed_audio(file_path: Path, allowed_extensions: list[str], ignore_patterns: list[str]) -> tuple[bool, str]:
    suffix = file_path.suffix.lower()
    if matches_any_pattern(file_path.name, ignore_patterns):
        return False, "ignored_by_pattern"
    if suffix not in {ext.lower() for ext in allowed_extensions}:
        return False, "unsupported_extension"
    return True, "valid_audio"


def make_song_id(file_path: Path) -> str:
    return file_path.stem


def expected_processed_path(processed_audio_dir: Path, split: str, genre: str, song_id: str) -> Path:
    return processed_audio_dir / split / genre / f"{song_id}.wav"


def expected_segment_dir(segments_dir: Path, split: str, genre: str, song_id: str) -> Path:
    return segments_dir / split / genre / song_id


def expected_segment_path(
    segments_dir: Path,
    split: str,
    genre: str,
    song_id: str,
    segment_index: int,
) -> Path:
    return expected_segment_dir(segments_dir, split, genre, song_id) / f"segment_{segment_index:02d}.wav"


def expected_segment_embedding_path(
    segment_embeddings_dir: Path,
    split: str,
    genre: str,
    song_id: str,
    segment_index: int,
    extension: str = ".pt",
) -> Path:
    return segment_embeddings_dir / split / genre / song_id / f"segment_{segment_index:02d}{extension}"


def expected_song_embedding_path(
    song_embeddings_dir: Path,
    split: str,
    genre: str,
    song_id: str,
    extension: str = ".pt",
    variant: str | None = None,
) -> Path:
    if variant:
        return song_embeddings_dir / variant / split / genre / f"{song_id}{extension}"
    return song_embeddings_dir / split / genre / f"{song_id}{extension}"
