from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

GENRE_COLORS = {
    "blues": "#4C78A8",
    "classical": "#8172B3",
    "country": "#9C755F",
    "disco": "#D37295",
    "hiphop": "#5B5F66",
    "jazz": "#B6992D",
    "metal": "#A34A5D",
    "pop": "#E17C9B",
    "reggae": "#59A14F",
    "rock": "#F28E2B",
}


def get_genre_color(genre: str) -> str:
    return GENRE_COLORS.get(genre, "#4C78A8")


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[index : index + 2], 16) for index in (0, 2, 4))


def _text_color_for_background(hex_color: str) -> str:
    red, green, blue = _hex_to_rgb(hex_color)
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    return "#0F172A" if luminance > 165 else "#F8FAFC"


def build_result_card_html(predicted_genre: str, confidence: float) -> str:
    genre_color = get_genre_color(predicted_genre)
    text_color = _text_color_for_background(genre_color)
    return f"""
<div style="
    background: linear-gradient(135deg, {genre_color}, #0F172A);
    border-radius: 18px;
    padding: 22px 24px;
    box-shadow: 0 12px 30px rgba(15, 23, 42, 0.18);
    border: 1px solid rgba(255,255,255,0.18);
">
  <div style="font-size: 13px; letter-spacing: 0.12em; text-transform: uppercase; color: rgba(248,250,252,0.82);">
    Predicted Genre
  </div>
  <div style="margin-top: 8px; font-size: 34px; font-weight: 800; color: {text_color};">
    {predicted_genre.upper()}
  </div>
  <div style="margin-top: 10px; font-size: 15px; color: rgba(248,250,252,0.92);">
    Top-1 Probability: <strong>{confidence * 100:.2f}%</strong>
  </div>
</div>
""".strip()


def build_topk_cards_html(topk_records: list[dict[str, float | str]]) -> str:
    card_chunks = []
    for record in topk_records:
        genre = str(record["genre"])
        probability = float(record["probability"])
        rank = int(record["rank"])
        color = get_genre_color(genre)
        card_chunks.append(
            f"""
<div style="
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:12px;
    padding:12px 14px;
    border-radius:14px;
    border:1px solid #D9E2EC;
    background:#FFFFFF;
">
  <div style="display:flex; align-items:center; gap:10px;">
    <div style="
        width:28px;
        height:28px;
        border-radius:999px;
        background:{color};
        color:{_text_color_for_background(color)};
        display:flex;
        align-items:center;
        justify-content:center;
        font-size:13px;
        font-weight:700;
    ">
      {rank}
    </div>
    <div>
      <div style="font-size:15px; font-weight:700; color:#102A43;">{genre}</div>
      <div style="font-size:12px; color:#627D98;">Top-{rank}</div>
    </div>
  </div>
  <div style="font-size:15px; font-weight:700; color:#102A43;">{probability * 100:.2f}%</div>
</div>
""".strip()
        )
    return (
        '<div style="display:flex; flex-direction:column; gap:10px;">'
        + "".join(card_chunks)
        + "</div>"
    )


def build_model_info_html(
    checkpoint_path: str,
    model_source: str,
    sample_rate: int,
    segment_duration_sec: int,
    segments_per_song: int,
    embedding_suffix: str,
    aggregation_type: str,
    tta_mode: str,
    num_views: int,
) -> str:
    items = [
        ("Checkpoint", checkpoint_path),
        ("MERT Source", model_source),
        ("Sample Rate", f"{sample_rate} Hz"),
        ("Segment Length", f"{segment_duration_sec} s"),
        ("Segments per Song", str(segments_per_song)),
        ("Feature Layer", embedding_suffix),
        ("Pooling", aggregation_type),
        ("TTA", f"{tta_mode} ({num_views} view{'s' if num_views != 1 else ''})"),
    ]
    rows = "".join(
        f"""
<div style="display:grid; grid-template-columns: 150px 1fr; gap:10px; padding:8px 0; border-bottom:1px solid #E5EAF0;">
  <div style="font-weight:700; color:#243B53;">{label}</div>
  <div style="color:#486581; word-break:break-word;">{value}</div>
</div>
""".strip()
        for label, value in items
    )
    return f'<div style="padding:4px 2px 2px 2px;">{rows}</div>'


def build_probability_bar_chart(labels: list[str], probabilities: list[float]):
    pairs = sorted(zip(labels, probabilities), key=lambda item: item[1], reverse=True)
    sorted_labels = [item[0] for item in pairs]
    sorted_probs = [item[1] for item in pairs]
    colors = [get_genre_color(label) for label in sorted_labels]

    figure_height = max(3.0, 0.42 * len(sorted_labels))
    figure, axis = plt.subplots(figsize=(8.2, figure_height))
    bars = axis.barh(sorted_labels, sorted_probs, color=colors, edgecolor="#E6EDF5")
    axis.invert_yaxis()
    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel("Probability")
    axis.set_title("Song-level Genre Probabilities", loc="left", pad=12, fontsize=12, fontweight="bold")
    axis.grid(axis="x", linestyle="--", linewidth=0.8, alpha=0.25)
    axis.set_facecolor("#FCFDFE")
    figure.patch.set_facecolor("#FCFDFE")

    for bar, probability in zip(bars, sorted_probs):
        axis.text(
            min(probability + 0.015, 0.985),
            bar.get_y() + bar.get_height() / 2.0,
            f"{probability * 100:.2f}%",
            va="center",
            ha="left",
            fontsize=10,
            color="#243B53",
            fontweight="semibold",
        )

    for spine in ("top", "right", "left"):
        axis.spines[spine].set_visible(False)
    axis.spines["bottom"].set_color("#BCCCDC")
    figure.tight_layout()
    return figure


def build_segment_timeline_chart(segment_rows: list[dict[str, float | str]]):
    figure, axis = plt.subplots(figsize=(8.2, 1.9))
    axis.set_facecolor("#FCFDFE")
    figure.patch.set_facecolor("#FCFDFE")

    for row in segment_rows:
        start_sec = float(row["start_sec"])
        end_sec = float(row["end_sec"])
        genre = str(row["predicted_genre"])
        confidence = float(row["confidence"])
        color = get_genre_color(genre)
        axis.barh(
            y=0,
            width=end_sec - start_sec,
            left=start_sec,
            height=0.62,
            color=color,
            alpha=0.28 + 0.72 * confidence,
            edgecolor="#F8FAFC",
            linewidth=1.5,
        )
        axis.text(
            start_sec + (end_sec - start_sec) / 2.0,
            0,
            f"{genre}\n{confidence * 100:.1f}%",
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="semibold",
            color=_text_color_for_background(color),
        )

    axis.set_xlim(0, max(float(row["end_sec"]) for row in segment_rows))
    axis.set_ylim(-0.7, 0.7)
    axis.set_yticks([])
    axis.set_xlabel("Time (seconds)")
    axis.set_title("Per-segment Timeline", loc="left", pad=10, fontsize=12, fontweight="bold")
    axis.set_xticks([float(row["start_sec"]) for row in segment_rows] + [float(segment_rows[-1]["end_sec"])])
    axis.grid(axis="x", linestyle="--", linewidth=0.8, alpha=0.25)
    for spine in ("top", "right", "left"):
        axis.spines[spine].set_visible(False)
    axis.spines["bottom"].set_color("#BCCCDC")
    figure.tight_layout()
    return figure


def build_audio_preview_chart(
    waveform: np.ndarray,
    sample_rate: int,
    segment_duration_sec: int,
    segments_per_song: int,
):
    duration_sec = waveform.shape[-1] / float(sample_rate)
    time_axis = np.linspace(0.0, duration_sec, num=waveform.shape[-1], endpoint=False)

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(8.2, 4.6),
        gridspec_kw={"height_ratios": [1.0, 1.25]},
    )
    figure.patch.set_facecolor("#FCFDFE")

    axes[0].plot(time_axis, waveform, color="#355C7D", linewidth=0.8)
    axes[0].fill_between(time_axis, waveform, 0.0, color="#355C7D", alpha=0.12)
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("Audio Preview", loc="left", pad=10, fontsize=12, fontweight="bold")
    axes[0].set_xlim(0.0, duration_sec)
    axes[0].grid(axis="x", linestyle="--", linewidth=0.8, alpha=0.18)

    try:
        import librosa

        mel = librosa.feature.melspectrogram(y=waveform, sr=sample_rate, n_mels=96, fmax=sample_rate // 2)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        axes[1].imshow(
            mel_db,
            aspect="auto",
            origin="lower",
            cmap="magma",
            extent=[0.0, duration_sec, 0.0, sample_rate / 2000.0],
        )
        axes[1].set_ylabel("kHz")
    except Exception:
        axes[1].specgram(waveform, Fs=sample_rate, NFFT=1024, noverlap=512, cmap="magma")
        axes[1].set_ylabel("Hz")

    axes[1].set_xlabel("Time (seconds)")

    boundary_times = [segment_duration_sec * index for index in range(segments_per_song + 1)]
    for axis in axes:
        axis.set_facecolor("#FCFDFE")
        for boundary_time in boundary_times:
            axis.axvline(boundary_time, color="#F8FAFC", linewidth=1.3)
            axis.axvline(boundary_time, color="#486581", linewidth=0.6, alpha=0.35)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
        axis.spines["left"].set_color("#BCCCDC")
        axis.spines["bottom"].set_color("#BCCCDC")

    figure.tight_layout()
    return figure


def save_current_figure(output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
