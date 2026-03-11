from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def load_wav(path: str | Path) -> tuple[torch.Tensor, int]:
    path = Path(path)
    with wave.open(str(path), "rb") as wav_file:
        num_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        num_frames = wav_file.getnframes()
        raw_bytes = wav_file.readframes(num_frames)

    if sample_width == 1:
        audio = np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32)
        audio = (audio - 128.0) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")

    audio = audio.reshape(-1, num_channels).T
    return torch.from_numpy(audio), sample_rate


def save_wav(path: str | Path, waveform: torch.Tensor, sample_rate: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    waveform = waveform.detach().cpu().to(torch.float32).clamp(-1.0, 1.0)
    pcm = (waveform.numpy().T * 32767.0).astype(np.int16)

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(waveform.shape[0])
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


def mix_down_to_mono(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.ndim == 1:
        return waveform.unsqueeze(0)
    if waveform.shape[0] == 1:
        return waveform
    return waveform.mean(dim=0, keepdim=True)


def resample_waveform(waveform: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    if orig_sr == target_sr:
        return waveform

    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    target_length = int(round(waveform.shape[-1] * target_sr / orig_sr))
    if target_length <= 0:
        raise ValueError("Resampled waveform would be empty.")

    resampled = F.interpolate(
        waveform.unsqueeze(0),
        size=target_length,
        mode="linear",
        align_corners=False,
    )
    return resampled.squeeze(0)


def pad_or_trim(waveform: torch.Tensor, target_num_samples: int) -> torch.Tensor:
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    current_num_samples = waveform.shape[-1]
    if current_num_samples == target_num_samples:
        return waveform
    if current_num_samples > target_num_samples:
        return waveform[:, :target_num_samples]
    return F.pad(waveform, (0, target_num_samples - current_num_samples))
