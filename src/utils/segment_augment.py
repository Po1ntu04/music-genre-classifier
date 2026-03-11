from __future__ import annotations

import math

import torch


class SegmentEmbeddingAugmenter:
    def __init__(
        self,
        time_mask_prob: float = 0.0,
        time_mask_max_width: int = 1,
        freq_mask_prob: float = 0.0,
        freq_mask_max_width_ratio: float = 0.0,
        gaussian_noise_prob: float = 0.0,
        gaussian_noise_std: float = 0.0,
        random_loudness_prob: float = 0.0,
        random_loudness_db_range: float = 0.0,
    ) -> None:
        self.time_mask_prob = float(time_mask_prob)
        self.time_mask_max_width = int(max(time_mask_max_width, 1))
        self.freq_mask_prob = float(freq_mask_prob)
        self.freq_mask_max_width_ratio = float(max(freq_mask_max_width_ratio, 0.0))
        self.gaussian_noise_prob = float(gaussian_noise_prob)
        self.gaussian_noise_std = float(max(gaussian_noise_std, 0.0))
        self.random_loudness_prob = float(random_loudness_prob)
        self.random_loudness_db_range = float(max(random_loudness_db_range, 0.0))

    def is_enabled(self) -> bool:
        return any(
            value > 0.0
            for value in (
                self.time_mask_prob,
                self.freq_mask_prob,
                self.gaussian_noise_prob,
                self.random_loudness_prob,
            )
        )

    def __call__(self, embeddings: torch.Tensor) -> torch.Tensor:
        augmented = embeddings
        if self.time_mask_prob > 0.0:
            augmented = self._apply_time_mask(augmented)
        if self.freq_mask_prob > 0.0:
            augmented = self._apply_frequency_mask(augmented)
        if self.gaussian_noise_prob > 0.0 and self.gaussian_noise_std > 0.0:
            augmented = self._apply_gaussian_noise(augmented)
        if self.random_loudness_prob > 0.0 and self.random_loudness_db_range > 0.0:
            augmented = self._apply_random_loudness(augmented)
        return augmented

    def _apply_time_mask(self, embeddings: torch.Tensor) -> torch.Tensor:
        batch_size, num_segments, _ = embeddings.shape
        if num_segments <= 1:
            return embeddings

        augmented = embeddings.clone()
        max_width = min(self.time_mask_max_width, num_segments)
        for sample_index in range(batch_size):
            if torch.rand(1, device=embeddings.device).item() >= self.time_mask_prob:
                continue
            width = int(torch.randint(1, max_width + 1, (1,), device=embeddings.device).item())
            start = int(torch.randint(0, num_segments - width + 1, (1,), device=embeddings.device).item())
            augmented[sample_index, start : start + width, :] = 0.0
        return augmented

    def _apply_frequency_mask(self, embeddings: torch.Tensor) -> torch.Tensor:
        batch_size, _, feature_dim = embeddings.shape
        if feature_dim <= 1:
            return embeddings

        augmented = embeddings.clone()
        max_width = max(1, int(math.ceil(feature_dim * self.freq_mask_max_width_ratio)))
        max_width = min(max_width, feature_dim)
        for sample_index in range(batch_size):
            if torch.rand(1, device=embeddings.device).item() >= self.freq_mask_prob:
                continue
            width = int(torch.randint(1, max_width + 1, (1,), device=embeddings.device).item())
            start = int(torch.randint(0, feature_dim - width + 1, (1,), device=embeddings.device).item())
            augmented[sample_index, :, start : start + width] = 0.0
        return augmented

    def _apply_gaussian_noise(self, embeddings: torch.Tensor) -> torch.Tensor:
        sample_mask = (
            torch.rand(embeddings.shape[0], device=embeddings.device) < self.gaussian_noise_prob
        ).to(embeddings.dtype)
        if sample_mask.sum().item() == 0:
            return embeddings
        noise = torch.randn_like(embeddings) * self.gaussian_noise_std
        return embeddings + noise * sample_mask.view(-1, 1, 1)

    def _apply_random_loudness(self, embeddings: torch.Tensor) -> torch.Tensor:
        sample_mask = (
            torch.rand(embeddings.shape[0], device=embeddings.device) < self.random_loudness_prob
        ).to(embeddings.dtype)
        if sample_mask.sum().item() == 0:
            return embeddings

        gain_db = (
            torch.rand(embeddings.shape[0], device=embeddings.device) * 2.0 - 1.0
        ) * self.random_loudness_db_range
        gain = torch.pow(10.0, gain_db / 20.0).to(embeddings.dtype)
        blended_gain = gain * sample_mask + (1.0 - sample_mask)
        return embeddings * blended_gain.view(-1, 1, 1)
