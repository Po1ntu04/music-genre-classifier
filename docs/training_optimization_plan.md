# Training Optimization Plan

## Baseline Audit

The current pipeline is stable and reproducible, but it is intentionally simple:

- `MERT-v1-95M` is frozen.
- Segment features use only `outputs.last_hidden_state`.
- Segment embeddings are reduced by time mean pooling.
- Song embeddings are reduced by plain mean over 6 segments.
- The classifier is either a single linear layer or a 2-layer MLP.

Measured baseline:

- `linear`: test accuracy `0.6832`, macro F1 `0.6826`
- `mlp`: test accuracy `0.7921`, macro F1 `0.7925`

Observed gap:

- The MLP reaches roughly `0.95` train accuracy but only `0.82` best validation accuracy.
- This indicates the frozen features are useful, but the small GTZAN training set is easy to overfit.

## What The Code Does Today

Relevant current behavior:

- [extract_mert_embeddings.py](/d:/study_file/cv/first_hw/music_genre_classifier/src/features/extract_mert_embeddings.py) pools the final hidden state over time and stores one embedding per segment.
- [aggregate_song_embeddings.py](/d:/study_file/cv/first_hw/music_genre_classifier/src/features/aggregate_song_embeddings.py) mean-pools the 6 segment embeddings into one song embedding.
- [train_classifier.py](/d:/study_file/cv/first_hw/music_genre_classifier/src/train/train_classifier.py) trains on the pre-aggregated song embedding only.
- [inference_utils.py](/d:/study_file/cv/first_hw/music_genre_classifier/src/models/inference_utils.py) also uses plain mean over the 6 segment embeddings for final song prediction.

So your core diagnosis is correct: the code still leaves performance on the table in layer choice, segment aggregation, and inference-time aggregation.

## Priority Order

### Phase 1: Layer Search

Goal:

- stop assuming the final transformer layer is optimal for genre classification

Why first:

- highest expected gain per engineering hour
- no backbone fine-tuning required
- almost no extra VRAM because MERT stays frozen

Implementation:

- save pooled segment embeddings for all transformer layers `1..12`
- aggregate a chosen layer into a song embedding
- train the same classifier on `layer_01 ... layer_12`
- compare validation macro F1

Acceptance:

- the best single-layer result should be compared against the current final-layer baseline
- keep the best-performing layer as the new default frozen feature set

### Phase 2: Learned Segment Aggregation

Goal:

- stop treating all 6 chunks as equally informative

Why second:

- the current mean song pooling is the biggest modeling simplification after layer choice
- some GTZAN songs clearly have stronger and weaker 5-second chunks

Implementation:

- keep training data at song level, but load all 6 segment embeddings per song
- support:
  - `mean`
  - `mean+std`
  - `attention pooling`
- add `chunk dropout` during training

Acceptance:

- attention pooling must outperform plain mean pooling on validation macro F1, otherwise mean stays as default

### Phase 3: Classifier Head And Regularization Search

Goal:

- improve generalization on small, noisy GTZAN labels

Implementation:

- compare:
  - `linear`
  - `LayerNorm + linear`
  - `2-layer MLP`
- add:
  - `label smoothing`
  - `weight decay`
  - optional `embedding dropout`

Acceptance:

- keep the simplest head that matches or exceeds the best validation macro F1

### Phase 4: Inference-Time Logit Averaging And TTA

Goal:

- improve robustness without extra training VRAM

Implementation:

- average segment logits instead of averaging post-softmax probabilities
- add shifted windows for test-time augmentation, for example:
  - `0.0s` offset
  - `2.5s` offset

Acceptance:

- apply only if it improves validation metrics or qualitative demo stability

### Phase 5: Learned Layer Weighting

Goal:

- learn a soft combination of transformer layers instead of picking only one

Implementation:

- keep per-layer frozen segment embeddings
- learn a softmax-normalized layer weight vector on top of the frozen layers

Why not first:

- it depends on the layer-aware feature extraction added in Phase 1
- it adds another trainable degree of freedom and should be compared against the best single-layer baseline, not the original last-layer baseline

### Phase 6: Optional Light Fine-Tuning

Goal:

- explore the ceiling only after frozen-feature improvements are exhausted

Implementation candidates:

- unfreeze the last `1-2` transformer blocks only
- or add LoRA/adapters to upper layers

Why last:

- highest instability
- highest engineering cost
- easiest way to overfit GTZAN

## Immediate Implementation Scope

This round implements the Phase 1 foundation:

- MERT extraction now saves pooled embeddings for all configured transformer layers.
- Song aggregation can now build a dedicated song-embedding set for any chosen layer.
- Training and evaluation can target a named embedding suffix, such as `layer_08`.

This is enough to run a proper layer sweep before touching the segment aggregator.

## Recommended Experiment Order

1. Re-extract segment embeddings with all transformer layers saved.
2. Aggregate song embeddings for layers `1..12`.
3. Train the same `linear` head on each layer.
4. Keep the best 2 to 3 layers.
5. Train `mlp` only on those shortlisted layers.
6. Freeze the best layer choice before moving to attention pooling.

## Phase 1 Commands

Re-extract segment embeddings once so every segment file contains all configured transformer layers:

```powershell
python -m src.features.extract_mert_embeddings --config configs/default.yaml --split train --model-source offline --overwrite
python -m src.features.extract_mert_embeddings --config configs/default.yaml --split val --model-source offline --overwrite
python -m src.features.extract_mert_embeddings --config configs/default.yaml --split test --model-source offline --overwrite
```

Build one song-embedding set for a chosen layer, for example layer 8:

```powershell
python -m src.features.aggregate_song_embeddings --config configs/default.yaml --split train --layer-index 8 --overwrite
python -m src.features.aggregate_song_embeddings --config configs/default.yaml --split val --layer-index 8 --overwrite
python -m src.features.aggregate_song_embeddings --config configs/default.yaml --split test --layer-index 8 --overwrite
```

Train and evaluate on that layer-specific embedding set:

```powershell
python -m src.train.train_classifier --config configs/default.yaml --model-type linear --embedding-suffix layer_08
python -m src.train.train_classifier --config configs/default.yaml --model-type mlp --embedding-suffix layer_08
python -m src.train.evaluate --config configs/default.yaml --checkpoint outputs\checkpoints\best_linear_layer_08.pt
python -m src.train.evaluate --config configs/default.yaml --checkpoint outputs\checkpoints\best_mlp_layer_08.pt
```

## Risks And Constraints

- GTZAN has only `999` valid songs in the local copy because `jazz.00054.wav` is unreadable.
- Any gain under roughly `0.5` to `1.0` macro F1 should be treated cautiously because validation size is small.
- More complex heads should not be trusted unless they improve validation and test consistently.
