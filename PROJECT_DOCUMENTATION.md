# Project Documentation: Music Genre Classification with GTZAN and MERT

This file is the full project documentation, including design decisions, data pipeline rules, module responsibilities, experiment planning, optimization records, and the promoted training recipe. For a quick-start overview and minimal reproduction commands, see `README.md`.

This project builds a music genre classifier on top of the GTZAN dataset and the frozen `MERT-v1-95M` encoder. The goal is to deliver a stable, course-ready system with a clear data pipeline, reproducible evaluation, and a later Gradio demo.

## Project decisions

The project follows these fixed decisions:

- The `MERT-v1-95M` backbone is used only as a frozen feature extractor.
- Only lightweight classifier heads are trained later.
- Dataset splitting happens at the song level before any slicing.
- All audio is standardized to `24 kHz` mono because GTZAN is `22050 Hz` while MERT expects `24 kHz`.
- The main task is song-level genre classification.
- Each 30-second track is sliced into `6` segments of `5` seconds, segment embeddings are extracted, then mean pooled into a song embedding.
- The report should explicitly mention GTZAN limitations and potential label or recording noise.

## Why freeze MERT

This is a course project, not a large-scale research benchmark. Freezing MERT keeps the system simpler to train, more reproducible, less compute-heavy, and easier to explain. It also keeps the engineering focus on the data pipeline, feature extraction, and evaluation quality instead of unstable backbone fine-tuning.

## Why split by song before slicing

The split order must be:

1. Split original 30-second tracks into `train / val / test`.
2. Resample and preprocess each split.
3. Slice each song into `6 x 5s` segments.

If slicing is done first and random splitting is done later, different segments from the same song can leak into both training and testing. That produces over-optimistic metrics and invalid evaluation.

## Canonical data layout

The canonical raw dataset path is:

`data/raw/gtzan/genres_original`

The old `datasets_10classes/` directory is kept only as historical material. It is not part of the new pipeline.

Current known anomaly in the raw dataset:

- `blues.00000.zip` exists alongside normal `.wav` files. The scan script records it as a non-audio anomaly and excludes it from split generation.

## Directory structure

```text
music_genre_classifier/
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ configs/
│  └─ default.yaml
├─ data/
│  ├─ raw/
│  │  └─ gtzan/
│  │     └─ genres_original/
│  ├─ processed/
│  │  ├─ audio_24k/
│  │  └─ segments_5s/
│  ├─ splits/
│  └─ embeddings/
│     ├─ segment_level/
│     └─ song_level/
├─ src/
│  ├─ data/
│  ├─ features/
│  ├─ models/
│  ├─ train/
│  ├─ app/
│  └─ utils/
├─ outputs/
│  ├─ figures/
│  ├─ logs/
│  ├─ checkpoints/
│  └─ reports/
└─ notebooks/
   └─ exploration.ipynb
```

## Module responsibilities

### `src/data`

- `scan_dataset.py`: scan raw data, record valid audio, detect anomalies, and generate `raw_inventory.csv`.
- `make_splits.py`: create song-level `train / val / test` CSV files with genre-aware stratification.
- `preprocess_audio.py`: resample raw audio to `24 kHz` mono and save standardized WAV files.
- `slice_audio.py`: cut each standardized 30-second song into `6` segments of `5` seconds.
- `dataset_utils.py`: shared helpers for dataset scanning, split bookkeeping, path generation, and validation.

### `src/features`

- `extract_mert_embeddings.py`: later extract segment-level embeddings from frozen MERT.
- `aggregate_song_embeddings.py`: later average segment embeddings into a song-level representation.

### `src/models`

- `classifier_head.py`: later contain the linear baseline and 2-layer MLP classifier heads.
- `inference_utils.py`: later contain inference-time loading, aggregation, and prediction helpers.

### `src/train`

- `train_classifier.py`: later training entry point.
- `evaluate.py`: later evaluation entry point.
- `metrics.py`: later metric helpers for accuracy, macro F1, confusion matrix, and classification report.

### `src/app`

- `gradio_app.py`: later provide upload-based inference, Top-3 probabilities, segment-level predictions, and a probability bar chart.

### `src/utils`

- `config.py`: load YAML config and resolve project-relative paths.
- `seed.py`: set Python, NumPy, and PyTorch random seeds.
- `io.py`: common CSV and filesystem helpers.
- `plotting.py`: later store confusion matrix and probability plotting helpers.

## Planned pipeline

```text
GTZAN raw audio
  -> dataset scan
  -> song-level train/val/test split
  -> 24 kHz mono preprocessing
  -> 6 x 5s slicing
  -> frozen MERT segment embeddings
  -> mean pooling to song embeddings
  -> lightweight classifier head
  -> evaluation and demo
```

## Stage 1 commands

Create the environment first:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Run the data preparation entry points:

```powershell
python -m src.data.scan_dataset --config configs/default.yaml
python -m src.data.make_splits --config configs/default.yaml
python -m src.data.generate_metadata --config configs/default.yaml
python -m src.data.preprocess_audio --config configs/default.yaml --split train
python -m src.data.slice_audio --config configs/default.yaml --split train
python -m src.features.extract_mert_embeddings --config configs/default.yaml --split train --model-source offline
python -m src.features.aggregate_song_embeddings --config configs/default.yaml --split train
python -m src.train.train_classifier --config configs/default.yaml --model-type linear
python -m src.train.train_classifier --config configs/default.yaml --model-type mlp
python -m src.train.evaluate --config configs/default.yaml --checkpoint outputs/checkpoints/best_linear.pt
python -m src.app.gradio_app --config configs/default.yaml --model-source offline --checkpoint outputs\checkpoints\best_mlp.pt
```

Repeat preprocessing and slicing for `val` and `test` in later steps.

## VSCode workflow

Recommended order for step-by-step execution in VSCode:

1. Open `configs/default.yaml` and confirm paths.
2. Run `scan_dataset.py` and inspect `data/splits/raw_inventory.csv`.
3. Run `make_splits.py` and inspect `train.csv`, `val.csv`, and `test.csv`.
4. Run `preprocess_audio.py` for one split and verify audio output.
5. Run `slice_audio.py` for the same split and verify each song has six segments.

This keeps every stage inspectable and makes debugging much easier than running the whole pipeline end to end.

## GTZAN limitations

GTZAN is widely used for music genre classification, but it has known limitations:

- small dataset size
- potential recording noise and label noise
- possible artist or production bias
- dataset artifacts that can inflate results if splitting is not done carefully
- edge cases such as non-audio or duplicated files in local copies

The report should mention these limitations explicitly so the experimental design looks rigorous and trustworthy.

## Optimization roadmap

The next improvement pass should prioritize frozen-feature quality before any backbone fine-tuning:

1. Run a transformer layer sweep instead of assuming the final layer is best.
2. Replace plain segment mean pooling with stronger song-level aggregation.
3. Compare simple heads and light regularization before considering any fine-tuning.
4. Add inference-time logit averaging and test-time augmentation.

Detailed rationale and implementation order are recorded in [docs/training_optimization_plan.md](/d:/study_file/cv/first_hw/music_genre_classifier/docs/training_optimization_plan.md).

## Final promoted training recipe

The optimization rounds are now finished, and the promoted setup is:

- frozen `MERT-v1-95M`
- feature layer: `layer_06`
- song aggregation: `mean_std`
- head: `2-layer MLP`
- label smoothing: `0.05`
- chunk dropout: `0.5`
- no warmup
- no extra feature-domain augmentation
- no TTA

Promoted checkpoint:

- [best_segment_mean_std_mlp_layer_06_ls05_cd50_seed7.pt](/d:/study_file/cv/first_hw/music_genre_classifier/outputs/checkpoints/best_segment_mean_std_mlp_layer_06_ls05_cd50_seed7.pt)

Promoted result:

- test accuracy: `0.9307`
- test macro F1: `0.9304`

Consolidated record:

- [final_training_recipe.md](/d:/study_file/cv/first_hw/music_genre_classifier/docs/final_training_recipe.md)
- [phase1_layer_search_summary.md](/d:/study_file/cv/first_hw/music_genre_classifier/outputs/reports/phase1_layer_search_summary.md)
- [phase2_segment_aggregation_summary.md](/d:/study_file/cv/first_hw/music_genre_classifier/outputs/reports/phase2_segment_aggregation_summary.md)
- [phase2_confirmation_summary.md](/d:/study_file/cv/first_hw/music_genre_classifier/outputs/reports/phase2_confirmation_summary.md)
- [phase3_tta_summary.md](/d:/study_file/cv/first_hw/music_genre_classifier/outputs/reports/phase3_tta_summary.md)
- [phase4_regularization_summary.md](/d:/study_file/cv/first_hw/music_genre_classifier/outputs/reports/phase4_regularization_summary.md)

## Current demo command

Use the promoted checkpoint for the Gradio demo:

```powershell
python -m src.app.gradio_app --config configs/default.yaml --model-source offline --checkpoint outputs\checkpoints\best_segment_mean_std_mlp_layer_06_ls05_cd50_seed7.pt
```
