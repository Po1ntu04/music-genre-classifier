# Final Training Recipe

This file records the promoted training recipe after the completed optimization rounds.

## Promoted model

- frozen `MERT-v1-95M`
- segment feature layer: `layer_06`
- song representation: `mean + std` over the 6 segment embeddings
- classifier head: `2-layer MLP`
- label smoothing: `0.05`
- chunk dropout: `0.5`
- no warmup
- no additional feature-domain augmentation
- no TTA at inference

Promoted checkpoint:

- `outputs/checkpoints/best_segment_mean_std_mlp_layer_06_ls05_cd50_seed7.pt`

## Why this is the promoted recipe

The final recommendation was selected from the completed experiments:

1. Layer search showed that the final transformer layer is not optimal for GTZAN.
2. Fixed `layer_06` outperformed the original final-layer baseline.
3. `mean_std` segment aggregation outperformed plain mean aggregation on repeated confirmation runs.
4. Learned layer weighting was worse than the best fixed layer.
5. Offline vector heads such as logistic regression were strong baselines, but still below the promoted segment model.
6. TTA decreased test performance.
7. Warmup and the added feature-domain augmentations also decreased performance.

## Key results snapshot

- Original final-layer MLP baseline:
  - test macro F1: about `0.7925`
- Promoted formal model:
  - test accuracy: `0.9307`
  - test macro F1: `0.9304`

The result files are already saved under:

- `outputs/reports/phase1_layer_search_summary.md`
- `outputs/reports/phase2_segment_aggregation_summary.md`
- `outputs/reports/phase2_confirmation_summary.md`
- `outputs/reports/phase3_tta_summary.md`
- `outputs/reports/phase4_regularization_summary.md`

## Reproduce the promoted checkpoint

```powershell
python -m src.train.train_segment_aggregator `
  --config configs/default.yaml `
  --embedding-suffix layer_06 `
  --aggregation-type mean_std `
  --model-type mlp `
  --label-smoothing 0.05 `
  --chunk-dropout-prob 0.5 `
  --seed 7

python -m src.train.evaluate_segment_aggregator `
  --config configs/default.yaml `
  --checkpoint outputs\checkpoints\best_segment_mean_std_mlp_layer_06_ls05_cd50_seed7.pt
```

## Demo checkpoint

The Gradio demo should use the same promoted checkpoint by default:

- `outputs/checkpoints/best_segment_mean_std_mlp_layer_06_ls05_cd50_seed7.pt`
