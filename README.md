# Music Genre Classifier with GTZAN and MERT

基于 `GTZAN` 和冻结的 `MERT-v1-95M` 实现音乐流派分类。项目包含完整的数据处理、特征提取、训练评估和 Gradio 推理界面。

完整实验设计、优化过程和正式训练方案见 [PROJECT_DOCUMENTATION.md](./PROJECT_DOCUMENTATION.md)。

## 项目要点

- `MERT-v1-95M` 仅作冻结特征提取器，不微调主干
- 先按歌曲级划分 `train / val / test`，再切 `6 x 5s`
- 所有音频统一到 `24 kHz`、单声道
- 正式方案：`layer_06 + mean_std + 2-layer MLP`
- 正式 checkpoint：
  - `outputs/checkpoints/best_segment_mean_std_mlp_layer_06_ls05_cd50_seed7.pt`
- 正式测试结果：
  - accuracy `0.9307`
  - macro F1 `0.9304`

## 关键路径

- 配置文件：`configs/default.yaml`
- 原始数据：`data/raw/gtzan/genres_original`
- 本地 MERT：`models/MERT-v1-95M`
- 输出目录：`outputs/`

## 环境

- Python `3.8+`
- `pip install -r requirements.txt`
- 若使用 Gradio，建议 Python `3.8` 搭配 `gradio<4` 或 Python `3.10+`

## 常用指令

### 数据扫描与划分

```powershell
python -m src.data.scan_dataset --config configs/default.yaml
python -m src.data.make_splits --config configs/default.yaml
python -m src.data.generate_metadata --config configs/default.yaml
```

### 重采样与切片

```powershell
python -m src.data.preprocess_audio --config configs/default.yaml --split train
python -m src.data.preprocess_audio --config configs/default.yaml --split val
python -m src.data.preprocess_audio --config configs/default.yaml --split test

python -m src.data.slice_audio --config configs/default.yaml --split train
python -m src.data.slice_audio --config configs/default.yaml --split val
python -m src.data.slice_audio --config configs/default.yaml --split test
```

### 使用本地 MERT 提取 embedding

```powershell
python -m src.features.extract_mert_embeddings --config configs/default.yaml --split train --model-source offline
python -m src.features.extract_mert_embeddings --config configs/default.yaml --split val --model-source offline
python -m src.features.extract_mert_embeddings --config configs/default.yaml --split test --model-source offline
```

### 使用 Hugging Face 在线拉取原始 MERT 提取 embedding

当前版本已支持，`configs/default.yaml` 中的模型 ID 为 `m-a-p/MERT-v1-95M`。

```powershell
python -m src.features.extract_mert_embeddings --config configs/default.yaml --split train --model-source online
python -m src.features.extract_mert_embeddings --config configs/default.yaml --split val --model-source online
python -m src.features.extract_mert_embeddings --config configs/default.yaml --split test --model-source online
```

说明：
- 在线模式通过 `transformers` 的 `AutoFeatureExtractor` 和 `AutoModel.from_pretrained(...)` 拉取 Hugging Face 模型
- 当前配置已启用 `trust_remote_code: true`
- 首次下载会写入 `.hf_cache/`

### 聚合、训练与评估

```powershell
python -m src.features.aggregate_song_embeddings --config configs/default.yaml --split train --layer-index 6
python -m src.features.aggregate_song_embeddings --config configs/default.yaml --split val --layer-index 6
python -m src.features.aggregate_song_embeddings --config configs/default.yaml --split test --layer-index 6

python -m src.train.train_segment_aggregator --config configs/default.yaml --embedding-suffix layer_06 --aggregation-type mean_std --model-type mlp --label-smoothing 0.05 --chunk-dropout-prob 0.5 --seed 7

python -m src.train.evaluate_segment_aggregator --config configs/default.yaml --checkpoint outputs\checkpoints\best_segment_mean_std_mlp_layer_06_ls05_cd50_seed7.pt
```

### 启动 Demo

本地模型：

```powershell
python -m src.app.gradio_app --config configs/default.yaml --model-source offline --checkpoint outputs\checkpoints\best_segment_mean_std_mlp_layer_06_ls05_cd50_seed7.pt
```

在线 Hugging Face 模型：

```powershell
python -m src.app.gradio_app --config configs/default.yaml --model-source online --checkpoint outputs\checkpoints\best_segment_mean_std_mlp_layer_06_ls05_cd50_seed7.pt
```

启动后访问终端输出的本地地址，通常是 `http://127.0.0.1:7860`。
