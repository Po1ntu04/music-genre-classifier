from __future__ import annotations

import argparse
from pathlib import Path


APP_CSS = """
.gradio-container {
  max-width: 1240px !important;
  background: linear-gradient(180deg, #F8FBFF 0%, #F5F7FA 100%);
}
.app-title h1, .app-title h2, .app-title p {
  margin: 0;
}
.panel-card {
  background: rgba(255, 255, 255, 0.92);
  border: 1px solid #D9E2EC;
  border-radius: 18px;
  box-shadow: 0 10px 28px rgba(15, 23, 42, 0.08);
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the Gradio demo for music genre classification.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the YAML config file.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Classifier checkpoint path. Defaults to the promoted segment-level formal checkpoint if present.",
    )
    parser.add_argument(
        "--model-source",
        choices=["offline", "online"],
        default="offline",
        help="Use a local MERT directory or download from Hugging Face.",
    )
    parser.add_argument(
        "--tta-mode",
        choices=["none", "dual", "random", "dual_random"],
        default="none",
        help="Inference-time view strategy for logit averaging.",
    )
    parser.add_argument(
        "--tta-random-views",
        type=int,
        default=2,
        help="Number of additional random views when tta-mode includes random crops.",
    )
    parser.add_argument(
        "--tta-seed",
        type=int,
        default=None,
        help="Random seed used by TTA random crops. Defaults to project seed.",
    )
    parser.add_argument("--server-name", default="127.0.0.1", help="Gradio server host.")
    parser.add_argument("--server-port", type=int, default=7860, help="Gradio server port.")
    parser.add_argument("--share", action="store_true", help="Enable Gradio share link.")
    return parser.parse_args()


def resolve_checkpoint(config: dict, checkpoint_path: Path | None) -> Path:
    if checkpoint_path is not None:
        return checkpoint_path.resolve()

    promoted_segment_path = (
        config["paths"]["outputs_dir"]
        / "checkpoints"
        / "best_segment_mean_std_mlp_layer_06_ls05_cd50_seed7.pt"
    )
    if promoted_segment_path.exists():
        return promoted_segment_path

    mlp_path = config["paths"]["outputs_dir"] / "checkpoints" / "best_mlp.pt"
    linear_path = config["paths"]["outputs_dir"] / "checkpoints" / "best_linear.pt"
    if mlp_path.exists():
        return mlp_path
    if linear_path.exists():
        return linear_path
    raise FileNotFoundError("No checkpoint found. Train a classifier first or pass --checkpoint explicitly.")


def main() -> None:
    args = parse_args()

    import gradio as gr

    from src.models.inference_utils import predict_from_audio
    from src.utils.config import load_config
    from src.utils.seed import set_seed

    config = load_config(args.config)
    set_seed(config["project"]["seed"])
    checkpoint_path = resolve_checkpoint(config, args.checkpoint)

    def run_inference(audio_path: str | None):
        if not audio_path:
            raise gr.Error("Please upload an audio file.")

        result = predict_from_audio(
            audio_path=audio_path,
            config=config,
            checkpoint_path=checkpoint_path,
            model_source=args.model_source,
            tta_mode=args.tta_mode,
            random_views=args.tta_random_views,
            tta_seed=args.tta_seed,
        )
        return (
            result["result_card_html"],
            result["top3_html"],
            result["timeline_chart"],
            result["chart"],
            result["audio_preview_chart"],
            result["model_info_html"],
        )

    with gr.Blocks(title="Music Genre Classifier", css=APP_CSS) as demo:
        gr.Markdown(
            "## Music Genre Classifier\n"
            "Frozen `MERT-v1-95M` with the promoted course-project configuration:\n"
            "`layer_06 + mean_std + MLP + label smoothing 0.05 + chunk dropout 0.5`."
        )

        with gr.Row():
            with gr.Column(scale=7):
                audio_input = gr.Audio(label="Upload Audio", type="filepath")
                run_button = gr.Button("Predict", variant="primary")
            with gr.Column(scale=5):
                result_card = gr.HTML(label="Prediction Result")
                top3_cards = gr.HTML(label="Top-3")

        with gr.Row():
            audio_preview_plot = gr.Plot(label="Waveform and Spectrogram Preview")
            timeline_plot = gr.Plot(label="Per-segment Timeline")

        probability_plot = gr.Plot(label="Song-level Genre Probabilities")

        with gr.Accordion("Model Information", open=False):
            model_info = gr.HTML(
                value=(
                    f"<div style='color:#486581;'>Checkpoint: {checkpoint_path}<br>"
                    f"MERT Source: {args.model_source}<br>"
                    f"TTA: {args.tta_mode}</div>"
                )
            )

        run_button.click(
            fn=run_inference,
            inputs=[audio_input],
            outputs=[result_card, top3_cards, timeline_plot, probability_plot, audio_preview_plot, model_info],
        )

    demo.launch(server_name=args.server_name, server_port=args.server_port, share=args.share)


if __name__ == "__main__":
    main()
