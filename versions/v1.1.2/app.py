"""Hugging Face Spaces entrypoint for Imagination v1.1.2."""
import os

from imagination_v1_1_2_colab_gradio import build_ui

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))
