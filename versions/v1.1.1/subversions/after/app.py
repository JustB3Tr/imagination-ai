import os

from imagination_v1_1_1_colab_gradio import build_ui


if __name__ == "__main__":
    # Hugging Face Spaces sets PORT. Gradio will pick it up automatically.
    # Set IMAGINATION_ROOT in Spaces Secrets or as an environment variable.
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))

