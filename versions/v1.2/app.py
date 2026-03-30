"""Imagination v1.2 — Entry point. Run with: python app.py"""
import os

from imagination_v1_2 import build_ui, preload_main_model

if __name__ == "__main__":
    if os.getenv("SKIP_PRELOAD", "").strip().lower() not in ("1", "true", "yes"):
        preload_main_model()
    else:
        print("[imagination] SKIP_PRELOAD=1 — main model loads on first message.", flush=True)
    demo = build_ui()
    port = int(os.getenv("PORT", "7860"))
    share = os.getenv("GRADIO_SHARE", "true").lower() in ("1", "true", "yes")
    print()
    print("=" * 50)
    print("  Imagination v1.2")
    print("  Local:  http://localhost:" + str(port))
    if share:
        print("  Share:  (Gradio tunnel URL will appear below)")
    print("=" * 50)
    print()
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=share,
        show_error=True,
    )
