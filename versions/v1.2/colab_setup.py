#!/usr/bin/env python3
"""
Imagination v1.2 — Google Colab one-shot setup.

Colab (recommended — two steps):

    # Cell 1 — run as Python (not !python), then authenticate:
    from google.colab import drive
    drive.mount("/content/drive")

    # Cell 2 — setup (subprocess cannot call drive.mount):
    !python colab_setup.py

Then either:

    !python app.py

Or one-shot (setup + load weights + Gradio in the **same** process — recommended if you want
everything in one cell after mounting Drive):

    !python colab_setup.py --launch

Note: `!python colab_setup.py` and `!python app.py` are **different processes** — weights loaded
in setup alone are **not** visible to app.py. Use `--launch` or rely on **app.py** preloading
(which runs before the server starts).

If Drive is already mounted, you can run only `!python colab_setup.py`.

Optional environment variables (set before running, or edit defaults below):

    IMAGINATION_DRIVE_PATH  Path to repo on Drive (default: /content/drive/MyDrive/imagination-v1.1.0)
    IMAGINATION_ROOT        Content path for the repo (default: /content/imagination-v1.1.0)
    IMAGINATION_COPY        Set to "1" to copy Drive folder to /content instead of symlink (faster I/O, more disk)
    SKIP_PIP                Set to "1" to skip dependency install
    IMAGINATION_CAD_CODER_PATH        Override coder weights dir (abs or relative to IMAGINATION_ROOT)
    IMAGINATION_MAIN_MAX_NEW_TOKENS   Main chat desired max new tokens (default 8192; clamped to context left; stops on EOS)
    IMAGINATION_REASONING_MAX_NEW_TOKENS  Reasoning/Research module (default 8192; max 131072)
    IMAGINATION_CODER_MAX_NEW_TOKENS  Coder tab max new tokens (default 16384; e.g. 100000 for huge scripts)
    IMAGINATION_CODER_SKIP_CLOAK      Set to "1" to stream raw coder output (recommended with very high coder tokens)
    IMAGINATION_CODER_REPETITION_PENALTY  Coder generate repetition_penalty (default 1.22; try 1.35 if loops persist)
    IMAGINATION_CODER_NO_REPEAT_NGRAM   e.g. 32 to forbid repeating same 32-token span (0/off to disable; can hurt some code)
    IMAGINATION_TRAINING_LOG        Set to "0" to disable JSONL export of main-chat turns (default: on → temp/training_exports/)
    --launch                After setup: preload main LLM + start Gradio (same process as setup)
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def _in_colab() -> bool:
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


def _pip_install() -> None:
    req = Path(__file__).resolve().parent / "requirements.txt"
    if req.is_file():
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "-r", str(req)],
        )
    else:
        pkgs = [
            "gradio",
            "transformers",
            "accelerate",
            "safetensors",
            "bitsandbytes",
            "requests",
            "beautifulsoup4",
            "ddgs",
            "diffusers",
            "bcrypt",
        ]
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])


def _drive_already_mounted() -> bool:
    return Path("/content/drive/MyDrive").is_dir()


def _can_use_colab_drive_mount() -> bool:
    """drive.mount() only works in the notebook kernel, not under !python subprocess."""
    try:
        from IPython import get_ipython

        ip = get_ipython()
        if ip is None:
            return False
        return getattr(ip, "kernel", None) is not None
    except Exception:
        return False


def _mount_drive() -> None:
    from google.colab import drive

    drive.mount("/content/drive", force_remount=False)


def _ensure_drive_mounted_colab() -> None:
    if _drive_already_mounted():
        print("[setup] Google Drive already mounted — skipping drive.mount()")
        return
    if _can_use_colab_drive_mount():
        print("[setup] Mounting Google Drive…")
        _mount_drive()
        return
    print(
        "[setup] ERROR: Google Drive is not mounted, and this script was started with !python …\n"
        "        Colab only allows drive.mount() from a normal notebook cell.\n\n"
        "        Fix — run this in the cell ABOVE, then run colab_setup again:\n\n"
        "            from google.colab import drive\n"
        "            drive.mount('/content/drive')\n\n"
        "        Then:\n"
        "            !python colab_setup.py\n",
        file=sys.stderr,
    )
    raise SystemExit(1)


def _ensure_content_repo(drive_src: str, content_dst: str, copy: bool) -> None:
    drive_path = Path(drive_src)
    dst = Path(content_dst)

    if dst.resolve() == drive_path.resolve():
        return

    if dst.is_symlink():
        print(f"[setup] Using existing symlink: {dst}")
        return
    if dst.is_dir():
        try:
            if any(dst.iterdir()):
                print(f"[setup] Using existing folder: {dst}")
                return
        except OSError:
            pass
        shutil.rmtree(dst, ignore_errors=True)
    elif dst.exists():
        dst.unlink()

    if not drive_path.is_dir():
        raise FileNotFoundError(
            f"Repo not found on Drive: {drive_src}\n"
            "Upload your full imagination-v1.1.0 folder (with models) to that path, "
            "or set IMAGINATION_DRIVE_PATH to the correct folder."
        )

    dst_parent = dst.parent
    dst_parent.mkdir(parents=True, exist_ok=True)

    if copy:
        print(f"[setup] Copying (this may take a while)…\n  {drive_src}\n  -> {content_dst}")
        shutil.copytree(drive_src, content_dst, dirs_exist_ok=True)
    else:
        print(f"[setup] Symlinking:\n  {drive_src}\n  -> {content_dst}")
        os.symlink(drive_src, content_dst, target_is_directory=True)


def main() -> str:
    """
    Returns resolved IMAGINATION_ROOT (content path) for follow-up actions.
    """
    print("Imagination v1.2 — Colab setup\n")

    if os.environ.get("SKIP_PIP", "").strip().lower() in ("1", "true", "yes"):
        print("[setup] Skipping pip (SKIP_PIP set)")
    else:
        print("[setup] Installing Python dependencies…")
        _pip_install()
        print("[setup] Dependencies OK")

    drive_src = os.environ.get(
        "IMAGINATION_DRIVE_PATH",
        "/content/drive/MyDrive/imagination-v1.1.0",
    ).rstrip("/\\")

    content_root = os.environ.get(
        "IMAGINATION_ROOT",
        "/content/imagination-v1.1.0",
    ).rstrip("/\\")

    copy_mode = os.environ.get("IMAGINATION_COPY", "").strip().lower() in ("1", "true", "yes")

    if _in_colab():
        print("[setup] Colab detected — checking Google Drive…")
        _ensure_drive_mounted_colab()
        _ensure_content_repo(drive_src, content_root, copy=copy_mode)
    else:
        print("[setup] Not running in Colab — skipping Drive mount and symlink.")
        root = Path(content_root)
        if not root.is_dir():
            print(
                f"[setup] Warning: IMAGINATION_ROOT is not a directory: {content_root}\n"
                "Set IMAGINATION_ROOT to your local repo path if needed."
            )

    os.environ["IMAGINATION_ROOT"] = content_root

    v12 = Path(content_root) / "versions" / "v1.2"
    if not v12.is_dir():
        print(f"[setup] Warning: expected app folder missing: {v12}")

    print()
    print("=" * 56)
    print("  Setup finished.")
    print(f"  IMAGINATION_ROOT = {content_root}")
    print()
    print("  Next — start the UI (weights preload before the server opens):")
    print(f"    %cd {v12}")
    print("    !python app.py")
    print()
    print("  Or one cell after mount:")
    print("    !python colab_setup.py --launch")
    print("=" * 56)

    return content_root


def _launch_gradio(content_root: str) -> None:
    """Preload main LLM and open Gradio in this process (Colab-friendly)."""
    v12 = Path(content_root) / "versions" / "v1.2"
    if not v12.is_dir():
        raise FileNotFoundError(f"Missing app folder: {v12}")
    os.chdir(str(v12))
    p = str(v12)
    if p not in sys.path:
        sys.path.insert(0, p)
    os.environ["IMAGINATION_ROOT"] = content_root

    print("\n[setup] Pre-loading main model, then starting Gradio…\n", flush=True)
    from imagination_v1_2 import build_ui, preload_main_model

    if os.getenv("SKIP_PRELOAD", "").strip().lower() not in ("1", "true", "yes"):
        preload_main_model()
    demo = build_ui()
    port = int(os.getenv("PORT", "7860"))
    share = os.getenv("GRADIO_SHARE", "true").lower() in ("1", "true", "yes")
    print()
    print("=" * 50)
    print("  Imagination v1.2")
    print("  Local:  http://localhost:" + str(port))
    if share:
        print("  Share:  (tunnel URL below)")
    print("=" * 50)
    print()
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=share,
        show_error=True,
    )


if __name__ == "__main__":
    do_launch = "--launch" in sys.argv
    content_root = main()
    if do_launch:
        _launch_gradio(content_root)
