# IMAGINATION v1.1.2

Version 1.1.2 — spruced-up UI, Thinking Path panel, multi-provider auth, per-user memory, compute budget tracking.

## Features

- **Thinking Path**: AI explains its reasoning and source commentary when researching
- **Multi-provider auth**: Google SSO, Apple Sign-In (stub), HuggingFace, GitHub, email/password
- **Per-user memory**: Data, context, preferences saved per account
- **Global memory**: Admin-editable behavior instructions (source citing, guardrails)
- **Compute budget**: ~58h/month on L4 (100 pts at 1.71/hr), visible in UI

## Run

See `COLAB_SETUP.md` for Colab instructions. For local:

```bash
set IMAGINATION_ROOT=f:\imagination-v1.1.0
python imagination_v1_1_2_colab_gradio.py
```
