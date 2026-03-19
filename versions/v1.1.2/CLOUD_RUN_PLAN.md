# Imagination v1.1.2 — Cloud run plan

How to run v1.1.2 on cloud agents and see it work.

---

## 1. GitHub Actions (cloud validation agent)

When you push to `main`, GitHub Actions automatically runs `.github/workflows/validate-v112.yml`:

- **Checkout** the repo
- **Install** Python 3.11 + deps from `requirements.txt`
- **Create** a minimal fake model root so path resolution works
- **Validate** runtime imports (paths, registry, thinking, budget, auth, users)
- **Build** the Gradio UI as a smoke test

View runs: **GitHub repo → Actions tab**

This is your “cloud agent set up for the GitHub repo” — it runs in the cloud on every push.

---

## 2. Google Colab (cloud GPU)

For full inference with models:

1. Open [colab.research.google.com](https://colab.research.google.com)
2. New notebook
3. Runtime → Change runtime type → **GPU** (e.g. L4)
4. Run the cells from `COLAB_SETUP.md`:
   - Mount Drive, link `imagination-v1.1.0`
   - `pip install` deps
   - `cd versions/v1.1.2 && python imagination_v1_1_2_colab_gradio.py`
5. Use the Gradio URL (with `?fullscreen=true` if you want a static link)

Compute budget: ~58h/month on L4 (100 pts at 1.71/hr).

---

## 3. Hugging Face Spaces (cloud host)

For a public demo (no weights — lightweight UI only or with HF-hosted models):

1. Create a Space at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Set SDK to **Gradio**
3. Add repo as source or upload:
   - `app.py` (entrypoint)
   - `requirements.txt`
   - `imagination_runtime/`
   - `imagination_v1_1_2_colab_gradio.py`
4. Set `IMAGINATION_ROOT` in Space Settings → Variables (if needed)
5. Deploy; HF gives you a URL

Note: Without model weights, chat will fail on load. Use Spaces for UI demos or connect to an external model API.

---

## 4. Summary

| Where         | Purpose                      | Triggers                    |
|---------------|------------------------------|-----------------------------|
| GitHub Actions| Validate code on push        | `git push origin main`      |
| Google Colab  | Full inference with L4       | Manual run                  |
| HF Spaces     | Public UI demo               | Deploy from repo / upload   |

The GitHub Actions workflow is the “learn from example” setup: push to your repo and watch the Actions tab to see it run in the cloud.
