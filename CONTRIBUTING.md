# Contributing to imagination-ai

Thanks for helping improve this project. These guidelines keep changes reviewable and CI green.

## Scope of this repo

- **Versioned apps** live under `versions/<version>/` (Python, Gradio, FastAPI, training scripts).
- **Shared runtime** code is copied or symlinked per version (`imagination_runtime/` inside each active version).
- **`v0-imagination-ui/`** is a **nested repository** (its own git history). UI changes belong in that directory; the parent repo only records the submodule pointer when you bump it.
- **Large weights and datasets** are not committed. Use `IMAGINATION_ROOT` and document any new env vars in the relevant `README` or `AGENTS.md`.

## Before you open a PR

1. **Target the right tree** — Don’t mix unrelated versions in one PR (e.g. don’t refactor v1.1.2 and v1.3 together unless necessary).
2. **Match existing style** — Same formatting, imports, and patterns as neighboring files.
3. **Run checks** for the area you touched:
   - Python: `python -m py_compile` on edited files, or the version’s smoke import block from [`AGENTS.md`](AGENTS.md) where applicable.
   - Frontend (`v0-imagination-ui`): `pnpm lint` / `pnpm build` (or `npm run build`) before pushing.
4. **CI** — If your change affects `versions/v1.1.2/`, ensure `.github/workflows/validate-v112.yml` still makes sense (update the workflow only when the check itself should change).

## Commits and PRs

- Use clear, imperative commit messages (e.g. “Fix chat scroll overflow in v0 UI”).
- Describe **what** changed and **why** in the PR body; link issues if any.
- Keep diffs focused—avoid drive-by refactors or unrelated formatting.

## Security and safety

- Do not commit secrets (API keys, tokens, Tailscale keys, HF tokens). Use env vars and local `.env` files that are gitignored.
- If you find a vulnerability, report it privately to the maintainers instead of filing a public issue first.

## License

By contributing, you agree that your contributions are licensed under the same terms as the project—see [`LICENSE.md`](LICENSE.md).
