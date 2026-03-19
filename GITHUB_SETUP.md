# GitHub setup for Imagination

## 1. Install Git and GitHub CLI (if not done)

- **Git**: https://git-scm.com/download/win or `winget install Git.Git`
- **GitHub CLI**: https://cli.github.com/ or `winget install GitHub.cli`

## 2. Authenticate
```powershell
gh auth login
```
Follow prompts.

## 3. Init repo and push
From the project root (`f:\imagination-v1.1.0`):

```powershell
git init
git add .
git commit -m "v1.1.2 initial"
gh repo create imagination-v1 --private --source=. --push
```

Or if you already created the repo on GitHub:
```powershell
git remote add origin https://github.com/YOUR_USERNAME/imagination-v1.git
git branch -M main
git push -u origin main
```

## .gitignore
Weights (`*.safetensors`, `*.bin`, etc.), `.cache/`, `temp/`, `hf/`, `original/` are excluded.
