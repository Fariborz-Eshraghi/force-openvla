# AGENTS.md

## Project context

This repository contains Google Colab experiments for a visuo-tactile VLA robotics thesis project.

## Rules for Codex

- Keep all code compatible with Google Colab.
- Do not remove notebook cells unless clearly unnecessary.
- Prefer moving reusable logic into Python files under `src/`.
- Keep the notebook simple: setup, imports, configuration, execution, and visualisation.
- Avoid hardcoding private tokens, API keys, file paths, or credentials.
- If you change dependencies, update `requirements.txt`.
- After making changes, explain exactly what changed and how to run it in Colab.

## Suggested structure

- `notebooks/`: Colab notebooks
- `src/`: Python modules
- `configs/`: configuration files
- `scripts/`: setup or helper scripts
- `data/`: ignored data folder, not committed to GitHub
- `outputs/`: ignored output folder, not committed to GitHub

## Test commands

Run these when possible:

```bash
python -m py_compile src/*.py
