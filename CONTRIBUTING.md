# Contributing

Thanks for your interest in contributing.

## Dev Setup

1. Install [`uv`](https://github.com/astral-sh/uv).
2. Clone the repo and install dev dependencies:

   ```bash
   uv sync --extra dev
   ```

3. Install pre-commit hooks:

   ```bash
   uv run pre-commit install
   ```

## Branching

- Branch from `master`: `git checkout -b feat/<short-name>` or `fix/<short-name>`.
- Open a PR against `master`.
- CI (lint + test + build) must pass before merging.

## Code Style

- Formatter + linter: [`ruff`](https://docs.astral.sh/ruff/) (config in `pyproject.toml`).
- Run locally:

  ```bash
  make lint     # check
  make format   # auto-fix
  ```

- Line length 100. Target Python 3.11.

## Tests

- Use `pytest`. Tests live under `tests/`.
- Run: `make test` or `uv run pytest`.
- The test suite currently contains scaffolding only. Add real tests as
  behavior is stabilized.

## Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/) prefixes:

- `feat(scope): …` – new functionality
- `fix(scope): …` – bug fix
- `refactor(scope): …` – no behavior change
- `docs: …`, `chore: …`, `ci: …`, `test: …`

## Running Experiments

- Experiment configs live in `configs/<task>/*.yaml`.
- Never hardcode hyperparameters in Python code — add a field to the pydantic
  schema in `src/smoke_detection/configs/` and reference it from the
  LightningModule / DataModule.
- `lightning_logs/` is gitignored. Do not commit checkpoints or TensorBoard
  event files.
