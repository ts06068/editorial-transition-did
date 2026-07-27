# Editorial Transition DiD

## Study Objective

Assess whether the 2024 JACC editorial transition changed 1-year citation impact for original research articles relative to Circulation and European Heart Journal using difference-in-differences analyses.

## Folder structure

```text
src/editorial_transition_did/  Modular analysis package and CLI
notebooks/                     Output-free analysis walkthrough
tests/                         Unit and smoke tests
.github/workflows/             Automated smoke test
pyproject.toml                 Environment and tooling configuration
```

## Code structure

`config.py` defines study constants and paths; `data.py` builds the analysis cohort; `did.py` and `prediction.py` contain the statistical analyses; and `pipeline.py` coordinates the complete run. Output-specific code lives under `reporting/`, while `sample.py` and `tests/test_smoke.py` provide a private-data-free executable check.

## Environment

Python 3.12 is required. Exact runtime and development dependency versions are defined in `pyproject.toml`. Sample mode is synthetic and database-free; full mode reads local settings from `.env`.

## How to run

```powershell
python -m pip install -e ".[dev]"
python -m ruff check .
python -m pytest
editorial-transition-did --mode sample
```

For the full analysis, populate `.env` from `.env.example`, then run `editorial-transition-did --mode full`.
