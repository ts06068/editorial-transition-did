# Editorial Transition DiD

## Study Objective

Assess changes in observed 1-year citation impact after the 2024 JACC editorial transition relative to Circulation and European Heart Journal, overall and by article category, using difference-in-differences analyses. Private source data, credentials, and generated results are excluded.

## Folder structure

- `src/editorial_transition_did/`: reusable analysis package and CLI.
- `src/editorial_transition_did/reporting/`: tables, figures, and manuscript files.
- `notebooks/analysis.ipynb`: thin interactive wrapper around the full pipeline.
- `tests/test_smoke.py`: private-data-free smoke tests.
- `.github/workflows/smoke-test.yml`: automated CI checks.
- `pyproject.toml` and `.env.example`: dependencies and local configuration template.

## Code structure

- `cli.py` selects sample or full mode
- `pipeline.py` runs the analysis stages in order
- `config.py` defines the study periods and paths; `data.py` builds the analysis cohort
- `summaries.py`, `did.py`, and `prediction.py` produce descriptive, DiD, and forecasting results
- `validation.py` checks analysis invariants; `reporting/` writes the final outputs
- `sample.py` generates deterministic synthetic data for local and CI testing

## Environment

- Python 3.12
- Exact package versions are defined in `pyproject.toml`.
- Sample mode requires no private data or database connection; full mode reads local settings from `.env`.

## How to run

```powershell
python -m pip install -e ".[dev]"
editorial-transition-did sample
python -m pytest
```

For the full analysis, copy `.env.example` to `.env`, add the local workbook, HTML, and MSSQL settings, then run:

```powershell
editorial-transition-did full
```
