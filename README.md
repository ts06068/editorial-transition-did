# Editorial Transition DiD

## Study Objective

Estimate changes in observed 1-year citation impact after the 2024 JACC editorial transition with a difference-in-differences design. Circulation and European Heart Journal are the comparison journals; results are reported overall and by article category. Source data, credentials, and generated results are excluded.

## Folder structure

- `src/editorial_transition_did/`: analysis package and CLI.
- `src/editorial_transition_did/reporting/`: table, figure, and document writers.
- `notebooks/analysis.ipynb`: thin wrapper around the package pipeline.
- `tests/test_smoke.py`: synthetic-data smoke tests.
- `.github/workflows/smoke-test.yml`: CI workflow.
- `pyproject.toml` and `.env.example`: dependency and configuration contracts.

## Code structure

- `cli.py` dispatches sample or full mode. `pipeline.py` owns execution order.
- `config.py` defines dates, windows, and paths. `data.py` builds the analysis frame.
- `summaries.py` creates descriptive tables. `did.py` runs primary and sensitivity analyses.
- `prediction.py` handles temporal validation and forecasting.
- `validation.py` enforces hard checks. `reporting/` writes outputs.
- `sample.py` supplies deterministic synthetic data for local and CI runs.

## Environment

- Python 3.12.
- Package versions are pinned in `pyproject.toml`.
- Sample mode needs no private data or database. Full mode reads `.env`.

## How to run

```powershell
python -m pip install -e ".[dev]"
editorial-transition-did sample
python -m pytest
```

Full mode needs the local workbook, publisher HTML, and read-only MSSQL settings in `.env`:

```powershell
editorial-transition-did full
```
