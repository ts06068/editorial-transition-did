# Editorial Transition DiD

## Study Objective

Estimate changes in observed 1-year citation impact after the 2024 JACC editorial transition with a difference-in-differences design. Circulation and European Heart Journal are the comparison journals; results are reported overall and by article category. Source data, credentials, and generated results are excluded.

## Folder structure

```text
.
├── src/editorial_transition_did/
│   ├── __init__.py                  Public API and lazy pipeline entry
│   ├── __main__.py                  python -m entry point
│   ├── cli.py                       Sample/full command-line interface
│   ├── config.py                    Study dates, windows, and paths
│   ├── data.py                      Source linkage and analysis frame
│   ├── did.py                       Primary and sensitivity DiD
│   ├── pipeline.py                  End-to-end orchestration
│   ├── prediction.py                Temporal validation and forecasts
│   ├── reporting/
│   │   ├── __init__.py              Reporting API
│   │   ├── descriptive_figures.py   Composition and citation plots
│   │   ├── did_figures.py           DiD and pretrend plots
│   │   ├── excel.py                 Excel table writers
│   │   ├── manuscript_components.py Manuscript cell and row builders
│   │   ├── manuscript_tables.py     Manuscript table specifications
│   │   ├── plotting.py              Shared plot style and export
│   │   ├── prediction_figures.py    Validation and forecast plots
│   │   ├── text.py                  Figure legends and result notes
│   │   └── word.py                  Word/OOXML table renderer
│   ├── sample.py                    Deterministic synthetic analysis
│   ├── summaries.py                 Descriptive summary tables
│   ├── utils.py                     Shared parsing and formatting helpers
│   └── validation.py                Analysis checks and output manifest
├── notebooks/
│   └── analysis.ipynb               Thin pipeline runner
├── tests/
│   └── test_smoke.py                Synthetic-data smoke tests
├── .github/workflows/
│   └── smoke-test.yml               CI lint, tests, and sample run
├── .env.example                     Full-mode settings template
├── .gitattributes                   Line-ending and binary-file rules
├── .gitignore                       Private data and output exclusions
├── pyproject.toml                   Dependencies, CLI, and tool settings
└── README.md                        Study and execution guide
```

## Code structure

- `cli.py` dispatches sample or full mode
- `pipeline.py` owns execution order
- `config.py` defines dates, windows, and paths; `data.py` builds the analysis frame
- `summaries.py`, `did.py`, and `prediction.py` produce descriptive, DiD, and forecast results
- `validation.py` enforces hard checks; `reporting/` writes outputs
- `sample.py` supplies deterministic synthetic data for local and CI runs

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
