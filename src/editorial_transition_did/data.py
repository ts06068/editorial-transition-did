"""Load, merge, derive, and cache the article-level analysis dataset."""

from __future__ import annotations

import json
import os
import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pymssql
from dotenv import load_dotenv
from tqdm.auto import tqdm

from .config import (
    ANALYSIS_CUTOFF,
    ATOMIC_CATEGORIES,
    JOURNAL_DISPLAY,
    JOURNAL_ORDER,
    NORA_CATEGORIES,
    PUBLICATION_END,
    PUBLICATION_START,
    AnalysisPaths,
)
from .utils import normalize_doi, parse_issue_date

# Treat this as the reviewed workbook schema; missing columns must fail at ingestion.
ARTICLE_COLUMNS = [
    "journal_code",
    "journal",
    "publication_year",
    "cover_date",
    "volume",
    "issue",
    "paper_id",
    "eid",
    "doi_norm",
    "title",
    "issue_url",
    "html_file",
    "scopus_subtype",
    "scopus_subtype_description",
    "final_standardized_class",
    "final_standardized_category",
    "final_classification_source",
    "final_classification_status",
]

# These database fields are attached only after DOI/EID record linkage.
DB_COLUMNS = [
    "db_paper_id",
    "db_eid",
    "db_doi",
    "db_title",
    "db_journal",
    "db_pub_date",
    "db_subtype",
    "db_subtype_description",
    "db_citedby_count",
    "c3",
    "c6",
    "c12",
    "c15",
    "c18",
    "c24",
    "citations_at_snapshot",
    "source_database",
    "source_database_key",
]

ANALYSIS_EXPORT_COLUMNS = [
    "analysis_record_id",
    "doi_norm",
    "eid",
    "title",
    "journal_display",
    "publication_date",
    "publication_date_source",
    "category",
    "c3_obs",
    "c6_obs",
    "c12_obs",
    "c15_obs",
    "c18_obs",
    "c24_obs",
    "increment_3m",
    "increment_6m",
    "increment_12m",
    "source_database",
]


def _connect(database: str) -> pymssql.Connection:
    """Open a read-only analysis connection using environment credentials."""

    return pymssql.connect(
        server=os.environ["MSSQL_SERVER"],
        port=int(os.getenv("MSSQL_PORT", "1433")),
        user=os.environ["MSSQL_SCHEMA_USER"],
        password=os.environ["MSSQL_SCHEMA_PASSWORD"],
        database=database,
        login_timeout=20,
        timeout=600,
        charset="UTF-8",
    )


# LEFT JOIN retains uncited papers. Distinct citing EIDs are censored at one cutoff.
DB_AGGREGATE_QUERY = """
SELECT
    CAST(p.paper_id AS BIGINT) AS db_paper_id,
    p.eid AS db_eid,
    p.[prism:doi] AS db_doi,
    p.[dc:title] AS db_title,
    p.[prism:publicationName] AS db_journal,
    p.[prism:coverDate] AS db_pub_date,
    p.[subtype] AS db_subtype,
    p.[subtypeDescription] AS db_subtype_description,
    p.[citedby-count] AS db_citedby_count,
    COUNT(DISTINCT CASE
        WHEN c.[prism:coverDate] >= p.[prism:coverDate]
         AND c.[prism:coverDate] <= DATEADD(month, 3, p.[prism:coverDate])
         AND c.[prism:coverDate] <= %s THEN c.eid END) AS c3,
    COUNT(DISTINCT CASE
        WHEN c.[prism:coverDate] >= p.[prism:coverDate]
         AND c.[prism:coverDate] <= DATEADD(month, 6, p.[prism:coverDate])
         AND c.[prism:coverDate] <= %s THEN c.eid END) AS c6,
    COUNT(DISTINCT CASE
        WHEN c.[prism:coverDate] >= p.[prism:coverDate]
         AND c.[prism:coverDate] <= DATEADD(month, 12, p.[prism:coverDate])
         AND c.[prism:coverDate] <= %s THEN c.eid END) AS c12,
    COUNT(DISTINCT CASE
        WHEN c.[prism:coverDate] >= p.[prism:coverDate]
         AND c.[prism:coverDate] <= DATEADD(month, 15, p.[prism:coverDate])
         AND c.[prism:coverDate] <= %s THEN c.eid END) AS c15,
    COUNT(DISTINCT CASE
        WHEN c.[prism:coverDate] >= p.[prism:coverDate]
         AND c.[prism:coverDate] <= DATEADD(month, 18, p.[prism:coverDate])
         AND c.[prism:coverDate] <= %s THEN c.eid END) AS c18,
    COUNT(DISTINCT CASE
        WHEN c.[prism:coverDate] >= p.[prism:coverDate]
         AND c.[prism:coverDate] <= DATEADD(month, 24, p.[prism:coverDate])
         AND c.[prism:coverDate] <= %s THEN c.eid END) AS c24,
    COUNT(DISTINCT CASE
        WHEN c.[prism:coverDate] >= p.[prism:coverDate]
         AND c.[prism:coverDate] <= %s THEN c.eid END) AS citations_at_snapshot
FROM dbo.paper AS p
LEFT JOIN dbo.relationship AS r
    ON p.paper_id = r.paper_id_1
   AND r.relationship = 'citing'
LEFT JOIN dbo.citation AS c
    ON r.paper_id_2 = c.paper_id
WHERE p.[prism:publicationName] IN (
        'Circulation',
        'European Heart Journal',
        'Journal of the American College of Cardiology'
    )
  AND p.[prism:coverDate] >= %s
  AND p.[prism:coverDate] < %s
GROUP BY
    p.paper_id,
    p.eid,
    p.[prism:doi],
    p.[dc:title],
    p.[prism:publicationName],
    p.[prism:coverDate],
    p.[subtype],
    p.[subtypeDescription],
    p.[citedby-count]
"""


def _normalize_citation_records(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Combine database extracts using the original date-dependent priority rule."""

    combined = pd.concat(frames, ignore_index=True)
    combined["doi_norm"] = combined["db_doi"].map(normalize_doi)
    combined["db_eid"] = combined["db_eid"].astype("string").str.strip()
    combined["db_pub_date"] = pd.to_datetime(combined["db_pub_date"], errors="coerce")
    # Source authority changes at 2025: refreshed from then on, legacy before it.
    combined["db_priority"] = np.where(
        combined["db_pub_date"].ge(pd.Timestamp("2025-01-01")),
        np.where(combined["source_database_key"].eq("SCOPUS_DB_2026"), 0, 1),
        np.where(combined["source_database_key"].eq("SCOPUS_DB"), 0, 1),
    )
    # DOI is the primary identity key. EID keeps DOI-missing records linkable.
    combined["record_key"] = np.where(
        combined["doi_norm"].notna(),
        "doi:" + combined["doi_norm"].fillna(""),
        "eid:" + combined["db_eid"].fillna(""),
    )
    return (
        combined.sort_values(["record_key", "db_priority"])
        .drop_duplicates("record_key", keep="first")
        .drop(columns=["db_priority", "record_key"])
        .reset_index(drop=True)
    )


def query_citation_aggregates(
    paths: AnalysisPaths,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Load cached citation aggregates or query the two configured MSSQL databases."""

    paths.create_directories()
    cache_path = paths.cache_dir / "citation_aggregates_2008_2026_06.pkl"
    meta_path = paths.cache_dir / "citation_aggregates_metadata.json"
    if cache_path.exists() and not force_refresh:
        return pd.read_pickle(cache_path)

    load_dotenv(paths.env_file)
    cutoff_param = ANALYSIS_CUTOFF.strftime("%Y-%m-%d")
    # Parameter order is fixed: seven cutoffs, then inclusive lower/exclusive upper dates.
    params = tuple([cutoff_param] * 7) + (
        PUBLICATION_START.strftime("%Y-%m-%d"),
        (PUBLICATION_END + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
    )
    frames: list[pd.DataFrame] = []
    database_keys = ["SCOPUS_DB", "SCOPUS_DB_2026"]
    for key in tqdm(database_keys, desc="Querying MSSQL databases"):
        database = os.environ[key]
        connection = _connect(database)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                frame = pd.read_sql_query(DB_AGGREGATE_QUERY, connection, params=params)
        finally:
            connection.close()
        frame["source_database"] = database
        frame["source_database_key"] = key
        frames.append(frame)

    combined = _normalize_citation_records(frames)
    combined.to_pickle(cache_path)
    metadata = {
        "created_utc": datetime.now(UTC).isoformat(),
        "analysis_cutoff": ANALYSIS_CUTOFF.strftime("%Y-%m-%d"),
        "publication_end": PUBLICATION_END.strftime("%Y-%m-%d"),
        "rows": int(len(combined)),
        "databases": [os.environ[key] for key in database_keys],
        "query_definition": (
            "Distinct citing EIDs with source cover date between target cover date and "
            "each landmark, censored at analysis cutoff."
        ),
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return combined


def _attach_database_columns(articles: pd.DataFrame, db: pd.DataFrame) -> pd.DataFrame:
    """Attach database fields by normalized DOI, falling back to EID per field."""

    by_doi = (
        db.loc[db["doi_norm"].notna()].drop_duplicates("doi_norm").set_index("doi_norm", drop=False)
    )
    by_eid = db.loc[db["db_eid"].notna()].drop_duplicates("db_eid").set_index("db_eid", drop=False)
    output = articles.copy()
    article_eid = output["eid"].astype("string").str.strip()
    for column in DB_COLUMNS:
        # Fall back per field; a DOI-matched row can still lack an EID-held value.
        doi_values = output["doi_norm"].map(by_doi[column])
        eid_values = article_eid.map(by_eid[column])
        output[column] = doi_values.where(doi_values.notna(), eid_values)
    return output


def load_classification_articles(paths: AnalysisPaths) -> pd.DataFrame:
    """Read and normalize the three-journal classification workbook."""

    if not paths.input_xlsx.exists():
        raise FileNotFoundError(f"Classification file not found: {paths.input_xlsx}")
    articles = pd.read_excel(paths.input_xlsx, sheet_name="articles", usecols=ARTICLE_COLUMNS)
    articles["journal_code"] = articles["journal_code"].astype("string").str.lower().str.strip()
    articles = articles[articles["journal_code"].isin(JOURNAL_ORDER)].copy()
    articles["doi_norm"] = articles["doi_norm"].map(normalize_doi)
    articles["cover_date"] = pd.to_datetime(articles["cover_date"], errors="coerce")
    articles["category"] = articles["final_standardized_category"].astype("string")
    return articles


def build_analysis_dataset(
    articles: pd.DataFrame,
    db: pd.DataFrame,
    html_root: os.PathLike[str] | str,
) -> pd.DataFrame:
    """Purely derive the analysis fields after source records have been loaded."""

    data = _attach_database_columns(articles, db)
    # HTML is the expensive fallback. Parse each missing-date file once.
    issue_date_map = {
        value: parse_issue_date(value, root=Path(html_root))
        for value in tqdm(
            data.loc[data["cover_date"].isna(), "html_file"].dropna().unique(),
            desc="Recovering publisher issue dates",
        )
    }
    data["publisher_issue_date"] = data["html_file"].map(issue_date_map)
    # Date precedence is curated workbook, MSSQL, then publisher HTML.
    data["publication_date"] = data["cover_date"]
    data["publication_date"] = data["publication_date"].where(
        data["publication_date"].notna(), data["db_pub_date"]
    )
    data["publication_date"] = data["publication_date"].where(
        data["publication_date"].notna(), data["publisher_issue_date"]
    )
    data["publication_date"] = pd.to_datetime(data["publication_date"], errors="coerce")
    data["publication_date_source"] = np.select(
        [
            data["cover_date"].notna(),
            data["db_pub_date"].notna(),
            data["publisher_issue_date"].notna(),
        ],
        ["classification", "mssql", "publisher issue HTML"],
        default="missing",
    )
    data = data[
        data["publication_date"].between(PUBLICATION_START, PUBLICATION_END, inclusive="both")
    ].copy()
    data["publication_year"] = data["publication_date"].dt.year
    data["publication_month"] = data["publication_date"].dt.to_period("M").dt.to_timestamp()
    data["publication_quarter"] = data["publication_date"].dt.to_period("Q").astype(str)
    data["journal_display"] = data["journal_code"].map(JOURNAL_DISPLAY)
    data["is_jacc"] = data["journal_code"].eq("jacc").astype(int)
    data["is_nora"] = data["category"].isin(NORA_CATEGORIES)

    for months in [3, 6, 12, 15, 18, 24]:
        # Do not expose a landmark until its full follow-up ends by the cutoff.
        complete = data["publication_date"].map(
            lambda value, months=months: (
                value + pd.DateOffset(months=months) <= ANALYSIS_CUTOFF
                if pd.notna(value)
                else False
            )
        )
        data[f"complete_{months}m"] = complete
        data[f"c{months}_obs"] = pd.to_numeric(data[f"c{months}"], errors="coerce").where(complete)

    # Predict only citations accrued after C12; C12 itself is already observed.
    data["increment_3m"] = data["c15_obs"] - data["c12_obs"]
    data["increment_6m"] = data["c18_obs"] - data["c12_obs"]
    data["increment_12m"] = data["c24_obs"] - data["c12_obs"]
    # Study years roll over in June, not January.
    data["cohort_year"] = np.where(
        data["publication_date"].dt.month.ge(6),
        data["publication_date"].dt.year,
        data["publication_date"].dt.year - 1,
    )
    data["analysis_record_id"] = np.where(
        data["doi_norm"].notna(),
        "doi:" + data["doi_norm"].fillna(""),
        "eid:" + data["eid"].astype("string").fillna(""),
    )
    return data


def build_source_qa(articles: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    """Build the original source and follow-up quality-assurance counts."""

    # Negative increments flag linkage or source-refresh inconsistencies.
    qa_rows = [
        ("Classification rows for three journals", len(articles)),
        ("Analysis rows through 2026-06-30", len(data)),
        ("Unique analysis record IDs", data["analysis_record_id"].nunique()),
        ("Rows matched to an MSSQL paper", data["db_paper_id"].notna().sum()),
        ("Publisher-only rows without an MSSQL paper", data["db_paper_id"].isna().sum()),
        (
            "Rows dated from publisher issue HTML",
            data["publication_date_source"].eq("publisher issue HTML").sum(),
        ),
        ("Rows with complete classification", data["category"].isin(ATOMIC_CATEGORIES).sum()),
        ("Rows with missing publication date", data["publication_date"].isna().sum()),
        ("Rows with complete 1-year follow-up", data["c12_obs"].notna().sum()),
        ("Rows with complete 2-year follow-up", data["c24_obs"].notna().sum()),
        ("Negative 3-month increments", data["increment_3m"].lt(0).sum()),
        ("Negative 6-month increments", data["increment_6m"].lt(0).sum()),
        ("Negative 12-month increments", data["increment_12m"].lt(0).sum()),
    ]
    qa = pd.DataFrame(qa_rows, columns=["check", "value"])
    qa["value"] = qa["value"].astype(int)
    return qa


def prepare_analysis_data(
    paths: AnalysisPaths,
    force_refresh_db: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load all sources, derive analysis variables, cache them, and return QA counts."""

    paths.create_directories()
    articles = load_classification_articles(paths)
    db = query_citation_aggregates(paths, force_refresh=force_refresh_db)
    data = build_analysis_dataset(articles, db, paths.html_root)
    qa = build_source_qa(articles, data)

    data.to_pickle(paths.cache_dir / "analysis_dataset.pkl")
    data[ANALYSIS_EXPORT_COLUMNS].to_csv(
        paths.cache_dir / "analysis_dataset.csv.gz",
        index=False,
        compression="gzip",
    )
    return data.reset_index(drop=True), qa
