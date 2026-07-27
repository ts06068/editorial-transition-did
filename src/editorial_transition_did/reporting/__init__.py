"""Tabular and text reporting helpers for the analysis pipeline."""

# Re-export only the small, stable reporting surface used by callers; detailed
# figure and manuscript renderers remain available from their focused modules.
from .excel import (
    write_combined_workbook,
    write_flat_table,
    write_table_1,
    write_table_2,
    write_table_5,
)
from .text import write_figure_legends, write_results_readme

# An explicit export list keeps wildcard imports from exposing private formatters.
__all__ = [
    "write_combined_workbook",
    "write_figure_legends",
    "write_flat_table",
    "write_results_readme",
    "write_table_1",
    "write_table_2",
    "write_table_5",
]
