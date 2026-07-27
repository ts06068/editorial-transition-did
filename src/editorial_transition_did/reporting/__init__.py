"""Tabular and text reporting helpers for the analysis pipeline."""

# Keep the package API small; specialized renderers belong to their modules.
from .excel import (
    write_combined_workbook,
    write_flat_table,
    write_table_1,
    write_table_2,
    write_table_5,
)
from .text import write_figure_legends, write_results_readme

# Do not leak private formatters through wildcard imports.
__all__ = [
    "write_combined_workbook",
    "write_figure_legends",
    "write_flat_table",
    "write_results_readme",
    "write_table_1",
    "write_table_2",
    "write_table_5",
]
