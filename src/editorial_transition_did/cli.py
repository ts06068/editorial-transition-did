"""Command-line entry points for full and synthetic analyses."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

from dotenv import load_dotenv

from .config import AnalysisPaths
from .sample import run_sample

MODES = ("full", "sample")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="editorial-transition-did",
        description="Run the editorial-transition citation analysis.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=MODES,
        help="Analysis subcommand. Equivalent to --mode.",
    )
    parser.add_argument(
        "--mode",
        choices=MODES,
        help="Compatibility form of the full or sample subcommand.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Classification workbook; overrides EDITORIAL_DID_INPUT_PATH.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (analysis_results for full; outputs/sample for sample).",
    )
    parser.add_argument(
        "--html-root",
        type=Path,
        default=None,
        help="Publisher HTML root; overrides EDITORIAL_DID_HTML_ROOT.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Environment file containing MSSQL connection settings.",
    )
    parser.add_argument(
        "--force-refresh-db",
        action="store_true",
        help="Ignore the citation cache and query MSSQL again in full mode.",
    )
    parser.add_argument(
        "--primary-bootstrap-repetitions",
        type=int,
        default=2000,
        metavar="N",
        help="Bootstrap repetitions for the primary full-analysis DiD.",
    )
    parser.add_argument(
        "--robustness-bootstrap-repetitions",
        type=int,
        default=1500,
        metavar="N",
        help="Bootstrap repetitions for full-analysis robustness checks.",
    )
    parser.add_argument(
        "--bootstrap-repetitions",
        type=int,
        default=200,
        metavar="N",
        help="Bootstrap repetitions for sample mode.",
    )
    return parser


def _resolve_path(value: Path, project_root: Path) -> Path:
    # Resolve relative CLI and .env paths against the invocation directory.
    expanded = value.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (project_root / expanded).resolve()


def _selected_mode(parser: argparse.ArgumentParser, args: argparse.Namespace) -> str:
    # Support both subcommands and the legacy --mode spelling without ambiguity.
    if args.command and args.mode and args.command != args.mode:
        parser.error("the subcommand and --mode must select the same analysis")
    mode = args.command or args.mode
    if mode is None:
        parser.error("choose the 'full' or 'sample' subcommand (or pass --mode)")
    return str(mode)


def _run_full(args: argparse.Namespace) -> None:
    if args.primary_bootstrap_repetitions < 1:
        raise ValueError("primary bootstrap repetitions must be at least 1")
    if args.robustness_bootstrap_repetitions < 1:
        raise ValueError("robustness bootstrap repetitions must be at least 1")

    project_root = Path.cwd().resolve()
    env_file = _resolve_path(args.env_file, project_root)
    # Load .env first so AnalysisPaths can apply environment-backed defaults.
    load_dotenv(env_file)
    defaults = AnalysisPaths.from_environment(project_root)
    # Explicit command-line paths take precedence over every .env default.
    paths = AnalysisPaths(
        project_root=project_root,
        input_xlsx=(_resolve_path(args.input, project_root) if args.input else defaults.input_xlsx),
        html_root=(
            _resolve_path(args.html_root, project_root) if args.html_root else defaults.html_root
        ),
        env_file=env_file,
        output_dir=(
            _resolve_path(args.output_dir, project_root) if args.output_dir else defaults.output_dir
        ),
    )
    paths.create_directories()
    os.environ.setdefault("MPLCONFIGDIR", str(paths.project_root / ".matplotlib"))
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

    # Delay heavy plotting/model imports until full mode is actually requested.
    from .pipeline import run_all

    run_all(
        paths=paths,
        force_refresh_db=args.force_refresh_db,
        primary_bootstrap_repetitions=args.primary_bootstrap_repetitions,
        robustness_bootstrap_repetitions=args.robustness_bootstrap_repetitions,
    )
    print(f"Full analysis complete: {paths.output_dir}")


def _run_sample(args: argparse.Namespace) -> None:
    # Sample mode deliberately bypasses private workbooks, HTML, and MSSQL.
    output_dir = args.output_dir or Path("outputs") / "sample"
    results = run_sample(
        output_dir=output_dir,
        bootstrap_repetitions=args.bootstrap_repetitions,
    )
    print(f"Sample analysis complete: {results['output_dir']}")


def main(argv: Sequence[str] | None = None) -> int:
    """Parse command-line arguments and run the selected analysis."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    mode = _selected_mode(parser, args)
    if mode == "full":
        _run_full(args)
    else:
        _run_sample(args)
    return 0
