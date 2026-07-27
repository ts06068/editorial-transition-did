"""Allow ``python -m editorial_transition_did`` execution."""

from .cli import main

# Do not hide CLI failures from the calling shell.
raise SystemExit(main())
