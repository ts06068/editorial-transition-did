"""Allow ``python -m editorial_transition_did`` execution."""

from .cli import main

# Propagate the CLI return code when the package is invoked with ``python -m``.
raise SystemExit(main())
