#!/usr/bin/env bash
#
# Set up a dedicated Python environment for AgentDojo benchmarks.
#
# AgentDojo is not a workspace member, so the main repository's .venv
# doesn't have it. This script creates a repo-local venv at
# experiments/agentdojo/.venv (gitignored) with AgentDojo and the sasy SDK.
#
# Usage:
#   ./experiments/agentdojo/setup.sh
#
# After setup, run benchmarks with:
#   uv run scripts/evaluation/run_benchmark.py --targets agentdojo

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

test -f "$REPO_ROOT/sasy/sdk/python/pyproject.toml" || {
    echo "Run python3 scripts/setup_evaluation.py first." >&2; exit 1;
}
echo "Setting up AgentDojo environment at $VENV_DIR"

# Create the venv with Python 3.12. --no-project prevents consulting the
# workspace's pyproject.toml (which may conflict); --allow-existing preserves
# packages when re-running setup.
uv venv --no-project --allow-existing --python 3.12 "$VENV_DIR"

# Install dependencies. Use system uv with --python to target the venv;
# uv venv does not install .venv/bin/uv.
uv pip install --python "$VENV_DIR/bin/python" \
    agentdojo==0.1.35 \
    -e "$REPO_ROOT/sasy/sdk/python[otel]"

echo ""
echo "Setup complete."
echo "  Python:  $VENV_DIR/bin/python"
echo "  Version: $("$VENV_DIR/bin/python" --version)"
echo ""
echo "The benchmark runner will auto-detect this interpreter."
echo "To verify: $VENV_DIR/bin/python -c 'import agentdojo, sasy; print(\"OK\")'"
