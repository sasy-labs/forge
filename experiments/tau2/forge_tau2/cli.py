# Wraps the tau2 CLI with SASY observability instrumentation.

# First import: sets GRPC_ENABLE_FORK_SUPPORT=false before anything
# below loads gRPC (see cli_harness).
from forge_tau2.cli_harness import (
    install_harness,
    resolve_domain_policy,
)

from tau2.cli import main

import sasy.instrumentation as instrumentation
import sasy
import logging
from dotenv import load_dotenv
import os

load_dotenv()
# Explicit provider settings from the benchmark or shell take precedence.
load_dotenv(".env-credentials")

# If using http instrumentation for OpenAI, set a fake key for litellm
if not os.environ.get("AZURE_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = "fake"

logger = logging.getLogger(__name__)

def feedback_callback(accumulator, output, _):
    """Log and append authorization feedback to responses."""
    if accumulator.has_denials():
        feedback_msg = accumulator.format_for_llm()
        logger.warning(f"POLICY VIOLATION DETECTED:\n{feedback_msg}")

        if output is not None:
            original = output.content or ""
            output.content = original + f"\n\n{feedback_msg}"


def run():
    """Run tau2 with the shared prompt and runtime authorization instrumentation."""
    # Vertex OAuth token reuse for high-concurrency runs (see
    # litellm_patches).
    from forge_tau2.litellm_patches import apply_litellm_patches
    apply_litellm_patches()

    # Per-trial SASY session pinned to the domain policy, plus the
    # TAU2_EVALUATION_TYPE / TAU2_NL_JUDGE_MODEL overrides.
    install_harness(sasy_sessions=True, policy=resolve_domain_policy())

    instrumentation.configure(
        # Use mTLS for authentication
        auth_hook=sasy.auth.NoAuthHook(),
        log_denials=True,
        feedback_callback=feedback_callback,
    )

    # HTTP instrumentation routes LLM API calls through the reference monitor.
    # Disabled by default; enable with INSTRUMENT_HTTP=1.
    use_http = bool(os.environ.get("INSTRUMENT_HTTP"))
    instrumentation.instrument(http=use_http)
    from forge_tau2.instrumentation import instrument as instrument_tau2
    instrument_tau2()
    main()


if __name__ == "__main__":
    run()
