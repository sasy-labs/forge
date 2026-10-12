"""Translate a tau2 domain's natural-language policy into Soufflé Datalog.

Reads ``experiments/tau2-bench/data/tau2/domains/<domain>/policy.md`` as the
NL spec, sends the relevant tau2 codebase paths along as agent context,
and writes the generated ``<domain>_policy.dl`` + ``<domain>_functors.cpp``
into ``experiments/tau2/`` (where ``serve-<domain>`` and
``upload-<domain>`` already look).

Usage::

    SASY_TRANSLATE_URL=http://127.0.0.1:8081 \
    ANTHROPIC_API_KEY=... \
    uv run python scripts/evaluation/translate_tau2_domain.py retail

Defaults to the Opus model and forbids ``@llm_check_fn`` so the resulting
policy is purely deterministic — easier to iterate on without LLM
oracle costs in the local serve loop.

Skips the existing artifacts unless ``--overwrite`` is passed, so a
re-run after a partial failure won't silently clobber a working policy.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


_DEFAULT_INSTRUCTIONS = """\
Constraints for this translation:

- Do not emit any rule that uses the ``@llm_check_fn`` functor or any
  other LLM-oracle functor. The policy must be fully deterministic —
  every rule body must reduce to structural checks against
  SentMessage / ToolResult / Edge / EdgeData / CurrentDepends and
  string/JSON helper functors.
- Mirror the conventions used by ``experiments/tau2/airline_policy.dl``
  in this repo: dot-notation for accessing message fields, custom
  C++ functors for any JSON extraction the policy needs, ``IsAuthorized``
  as the default fall-through, ``Unauthorized`` rules with annotated
  denial reasons for blocked paths.
- The enforcement target is the *agent* (``LLMAgent``); the
  user-simulator is a separate LLM whose messages also flow through
  ``SentMessage`` and must NEVER be treated as agent context. Every
  rule that looks at agent-originated content must filter on
  ``msg.agent = "LLMAgent"``. Every rule that looks at end-user
  utterances arriving at the LLMAgent must additionally filter on
  ``msg.agent_role = $UserRole()`` — the airline policy's
  ``UserConfirmed`` / ``UserRequestedBags`` predicates are the
  canonical pattern. Tool-call requests and tool results don't need
  the agent_role filter, but they DO need ``msg.agent = "LLMAgent"``
  so a malicious user-simulator can't impersonate an agent action
  by emitting tool-call-shaped text.
"""


def _domain_paths(domain: str) -> tuple[Path, list[Path], Path, Path]:
    """Return (policy_md, codebase_paths, policy_dl_out, functors_out)."""
    bench = REPO_ROOT / "experiments" / "tau2-bench"
    policy_md = bench / "data" / "tau2" / "domains" / domain / "policy.md"
    out_dir = REPO_ROOT / "experiments" / "tau2"

    # Agent context: the domain's tool/data definitions plus shared
    # message schemas. Keeping the zip small focuses the translator
    # on the right surface area.
    src = bench / "src" / "tau2"
    codebase_paths = [
        src / "domains" / domain,
        src / "data_model",
        src / "environment",
    ]

    return (
        policy_md,
        codebase_paths,
        out_dir / f"{domain}_policy.dl",
        out_dir / f"{domain}_functors.cpp",
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "domain",
        help="tau2 domain name (e.g. 'retail', 'airline').",
    )
    parser.add_argument(
        "--model",
        default="opus",
        choices=("haiku", "sonnet", "opus"),
        help="Anthropic model used by the translator (default: opus).",
    )
    parser.add_argument(
        "--instructions",
        default=_DEFAULT_INSTRUCTIONS,
        help="Extra agent instructions; defaults to the no-llm-check"
        " preamble. Pass an empty string to disable.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing policy/functors artifacts in"
        " experiments/tau2/.",
    )
    parser.add_argument(
        "--stage1-only",
        action="store_true",
        help="Stop after the agent analysis stage; cheap dry-run for"
        " checking the codebase scope.",
    )
    args = parser.parse_args()

    policy_md, codebase_paths, dl_out, fn_out = _domain_paths(args.domain)

    if not policy_md.is_file():
        sys.stderr.write(f"NL policy not found: {policy_md}\n")
        return 2

    missing = [p for p in codebase_paths if not p.exists()]
    if missing:
        sys.stderr.write(
            "Missing codebase paths:\n"
            + "\n".join(f"  - {p}" for p in missing)
            + "\n"
        )
        return 2

    if not args.overwrite and (dl_out.exists() or fn_out.exists()):
        sys.stderr.write(
            f"Refusing to overwrite existing artifacts:\n"
            f"  - {dl_out}\n  - {fn_out}\n"
            "Pass --overwrite to replace them.\n"
        )
        return 2

    # Lazy import so --help works without the SDK installed.
    from sasy.policy.translate import translate

    nl_policy = policy_md.read_text()
    print(
        f"Translating {args.domain}: NL policy {len(nl_policy):,} chars,"
        f" {len(codebase_paths)} codebase paths, model={args.model}"
    )

    result = translate(
        policy=nl_policy,
        codebase_paths=codebase_paths,
        codebase_root=REPO_ROOT,
        instructions=args.instructions,
        model=args.model,
        stage1_only=args.stage1_only,
        on_progress=lambda stage, elapsed: print(
            f"  [{elapsed:>5.0f}s] {stage}"
        ),
    )

    print()
    result.print_summary()
    print()

    if args.stage1_only:
        if result.agent_summary_md:
            summary_path = REPO_ROOT / "outputs" / f"{args.domain}_agent_summary.md"
            result.save_summary(summary_path)
            print(f"Stage-1 summary saved to {summary_path}")
        return 0

    if result.status != "success" or not result.policy_dl:
        sys.stderr.write("Translation did not produce a policy; aborting save.\n")
        return 1

    dl_out.parent.mkdir(parents=True, exist_ok=True)
    result.save_policy(dl_out)
    print(f"Policy: {dl_out}")
    if result.functors_cpp:
        result.save_functors(fn_out)
        print(f"Functors: {fn_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
