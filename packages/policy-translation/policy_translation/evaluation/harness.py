"""Run Souffle policies against test scenarios in interpreted mode.

Generates RFC 4180 fact files from JSON scenario definitions, invokes
Souffle in interpreted mode, and compares output relations against
expected results. Supports all Action ADT variants (CallTool,
HTTPRequest, SendAttempt).

Usage:
    python3 scripts/policy_test_harness.py \\
        --policy path/to/policy.dl \\
        --scenarios path/to/tests.json

Exit code 0 = all pass, 1 = any failure.

Reference: sasy-services/souffle/interpreted_shim.cpp (ADT format)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ── Paths and the sugar pass (from the package's one seam) ─────
#
# A module inside a package cannot count its own parent directories to
# find the checkout the way a script under scripts/ could, and the
# plan's section 1.7 puts every use of sugar.py behind one adapter. The
# paths and the call both live there now, each with the comment it
# carried here.
from policy_translation.souffle_text import FUNCTORS_SRC, HARNESS_CACHE_DIR, SUGAR_SCRIPT, desugar

# The name this module has always exported for the pass's own path.
SUGAR_PY = SUGAR_SCRIPT


# ── RFC 4180 / ADT serialization (from interpreted_shim.cpp) ─


def rec_escape(s: str) -> str:
    """Quote a symbol for Souffle record/ADT fields.

    Matches the C++ ``rec_escape()`` in interpreted_shim.cpp:
    wraps in double quotes if the value contains delimiters.
    """
    if not s:
        return '""'
    delimiters = set(',[](){} "\\')
    if not any(c in delimiters or ord(c) < 0x20 for c in s):
        return s
    out = ['"']
    for c in s:
        if c == '"':
            out.append('\\"')
        elif c == "\\":
            out.append("\\\\")
        else:
            out.append(c)
    out.append('"')
    return "".join(out)


def adt_agent_role(role: str) -> str:
    """Serialize an AgentRole ADT branch (nullary)."""
    mapping = {
        "system": "$SystemRole",
        "user": "$UserRole",
        "agent": "$AgentType",
    }
    return mapping.get(role, "$Assistant")


def adt_action(action: dict[str, Any]) -> str:
    """Serialize an Action ADT to Souffle record syntax."""
    atype = action["type"]
    if atype == "CallTool":
        fn = rec_escape(action["fn_name"])
        args = rec_escape(action.get("args", "{}"))
        return f"$CallTool({fn}, {args})"
    elif atype == "HTTPRequest":
        url = rec_escape(action["url"])
        body = rec_escape(action.get("body", ""))
        headers = rec_escape(action.get("headers_json", "[]"))
        return f"$HTTPRequest({url}, {body}, {headers})"
    elif atype == "SendAttempt":
        contents = rec_escape(action.get("contents", ""))
        tools = '"[]"'
        agent = rec_escape(action.get("agent", ""))
        role = adt_agent_role(action.get("agent_role", "assistant"))
        entity = rec_escape(action.get("entity", ""))
        return f'$SendAttempt([{contents}, {tools}, {agent}, {role}, {entity}, ""])'
    else:
        raise ValueError(f"Unknown action type: {atype}")


def adt_message(msg: dict[str, Any]) -> str:
    """Serialize a Message record for SentMessage facts."""
    contents = rec_escape(msg.get("contents", ""))
    # Real tool calls, not a hardcoded empty list. Pinning this to "[]"
    # does not make a rule that reads tools_json fail — it makes it
    # evaluate against an empty tool list and pass, then behave
    # differently against the live graph, where the field is populated.
    # Convergence then certifies a policy on evidence the deployment
    # will not reproduce.
    tools = rec_escape(msg.get("tools_json") or "[]")
    agent = rec_escape(msg.get("agent", ""))
    role = adt_agent_role(msg.get("agent_role", "assistant"))
    entity = rec_escape(msg.get("entity", ""))
    principal = rec_escape(msg.get("principal", ""))
    return f"[{contents}, {tools}, {agent}, {role}, {entity}, {principal}]"


# ── Fact file generation ───────────────────────────────────────


def _write_rfc4180_rows(
    path: Path,
    rows: Iterable[Sequence[str]],
) -> None:
    """Write rows using the format expected by common_policy.dl.

    Args:
        path: Destination ``.facts`` file.
        rows: String fields to serialize as RFC 4180 CSV records.
    """
    with path.open("w", encoding="utf-8", newline="") as fact_file:
        writer = csv.writer(fact_file, dialect="excel")
        writer.writerows(rows)


def write_fact_files(scenario: dict[str, Any], fact_dir: Path) -> None:
    """Write all RFC 4180 ``.facts`` files for a single scenario."""
    fact_dir.mkdir(parents=True, exist_ok=True)

    action = scenario.get("action")
    action_rows = [["0", adt_action(action)]] if action else []
    _write_rfc4180_rows(fact_dir / "Actions.facts", action_rows)

    entity = scenario.get("entity", "")
    principal = scenario.get("principal", entity)
    roles = scenario.get("roles", [])
    principal_rows = [[principal]] if principal else []
    entity_rows = [[entity]] if entity else []
    role_rows = [[principal, role] for role in roles] if principal else []
    _write_rfc4180_rows(fact_dir / "Principal.facts", principal_rows)
    _write_rfc4180_rows(fact_dir / "Entity.facts", entity_rows)
    _write_rfc4180_rows(fact_dir / "PrincipalRole.facts", role_rows)

    graph = scenario.get("graph", {})
    current_rows = [[nid] for nid in graph.get("current_nodes", [])]
    _write_rfc4180_rows(fact_dir / "Current.facts", current_rows)

    edge_rows = [[edge[0], edge[1]] for edge in graph.get("edges", [])]
    _write_rfc4180_rows(fact_dir / "Edge.facts", edge_rows)

    message_rows = [[msg["id"], adt_message(msg)] for msg in graph.get("sent_messages", [])]
    _write_rfc4180_rows(fact_dir / "SentMessage.facts", message_rows)

    # Match the live adapter's sparse message metadata relation.
    metadata_rows = [
        [msg["id"], msg["metadata"]]
        for msg in graph.get("sent_messages", [])
        if msg.get("metadata") is not None
    ]
    _write_rfc4180_rows(fact_dir / "MessageMetadata.facts", metadata_rows)

    tool_result_rows = [[tr["id"], tr["fn_name"], tr.get("args", "{}")] for tr in graph.get("tool_results", [])]
    _write_rfc4180_rows(
        fact_dir / "ToolResult.facts",
        tool_result_rows,
    )

    # EdgeData.facts — sparse edge metadata, record
    # EdgeDataRecord = [proximal, message_index], with -1 sentinels for
    # unset fields (matches insert_edge_data in evaluator_shim.cpp).
    # Leaving this empty does not fail loudly: it silently starves any
    # rule that orders turns by message_index, so a "did the user
    # confirm on their LATEST turn" check degrades to "never confirmed"
    # and denies everything it guards.
    edge_data_rows = []
    for ed in graph.get("edges_data", []):
        proximal = ed.get("proximal")
        if proximal is None:
            proximal = -1
        elif isinstance(proximal, bool):
            proximal = 1 if proximal else 0
        message_index = ed.get("message_index")
        if message_index is None:
            message_index = -1
        edge_data_rows.append(
            [
                ed["source"],
                ed["destination"],
                f"[{int(proximal)}, {int(message_index)}]",
            ]
        )
    _write_rfc4180_rows(fact_dir / "EdgeData.facts", edge_data_rows)

    tenant = scenario.get("tenant_id", "")
    tenant_rows = [[tenant]] if tenant else []
    _write_rfc4180_rows(fact_dir / "TenantId.facts", tenant_rows)

    # ActionMetadata(idx, rel, a, b) — per-action facts the caller
    # attested at check time. Writing this empty when the scenario has
    # them makes every rule reading them silently unsatisfiable, which
    # is the same class as the tools_json and EdgeData defects: the
    # rule passes offline against nothing and behaves differently live.
    action_metadata_rows = [
        ["0", str(fact[0]), str(fact[1]), str(fact[2])]
        for fact in scenario.get("action_metadata", [])
        if len(fact) >= 3
    ]
    _write_rfc4180_rows(fact_dir / "ActionMetadata.facts", action_metadata_rows)

    # PolicyMetadata(rel, a, b) — configuration that travels with the
    # policy binding rather than the conversation. The AgentDojo
    # policies derive their sink/source taxonomy from it, so writing it
    # empty makes those rules unsatisfiable offline while they fire
    # live. Fourth relation in this family after EdgeData, tools_json
    # and ActionMetadata; with this one every declared .input relation
    # is populated from the scenario.
    policy_metadata_rows = [
        [str(fact[0]), str(fact[1]), str(fact[2])] for fact in scenario.get("policy_metadata", []) if len(fact) >= 3
    ]
    _write_rfc4180_rows(fact_dir / "PolicyMetadata.facts", policy_metadata_rows)


# ── Policy desugaring ─────────────────────────────────────────


def desugar_policy(
    policy_path: Path,
    sugar_py: Path = SUGAR_PY,
) -> Path:
    """Run sugar.py --resolve-includes and return desugared path.

    Uses a unique temp file per invocation to avoid races when
    multiple harness processes run concurrently.
    """
    fd, tmp = tempfile.mkstemp(suffix=".dl", prefix="policy_test_")
    out_path = Path(tmp)
    try:
        result = desugar(policy_path, sugar_py)
        if result.returncode != 0:
            print(f"ERROR: sugar.py failed:\n{result.stderr or result.stdout}")
            sys.exit(1)
        out_path.write_text(result.stdout)
    finally:
        import os

        os.close(fd)
    return out_path


# ── Functor library ────────────────────────────────────────────


def _lib_suffix() -> str:
    return ".dylib" if platform.system() == "Darwin" else ".so"


def _lib_flag() -> str:
    return "-dynamiclib" if platform.system() == "Darwin" else "-shared"


def discover_companion_functors(policy_path: Path) -> Path | None:
    """Find the policy's companion functor source, if any.

    Mirrors the candidate order of the Rust compiler and
    sasy.policy.api.find_functors: functors.cpp in the same
    directory, then <stem with _policy -> _functors>.cpp, then
    <base>_functors.cpp. Keeping the orders identical is what makes
    an offline harness verdict predict the served one.
    """
    stem = policy_path.stem
    candidates = [
        policy_path.parent / "functors.cpp",
        policy_path.parent / (stem.replace("_policy", "_functors") + ".cpp"),
    ]
    if "_policy" in stem:
        base = stem.split("_policy")[0]
        candidates.append(policy_path.parent / f"{base}_functors.cpp")
    for fp in candidates:
        if fp.exists():
            return fp
    return None


def _cpp_escape(s: str) -> str:
    """Escape a Python string for embedding in a C++ literal."""
    out = []
    for c in s:
        if c == "\\":
            out.append("\\\\")
        elif c == '"':
            out.append('\\"')
        elif c == "\n":
            out.append("\\n")
        elif c == "\r":
            out.append("\\r")
        elif c == "\t":
            out.append("\\t")
        else:
            out.append(c)
    return "".join(out)


def _generate_llm_stub_cpp(stub_json: Path, sha8: str) -> Path:
    """Generate a C++ TU that strongly overrides llm_oracle_query.

    The weak default in functors_common.cpp returns false in
    interpreted mode. The generated override answers from a
    baked-in substring table so offline replay of @llm_check_fn
    policies is deterministic.

    Stub JSON format:
        {"entries": [{"prompt_contains": "...",
                      "context_contains": "...",   # optional
                      "answer": 1}],
         "default": 0}
    First matching entry wins; both *_contains fields are plain
    substring matches (omitted = match anything).
    """
    spec = json.loads(stub_json.read_text())
    entries = spec.get("entries", [])
    default = "true" if spec.get("default", 0) else "false"

    rows = []
    for e in entries:
        pc = e.get("prompt_contains")
        cc = e.get("context_contains")
        p_lit = f'"{_cpp_escape(pc)}"' if pc is not None else "nullptr"
        c_lit = f'"{_cpp_escape(cc)}"' if cc is not None else "nullptr"
        ans = "true" if e.get("answer", 0) else "false"
        rows.append(f"    {{{p_lit}, {c_lit}, {ans}}},")

    body = "\n".join(rows)
    table = (
        "static const LlmStubEntry kEntries[] = {\n"
        f"{body}\n}};\n"
        "static const int kNumEntries = "
        "sizeof(kEntries) / sizeof(kEntries[0]);\n"
        if rows
        else "static const LlmStubEntry* kEntries = nullptr;\nstatic const int kNumEntries = 0;\n"
    )
    code = f"""\
// Auto-generated by policy_test_harness.py from {stub_json.name}.
// Strong override of the weak llm_oracle_query default in
// functors_common.cpp - deterministic @llm_check_fn answers for
// offline replay. Do not edit.
#include <string>

struct LlmStubEntry {{
    const char* prompt_sub;
    const char* context_sub;
    bool answer;
}};

{table}
bool llm_oracle_query(const std::string& prompt,
                      const std::string& context) {{
    for (int i = 0; i < kNumEntries; ++i) {{
        const LlmStubEntry& e = kEntries[i];
        if (e.prompt_sub
            && prompt.find(e.prompt_sub) == std::string::npos)
            continue;
        if (e.context_sub
            && context.find(e.context_sub) == std::string::npos)
            continue;
        return e.answer;
    }}
    return {default};
}}
"""
    out = HARNESS_CACHE_DIR / f"llm_stub_{sha8}.cpp"
    out.write_text(code)
    return out


def _find_souffle_include() -> str | None:
    """Find Souffle header include path."""
    # Try include paths in order. Some homebrew Souffle installs
    # have a nested layout where -I must point at the inner
    # souffle/ dir to avoid duplicate-definition errors.
    candidates = [
        "/opt/homebrew/Cellar/souffle/2.5/include/souffle",
        "/opt/homebrew/Cellar/souffle/2.5/include",
        "/opt/homebrew/include/souffle",
        "/opt/homebrew/include",
        "/usr/local/include",
        "/usr/include",
    ]
    for d in candidates:
        if Path(d).joinpath("souffle", "SouffleInterface.h").exists():
            return d
    # Try souffle --include-dir or pkg-config
    try:
        result = subprocess.run(
            ["pkg-config", "--cflags", "souffle"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            for part in result.stdout.split():
                if part.startswith("-I"):
                    return part[2:]
    except FileNotFoundError:
        pass
    return None


def ensure_functor_lib(
    policy_path: Path | None = None,
) -> Path | None:
    """Build (or reuse) the functor library for a policy.

    Compiles functors_common.cpp together with the policy's
    companion functor file (if any) and an optional @llm_check_fn
    stub (SASY_LLM_STUB_FILE) into a per-policy shared library under
    .harness-cache/, keyed by a hash of all source bytes.

    Passing no policy builds the shared functors alone. That is the
    right default for a policy with no companion file, but it is the
    wrong answer for one that has it: Souffle aborts on the first
    unresolved user-defined operator, so a missing companion reads
    as a crash, not as a quiet fallback.
    """
    if not FUNCTORS_SRC.exists():
        return None

    sources = [FUNCTORS_SRC]
    companion = discover_companion_functors(policy_path) if policy_path else None
    if companion:
        sources.append(companion)

    stub_env = os.environ.get("SASY_LLM_STUB_FILE")
    stub_json = Path(stub_env) if stub_env else None
    if stub_json and not stub_json.exists():
        # Hard failure, not a warning. Dropping the stub links the weak
        # llm_oracle_query default, which answers false to everything,
        # while convergence still sees the env var set and records the
        # mode as `cache` — so it withholds its default-0 warning and
        # certifies an @llm_check_fn policy against fabricated answers.
        raise FileNotFoundError(
            f"SASY_LLM_STUB_FILE is set but missing: {stub_json}. "
            f"Unset it to run without a stub; leaving it dangling "
            f"would evaluate @llm_check_fn as always-false."
        )

    hasher = hashlib.sha256()
    for src in sources:
        hasher.update(src.read_bytes())
    if stub_json:
        hasher.update(b"llm-stub")
        hasher.update(stub_json.read_bytes())
    sha8 = hasher.hexdigest()[:8]

    HARNESS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    lib = HARNESS_CACHE_DIR / f"libfunctors_{sha8}{_lib_suffix()}"
    if lib.exists():
        return lib

    if stub_json:
        sources.append(_generate_llm_stub_cpp(stub_json, sha8))

    names = ", ".join(src.name for src in sources)
    print(f"Building functor library ({names}) at {lib}...")
    # Build to a unique temp path and rename into place. The cache key
    # is a content hash, so two concurrent runs compiling the SAME
    # sources target the same filename — writing both directly would
    # let one load a half-written library. Rename is atomic within a
    # directory, so the loser simply replaces an identical file.
    tmp = lib.with_name(f"{lib.stem}.{os.getpid()}.tmp{lib.suffix}")
    cmd = [
        "g++",
        "-std=c++17",
        "-fPIC",
        "-O2",
        _lib_flag(),
        "-o",
        str(tmp),
        *[str(src) for src in sources],
    ]
    inc = _find_souffle_include()
    if inc:
        cmd.insert(3, f"-I{inc}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        tmp.unlink(missing_ok=True)
        print(f"WARNING: functor build failed:\n{result.stderr}")
        return None
    tmp.replace(lib)
    return lib


def prepare_policy(policy_path: Path) -> tuple[Path, Path | None]:
    """Desugar a policy and build its functor lib.

    Returns (desugared_path, functor_lib). Importable entry point for
    tools that evaluate scenarios programmatically, e.g.
    policy_convergence.py.
    """
    desugared = desugar_policy(policy_path)
    # Scan the desugared file so we catch indirect functor usage via
    # common_policy.dl helpers (e.g. QueriesHost uses @url_host).
    functor_lib: Path | None = None
    if policy_uses_functors(desugared):
        functor_lib = ensure_functor_lib(policy_path)
        if functor_lib is None:
            print("WARNING: Policy uses functors but library could not be built. Functor calls may fail.")
    return desugared, functor_lib


def run_one_scenario(
    desugared: Path,
    scenario: dict[str, Any],
    functor_lib: Path | None = None,
) -> SouffleResults:
    """Evaluate a single scenario against a desugared policy.

    Raises RuntimeError if Souffle fails. raw_output on the result
    carries the full -D- stdout.
    """
    with tempfile.TemporaryDirectory(prefix="souffle_test_") as tmp:
        fact_dir = Path(tmp)
        write_fact_files(scenario, fact_dir)
        stdout = run_souffle(desugared, fact_dir, functor_lib)
        results = parse_souffle_output(stdout)
        results.raw_output = stdout
        return results


def policy_uses_functors(desugared_path: Path) -> bool:
    """Check if the desugared policy uses @ functors in rule bodies.

    Checks the desugared file (which inlines common_policy.dl) so
    we detect indirect functor usage, e.g. QueriesHost using
    @url_host. Ignores .functor declarations.
    """
    text = desugared_path.read_text()
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(".functor"):
            continue
        if re.search(r"@\w+\(", stripped):
            return True
    return False


def policy_uses_llm_check(desugared_path: Path) -> bool:
    """Check if the desugared policy uses @llm_check_fn."""
    text = desugared_path.read_text()
    return "@llm_check_fn" in text


# ── Run Souffle ────────────────────────────────────────────────


def run_souffle(
    desugared_path: Path,
    fact_dir: Path,
    functor_lib: Path | None = None,
) -> str:
    """Invoke souffle in interpreted mode, return stdout."""
    souffle_bin = shutil.which("souffle")
    if not souffle_bin:
        print("ERROR: 'souffle' not found on PATH")
        sys.exit(1)

    cmd = [souffle_bin, str(desugared_path), f"-F{fact_dir}", "-D-"]
    if functor_lib:
        lib_dir = str(functor_lib.parent)
        lib_name = functor_lib.stem.removeprefix("lib")
        cmd += [f"-l{lib_name}", f"-L{lib_dir}"]
        # Also search common lib paths
        for p in ["/usr/local/lib", "/opt/homebrew/lib"]:
            if Path(p).is_dir():
                cmd += [f"-L{p}"]

    result = subprocess.run(
        cmd,
        capture_output=True,
        check=False,
        timeout=30,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Souffle failed (exit {result.returncode}):\n{stderr}")
    return result.stdout.decode("utf-8")


# ── Parse Souffle -D- output ───────────────────────────────────


@dataclass
class SouffleResults:
    """Parsed output from a Souffle -D- run."""

    authorized: set[int] = field(default_factory=set)
    is_authorized: bool = False
    unauthorized: bool = False
    has_principal: bool = False
    apply_transform: dict[int, list[str]] = field(default_factory=dict)
    deny_unauthorized: set[int] = field(default_factory=set)
    allow_passthrough: set[int] = field(default_factory=set)
    # (idx, kind, reason, suggestion) — which deny rule fired, and
    # whether it blocks or asks. Drives the convergence mismatch
    # report, which has to name the rule, not just the verdict.
    denial_reasons: list[tuple[int, str, str, str]] = field(default_factory=list)
    # All controller-requested output relations, including clause-specific
    # helper roots added only to an evaluator copy of the policy.
    relations: dict[str, list[list[str]]] = field(default_factory=dict)
    # Raw -D- stdout, for verbose/debug consumers.
    raw_output: str = ""


def three_valued_verdict(results: SouffleResults, idx: int = 0) -> str:
    """Return the monitor's deny > ask > allow verdict for one action.

    Souffle represents ``ask`` as a DenialReason that intentionally does not
    derive ``Unauthorized``.  Exposing only ``authorized`` therefore collapses
    ASK into ALLOW in offline tests even though the Claude Code host presents an
    approval prompt.  Keep the legacy booleans, but also offer the deployed
    three-valued composition for scenario assertions.
    """
    kinds = {kind for row_idx, kind, _, _ in results.denial_reasons if row_idx == idx}
    if results.unauthorized or "block" in kinds:
        return "DENY"
    if "ask" in kinds:
        return "ASK"
    if idx in results.authorized:
        return "ALLOW"
    return "DENY"


def parse_souffle_output(stdout: str) -> SouffleResults:
    """Parse Souffle -D- output into structured results.

    The output format uses separator lines (===... or ---...)
    to delimit sections, with the relation name on its own line.
    """
    results = SouffleResults()
    current_section = ""

    rows = iter(csv.reader(io.StringIO(stdout, newline="")))
    for row in rows:
        line = row[0].strip() if row else ""
        if not line:
            continue
        if len(row) == 1 and "===============" in line:
            continue
        if len(row) == 1 and "---------------" in line:
            # Next line is section name, line after is column header
            section_row = next(rows, [])
            current_section = section_row[0].strip() if section_row else ""
            next(rows, None)  # skip column header
            continue

        if current_section:
            results.relations.setdefault(current_section, []).append(list(row))

        if current_section == "Authorized":
            try:
                results.authorized.add(int(line))
            except ValueError:
                pass
        elif current_section == "AllowPassthrough":
            try:
                results.allow_passthrough.add(int(line))
            except ValueError:
                pass
        elif current_section == "DenyUnauthorized":
            try:
                results.deny_unauthorized.add(int(line))
            except ValueError:
                pass
        elif current_section == "HasPrincipal":
            if line == "()":
                results.has_principal = True
        elif current_section == "Unauthorized":
            if line and line != "()":
                results.unauthorized = True
        elif current_section == "IsAuthorized":
            if line and line != "()":
                results.is_authorized = True
        elif current_section == "ApplyTransform" and len(row) == 2:
            try:
                idx = int(row[0])
                tid = row[1].replace('\\"', '"')
                results.apply_transform.setdefault(idx, []).append(tid)
            except ValueError:
                pass
        elif current_section == "DenialReason" and len(row) == 4:
            # idx, kind, reason, suggestion. An empty suggestion is a
            # real row ('0,"block","...",""'), so key off the column
            # count rather than truthiness or the row would vanish
            # from the convergence report.
            try:
                results.denial_reasons.append((int(row[0]), row[1], row[2], row[3]))
            except ValueError:
                pass

    return results


# ── Scenario checking ──────────────────────────────────────────


def check_scenario(
    scenario: dict[str, Any],
    results: SouffleResults,
) -> tuple[bool, list[str]]:
    """Check scenario expected values against Souffle results.

    Returns (passed, list_of_failure_messages).
    """
    # `expected: null` is a real value, not a missing key: the labeler
    # writes it for advisory (weak) labels. `.get(k, {})` returns None
    # there, and the membership tests below then raise, which crashed
    # the whole advisory-label workflow before the convergence code
    # that understands it could run.
    expected = scenario.get("expected") or {}
    failures: list[str] = []

    if "verdict" in expected:
        actual_verdict = three_valued_verdict(results)
        expected_verdict = expected["verdict"]
        if expected_verdict not in {"ALLOW", "ASK", "DENY"}:
            failures.append(f"verdict: expected must be ALLOW, ASK, or DENY, got={expected_verdict!r}")
        elif actual_verdict != expected_verdict:
            failures.append(f"verdict: expected={expected_verdict}, actual={actual_verdict}")

    if "authorized" in expected:
        actual = 0 in results.authorized
        if actual != expected["authorized"]:
            failures.append(f"authorized: expected={expected['authorized']}, actual={actual}")

    if "unauthorized" in expected:
        if results.unauthorized != expected["unauthorized"]:
            failures.append(f"unauthorized: expected={expected['unauthorized']}, actual={results.unauthorized}")

    if "is_authorized" in expected:
        if results.is_authorized != expected["is_authorized"]:
            failures.append(f"is_authorized: expected={expected['is_authorized']}, actual={results.is_authorized}")

    if "deny_unauthorized" in expected:
        actual = 0 in results.deny_unauthorized
        if actual != expected["deny_unauthorized"]:
            failures.append(f"deny_unauthorized: expected={expected['deny_unauthorized']}, actual={actual}")

    if "allow_passthrough" in expected:
        actual = 0 in results.allow_passthrough
        if actual != expected["allow_passthrough"]:
            failures.append(f"allow_passthrough: expected={expected['allow_passthrough']}, actual={actual}")

    if "denials" in expected:
        # The exact set of [deny message, suggestion] pairs raised for
        # the action: which rules fired, not only whether one did.
        actual_d = sorted({(reason, suggestion) for row_idx, _, reason, suggestion in results.denial_reasons if row_idx == 0})
        expected_d = sorted({(pair[0], pair[1]) for pair in expected["denials"]})
        if actual_d != expected_d:
            failures.append(f"denials: expected={expected_d}, actual={actual_d}")

    if "transforms" in expected:
        actual_t = results.apply_transform.get(0, [])
        expected_t = expected["transforms"]
        if sorted(actual_t) != sorted(expected_t):
            failures.append(f"transforms: expected={expected_t}, actual={actual_t}")

    return (len(failures) == 0, failures)


# ── Main ───────────────────────────────────────────────────────


def run_harness(
    policy_path: Path,
    scenarios_path: Path,
    verbose: bool = False,
) -> bool:
    """Run all scenarios against a policy. Return True if all pass."""
    with open(scenarios_path) as f:
        data = json.load(f)

    scenarios = data.get("scenarios", [])
    if not scenarios:
        print("WARNING: No scenarios found in test file")
        return True

    desc = data.get("policy_description", scenarios_path.stem)
    print(f"\nPolicy: {policy_path.name}")
    print(f"Description: {desc}")
    print(f"Scenarios: {len(scenarios)}")
    print("-" * 60)

    # Desugar once, and build the functor lib for THIS policy so a
    # companion <name>_functors.cpp is linked in.
    desugared, functor_lib = prepare_policy(policy_path)
    if policy_uses_llm_check(desugared):
        print("WARNING: Policy uses @llm_check_fn which defaults to 0 (deny) without LLM cache.")

    passed = 0
    failed = 0
    errors = 0

    for scenario in scenarios:
        name = scenario.get("name", "unnamed")
        clause = scenario.get("clause", "?")

        try:
            results = run_one_scenario(desugared, scenario, functor_lib)
            if verbose:
                print(f"\n  [DEBUG] Souffle output for '{name}':")
                for line in results.raw_output.splitlines():
                    print(f"    {line}")

            ok, failures = check_scenario(scenario, results)

            if ok:
                print(f"  PASS [{clause}] {name}")
                passed += 1
            else:
                print(f"  FAIL [{clause}] {name}")
                for msg in failures:
                    print(f"       {msg}")
                failed += 1

        except RuntimeError as e:
            print(f"  ERROR [{clause}] {name}")
            print(f"       {e}")
            errors += 1

    # Summary
    print("-" * 60)
    total = passed + failed + errors
    print(f"Results: {passed}/{total} passed", end="")
    if failed:
        print(f", {failed} failed", end="")
    if errors:
        print(f", {errors} errors", end="")
    print()

    return failed == 0 and errors == 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Souffle policies against test scenarios",
    )
    parser.add_argument(
        "--policy",
        required=True,
        type=Path,
        help="Path to the Souffle policy .dl file",
    )
    parser.add_argument(
        "--scenarios",
        required=True,
        type=Path,
        help="Path to test scenarios JSON file",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print Souffle output for each scenario",
    )
    args = parser.parse_args()

    if not args.policy.exists():
        print(f"ERROR: Policy file not found: {args.policy}")
        sys.exit(1)
    if not args.scenarios.exists():
        print(f"ERROR: Scenarios file not found: {args.scenarios}")
        sys.exit(1)

    success = run_harness(args.policy, args.scenarios, args.verbose)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
