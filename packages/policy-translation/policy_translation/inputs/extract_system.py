"""Derive a system profile from captured traces — no source, no LLM.

The translator needs structural facts before it can write a rule:
which agents exist, what each one is called in ``msg.agent``, which
roles they speak with, what tools they call, and what those tools
actually return. The basic translator path gets these by having a
model read the agent's source code. This script derives them from
observed behaviour instead, which is portable to any system
instrumented with the sasy SDK and cannot be wrong about the
deployment in the way a declared schema can.

The distinction that motivates it: ``msg.agent_role = $UserRole()``
does not identify "the end user". Simulators, wrappers, and peer
agents emit user-role turns too, so a content rule keyed on role
alone matches messages it must not. That fact is a crosstab away
from any capture, and this script prints it rather than asking
someone to know it.

An agent NAME is also not an agent IDENTITY. A map-reduce fan-out
gives every worker the same name; so do two unrelated sessions in
one capture file. Replica partitioning is reported under both
available rules (see ``agent_replicas``), together with any
disagreement between them, because which rule is right is a
property of the system under study and not something to assume.

Inputs (either form):
    --capture   capture JSONL written by SASY_CAPTURE_FILE
    --scenarios scenarios JSON from capture_to_scenarios.py

Outputs:
    --out       machine-readable profile (JSON)
    --report    human-readable profile (Markdown)
    --traces    per-agent, per-replica linear traces (Markdown)

Usage:
    uv run python experiments/translation/extract_system.py \\
        --capture outputs/capture/run.jsonl \\
        --out outputs/loop/system.json \\
        --report outputs/loop/system.md \\
        --traces outputs/loop/traces.md
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any

from policy_translation.inputs.capture import (
    SliceGraph,
    agent_replicas,
    merge_graphs,
    render_agent_trace,
    replica_ids,
    slice_graph_from_capture,
    slice_graph_from_scenario,
)

REPLICA_RULES = ("direct", "reach")


# ── Loading ───────────────────────────────────────────────────


def load_capture(path: Path) -> tuple[list[SliceGraph], list[dict]]:
    """Read a capture JSONL. Unparseable lines are skipped — records
    larger than PIPE_BUF can interleave under multi-process writes
    (see capture.py), and one torn line must not lose the run."""
    graphs, records, skipped = [], [], 0
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            records.append(rec)
            graphs.append(slice_graph_from_capture(rec))
    if skipped:
        print(f"WARNING: skipped {skipped} unparseable line(s)", file=sys.stderr)
    return graphs, records


def load_scenarios(path: Path) -> tuple[list[SliceGraph], list[dict]]:
    data = json.loads(path.read_text())
    scenarios = data.get("scenarios", [])
    graphs = [slice_graph_from_scenario(s) for s in scenarios]
    records = [
        {
            "fn_name": (s.get("action") or {}).get("fn_name"),
            "args": (s.get("action") or {}).get("args"),
            "decision": {"authorized": (s.get("expected") or {}).get("authorized")},
            "tool_id": ((s.get("graph") or {}).get("current_nodes") or [None])[0],
        }
        for s in scenarios
    ]
    return graphs, records


# ── Analyses ──────────────────────────────────────────────────


def agent_role_crosstab(graph: SliceGraph) -> list[dict[str, Any]]:
    """(agent, role) counts with example node ids.

    Two names sharing a role — especially ``user`` — is the
    impersonation trap stated as data.
    """
    counts: collections.Counter = collections.Counter()
    examples: dict[tuple[str, str], str] = {}
    for nid, data in graph.nodes():
        key = (data.get("agent") or "", str(data.get("role") or ""))
        counts[key] += 1
        examples.setdefault(key, nid)
    rows = [
        {
            "agent": agent,
            "role": role,
            "count": n,
            "example_node": examples[(agent, role)],
        }
        for (agent, role), n in counts.most_common()
    ]
    return rows


def role_collisions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Roles emitted by more than one agent name."""
    by_role: dict[str, list[str]] = {}
    for r in rows:
        by_role.setdefault(r["role"], []).append(r["agent"])
    return [
        {"role": role, "agents": sorted(set(agents))}
        for role, agents in sorted(by_role.items())
        if len(set(agents)) > 1
    ]


def _json_type(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "object"
    return "null"


def _observed_fields(blob: Any) -> dict[str, str]:
    """Top-level field → observed JSON type, for a dict or a JSON
    string. Anything else yields no fields."""
    if isinstance(blob, str):
        try:
            blob = json.loads(blob)
        except (json.JSONDecodeError, TypeError):
            return {}
    if not isinstance(blob, dict):
        return {}
    return {k: _json_type(v) for k, v in blob.items()}


def _canonical(args: Any) -> str:
    """Key form for an argument blob, so the same call recorded two
    ways compares equal despite key ordering or whitespace."""
    if not isinstance(args, str):
        args = json.dumps(args, sort_keys=True) if args else ""
    try:
        return json.dumps(json.loads(args), sort_keys=True)
    except (json.JSONDecodeError, TypeError):
        return args.strip()


def tool_inventory(graph: SliceGraph, records: list[dict]) -> list[dict[str, Any]]:
    """Tools as actually exercised: call sites, argument keys with
    observed types, and result fields recovered from the tool-result
    nodes' ``derived_from`` provenance.

    Observed beats declared. A declared schema can name a tool the
    deployment does not expose, omit one it does, leave arguments
    untyped, or promise a result shape the implementation cannot
    produce; traces record what actually crossed the boundary.
    """
    calls: collections.Counter = collections.Counter()
    arg_types: dict[str, dict[str, collections.Counter]] = {}
    result_types: dict[str, dict[str, collections.Counter]] = {}
    results_seen: collections.Counter = collections.Counter()

    def note_args(tool: str, args: Any) -> None:
        for field, typ in _observed_fields(args).items():
            arg_types.setdefault(tool, {}).setdefault(field, collections.Counter())[typ] += 1

    # A captured call appears twice: once as the decision record, and
    # once as the tool call attached to the assistant node it hangs
    # off. Counting both doubles every tool, and since results are
    # counted once from result nodes, the inflated numerator also
    # produces false "called N times but no result" warnings.
    # Identity is (node, tool name, canonical args).
    seen: set[tuple[str, str, str]] = set()

    for nid, data in graph.nodes():
        for call in data.get("tools") or []:
            name = call.get("name")
            if not name:
                continue
            raw = call.get("arguments")
            if not isinstance(raw, str):
                raw = json.dumps(raw, sort_keys=True) if raw else ""
            seen.add((nid, name, _canonical(raw)))
            calls[name] += 1
            note_args(name, call.get("arguments"))
        derived = data.get("derived_from")
        if isinstance(derived, dict) and derived.get("name"):
            tool = derived["name"]
            results_seen[tool] += 1
            for field, typ in _observed_fields(data.get("text")).items():
                result_types.setdefault(tool, {}).setdefault(field, collections.Counter())[typ] += 1

    for rec in records:
        fn = rec.get("fn_name")
        if not fn:
            continue
        key = (
            str(rec.get("tool_id") or ""),
            fn,
            _canonical(rec.get("args") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        calls[fn] += 1
        note_args(fn, rec.get("args"))

    def render(spec: dict[str, collections.Counter]) -> dict[str, str]:
        return {field: "|".join(sorted(types)) for field, types in sorted(spec.items())}

    return [
        {
            "tool": tool,
            "calls_observed": n,
            "results_observed": results_seen.get(tool, 0),
            "arg_fields": render(arg_types.get(tool, {})),
            "result_fields": render(result_types.get(tool, {})),
        }
        for tool, n in calls.most_common()
    ]


def replica_analysis(graph: SliceGraph) -> dict[str, Any]:
    """Replica partitions under every rule, plus disagreements."""
    partitions = {rule: agent_replicas(graph, rule) for rule in REPLICA_RULES}
    names = sorted({n for part in partitions.values() for n in part})
    per_agent = []
    for name in names:
        counts = {rule: len(partitions[rule].get(name, [])) for rule in REPLICA_RULES}
        per_agent.append(
            {
                "agent": name,
                "replicas": counts,
                "agrees": len(set(counts.values())) == 1,
                "nodes": sum(len(g) for g in partitions["direct"].get(name, [])),
                "replica_sizes": {rule: [len(g) for g in partitions[rule].get(name, [])] for rule in REPLICA_RULES},
                # Stable handle per replica: the label the renderer
                # prints, plus the earliest node it covers so a
                # reader can find it in the trace.
                "replica_index": {
                    rule: [
                        {
                            "id": (f"{name}#{i}" if len(partitions[rule].get(name, [])) > 1 else name),
                            "anchor_node": g[0],
                            "turns": len(g),
                        }
                        for i, g in enumerate(partitions[rule].get(name, []))
                    ]
                    for rule in REPLICA_RULES
                },
            }
        )
    return {"per_agent": per_agent, "partitions": partitions}


def slice_shapes(graphs: list[SliceGraph], records: list[dict]) -> list[dict[str, Any]]:
    """Per-gated-tool slice statistics: how much context a rule for
    this tool can actually rely on."""
    stats: dict[str, dict[str, list[int]]] = {}
    for g, rec in zip(graphs, records):
        fn = rec.get("fn_name")
        if not fn:
            continue
        nodes = list(g.nodes())
        entry = stats.setdefault(fn, {"nodes": [], "edges": [], "agents": []})
        entry["nodes"].append(len(nodes))
        entry["edges"].append(len(list(g.edges())))
        entry["agents"].append(len({(d.get("agent") or "") for _, d in nodes}))

    def summary(xs: list[int]) -> dict[str, float]:
        return {
            "min": min(xs),
            "max": max(xs),
            "mean": round(sum(xs) / len(xs), 1),
        }

    return [
        {
            "tool": tool,
            "decision_points": len(v["nodes"]),
            "slice_nodes": summary(v["nodes"]),
            "slice_edges": summary(v["edges"]),
            "distinct_agents_in_slice": summary(v["agents"]),
        }
        for tool, v in sorted(stats.items(), key=lambda kv: -len(kv[1]["nodes"]))
    ]


# ── Rendering ─────────────────────────────────────────────────


def render_report(profile: dict[str, Any]) -> str:
    out: list[str] = ["# System profile (derived from traces)", ""]
    out.append(
        f"Decision points: {profile['decision_points']} · "
        f"nodes: {profile['graph']['nodes']} · "
        f"edges: {profile['graph']['edges']}"
    )
    out += ["", "## Agents and roles", ""]
    out.append("| agent | role | messages | example node |")
    out.append("|---|---|---:|---|")
    for r in profile["agent_roles"]:
        out.append(f"| `{r['agent']}` | `{r['role']}` | {r['count']} | `{r['example_node']}` |")

    collisions = profile["role_collisions"]
    out += ["", "## Role collisions", ""]
    if collisions:
        out.append(
            "More than one agent name speaks with these roles. A rule "
            "keyed on the role alone will match all of them, so it "
            "needs an `msg.agent` filter too."
        )
        out.append("")
        for c in collisions:
            agents = ", ".join(f"`{a}`" for a in c["agents"])
            out.append(f"- role `{c['role']}` — emitted by {agents}")
    else:
        out.append("None: every role is emitted by a single agent name.")

    out += ["", "## Replicas", ""]
    out.append(
        "An agent name can cover several independent instances. "
        "`direct` joins same-named nodes that a dependency edge links "
        "directly; `reach` joins them when one is an ancestor of the "
        "other through any path. Where the two disagree, the system "
        "interleaves an agent's turns with foreign nodes and the "
        "choice of rule is load-bearing."
    )
    out += ["", "| agent | nodes | replicas (direct) | replicas (reach) | agree |"]
    out.append("|---|---:|---:|---:|---|")
    for a in profile["replicas"]["per_agent"]:
        out.append(
            f"| `{a['agent']}` | {a['nodes']} | "
            f"{a['replicas']['direct']} | {a['replicas']['reach']} | "
            f"{'yes' if a['agrees'] else '**no**'} |"
        )

    out += ["", "## Tools", ""]
    out.append("| tool | calls | results | argument fields | result fields |")
    out.append("|---|---:|---:|---|---|")
    for t in profile["tools"]:
        args = ", ".join(f"{k}:{v}" for k, v in t["arg_fields"].items())
        res = ", ".join(f"{k}:{v}" for k, v in t["result_fields"].items())
        out.append(
            f"| `{t['tool']}` | {t['calls_observed']} | {t['results_observed']} | {args or '—'} | {res or '—'} |"
        )

    out += ["", "## Slice shapes per gated tool", ""]
    out.append("| tool | decision points | nodes (min/mean/max) | agents in slice (max) |")
    out.append("|---|---:|---|---:|")
    for s in profile["slice_shapes"]:
        n = s["slice_nodes"]
        out.append(
            f"| `{s['tool']}` | {s['decision_points']} | "
            f"{n['min']}/{n['mean']}/{n['max']} | "
            f"{s['distinct_agents_in_slice']['max']} |"
        )

    warnings = profile["warnings"]
    out += ["", "## Warnings", ""]
    out += [f"- {w}" for w in warnings] if warnings else ["None."]
    return "\n".join(out) + "\n"


def render_traces(graph: SliceGraph, partitions: dict[str, Any], rule: str) -> str:
    out = [f"# Per-agent linear traces (replica rule: `{rule}`)", ""]
    # Ambiguity here is judged over the whole merged graph, which is
    # the right scope for a document that shows every replica side
    # by side. Single-instance agents stay unlabelled.
    replica_of = replica_ids(graph, rule)
    for name in sorted(partitions[rule]):
        replicas = partitions[rule][name]
        out.append(f"## `{name}` — {len(replicas)} replica(s)")
        out.append("")
        for i, node_ids in enumerate(replicas):
            label = f"{name}#{i}" if len(replicas) > 1 else name
            out.append(f"### `{label}` — {len(node_ids)} turns, from `{node_ids[0]}`")
            out.append("")
            out.append("```")
            out.append(render_agent_trace(graph, node_ids, replica_of))
            out.append("```")
            out.append("")
    return "\n".join(out)


# ── Main ──────────────────────────────────────────────────────


def build_profile(graphs: list[SliceGraph], records: list[dict]) -> tuple[dict[str, Any], SliceGraph]:
    merged = merge_graphs(graphs)
    rows = agent_role_crosstab(merged)
    collisions = role_collisions(rows)
    replicas = replica_analysis(merged)
    tools = tool_inventory(merged, records)

    warnings: list[str] = []
    for c in collisions:
        agents = ", ".join(c["agents"])
        warnings.append(
            f"role `{c['role']}` is emitted by {len(c['agents'])} "
            f"agent names ({agents}) — rules matching this role need "
            f"an msg.agent filter"
        )
    for a in replicas["per_agent"]:
        if not a["agrees"]:
            warnings.append(
                f"agent `{a['agent']}` splits into "
                f"{a['replicas']['direct']} replicas under `direct` but "
                f"{a['replicas']['reach']} under `reach` — decide which "
                f"rule matches this system before writing per-agent rules"
            )
        elif a["replicas"]["direct"] > 1:
            warnings.append(
                f"agent `{a['agent']}` has {a['replicas']['direct']} "
                f"independent replicas — a rule that treats the name as "
                f"one instance will cross-contaminate them"
            )
    for t in tools:
        if t["results_observed"] == 0:
            warnings.append(
                f"tool `{t['tool']}` was called {t['calls_observed']}x "
                f"but no result node carries its provenance — rules "
                f"joining on ToolResult for it cannot fire"
            )
        elif not t["result_fields"]:
            warnings.append(
                f"tool `{t['tool']}` returns no parseable JSON object — field accessors will not work on its result"
            )

    profile = {
        "decision_points": len(records),
        "graph": {
            "nodes": len(list(merged.nodes())),
            "edges": len(list(merged.edges())),
        },
        "agent_roles": rows,
        "role_collisions": collisions,
        "replicas": {"per_agent": replicas["per_agent"]},
        "tools": tools,
        "slice_shapes": slice_shapes(graphs, records),
        "warnings": warnings,
    }
    return profile, merged


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Derive a system profile from captured traces",
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--capture", type=Path, help="capture JSONL")
    src.add_argument("--scenarios", type=Path, help="scenarios JSON")
    ap.add_argument("--out", type=Path, help="profile JSON")
    ap.add_argument("--report", type=Path, help="profile Markdown")
    ap.add_argument("--traces", type=Path, help="per-agent traces Markdown")
    ap.add_argument(
        "--replica-rule",
        choices=REPLICA_RULES,
        default="direct",
        help="rule used for the --traces grouping (default: direct)",
    )
    args = ap.parse_args()

    path = args.capture or args.scenarios
    if not path.exists():
        print(f"ERROR: not found: {path}", file=sys.stderr)
        sys.exit(1)
    graphs, records = load_capture(path) if args.capture else load_scenarios(path)
    if not graphs:
        print("ERROR: no records found", file=sys.stderr)
        sys.exit(1)

    profile, merged = build_profile(graphs, records)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(profile, indent=2) + "\n")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(render_report(profile))
    if args.traces:
        partitions = {rule: agent_replicas(merged, rule) for rule in REPLICA_RULES}
        args.traces.parent.mkdir(parents=True, exist_ok=True)
        args.traces.write_text(render_traces(merged, partitions, args.replica_rule))

    print(render_report(profile))
    if not (args.out or args.report or args.traces):
        print(
            "(no --out/--report/--traces given; printed report only)",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
