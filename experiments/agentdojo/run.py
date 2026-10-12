"""Run AgentDojo live, with or without the SASY reference monitor in the tool path.

    # one cell of the results table (five sweeps, all four suites)
    python run.py --preset provenance-rep

    # every cell of the results table
    python run.py --preset table

    # list the presets and the flags each one stands for
    python run.py --list-presets

    # an arm selected by hand
    python run.py --arm sasy --tag my-arm --strict-taint --repeats 1

Both arms go through AgentDojo's own scoring, so `utility` and `security` mean
what they mean on the published leaderboard. The policy arm (`--arm sasy`) is
AgentDojo's stock pipeline with `SasyToolsExecutor` in place of the stock
`ToolsExecutor`; the comparison arm (`--arm baseline`) is the stock pipeline,
optionally behind one of AgentDojo's own defenses (`--defense`).

Everything runs in one process with one thread per run (`--threads`). Results
are one JSON file per run under `--logdir` (default `outputs/agentdojo/runs` at
the repository root), keyed by pipeline name; a run whose file exists is
skipped unless `--force-rerun` is given, so an interrupted sweep resumes.
`--shard K/N` splits the job list across N processes or machines.

The policy arm needs a running SASY binary (`make serve`) and `SASY_API_KEY`;
the model needs provider credentials in the environment or `--env-file`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from sasy_agentdojo.presets import (  # noqa: E402
    GROUPS,
    POLICY_SWITCHES,
    PRESETS,
    TABLE_SETTINGS,
    resolve,
)

REPO_ROOT = HERE.parents[1]
SUITES = ("banking", "slack", "travel", "workspace")

#: Flags that define an arm. A preset already fixes all of them, so giving one
#: beside --preset would run something other than the cell the tag names.
ARM_FLAGS = ("arm", "tag", "defense", "repeat_user_prompt", "check_response",
             "vouch_channels", *POLICY_SWITCHES)

SWITCH_HELP = {
    "fetch_gate": "treat URL fetches as sinks whose destination is the URL",
    "strict_taint": "trusted-action (information-flow) mode: no sink once anything "
                    "untrusted is in context; implies the fetch gate and --confidentiality",
    "confidentiality": "once the user's own data is in context, a disclosing "
                       "destination must be anchored",
    "gate_unanchorable": "refuse a tainted sink that names no destination at all",
    "account_provenance": "an account-number destination must come from the user's "
                          "request or the operator's records",
    "browse_exception": "allow a user-requested fetch of a URL that cannot carry data out",
    "selection_check": "where the request names a criterion the listing settles, the "
                       "target must be the one it selects",
    "strict_consent": "the target of a classified effect must be one the user identified",
    "no_target_exemption": "let the target rule govern every classified kind, including "
                           "those with no value the policy can call the target",
    "dictated_content": "refuse a call whose text was copied verbatim out of an "
                        "outsider's text",
}


def parse_shard(value: str) -> tuple[int, int]:
    try:
        k, n = (int(x) for x in value.split("/"))
    except ValueError:
        raise argparse.ArgumentTypeError("expected K/N, e.g. 0/4") from None
    if n < 1 or not 0 <= k < n:
        raise argparse.ArgumentTypeError("need 0 <= K < N")
    return k, n


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Run the AgentDojo benchmark live, with or without SASY.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="presets: " + ", ".join([*GROUPS, *PRESETS]) + " (see --list-presets)",
    )
    sel = ap.add_argument_group("what to run")
    sel.add_argument("--preset", action="append", default=[], metavar="NAME",
                     help="a results-table cell by name or tag, or 'table' for all "
                          "eight; repeatable")
    sel.add_argument("--list-presets", action="store_true",
                     help="print the presets and their flags, then exit")
    sel.add_argument("--arm", action="append", default=None, choices=("sasy", "baseline"),
                     help="arm to run by hand; repeatable (default: sasy)")
    sel.add_argument("--tag", default=None,
                     help="label in the pipeline name, so arms land in separate directories")
    sel.add_argument("--defense", default=None,
                     help="run the baseline arm behind an AgentDojo defense: "
                          "repeat_user_prompt, spotlighting_with_delimiting, tool_filter, "
                          "tool_filter+repeat_user_prompt, "
                          "spotlighting_with_delimiting+repeat_user_prompt")
    sel.add_argument("--repeat-user-prompt", action="store_true", default=None,
                     help="compose AgentDojo's repeat_user_prompt defense with the policy arm")
    sel.add_argument("--check-response", action="store_true", default=None,
                     help="also check the agent's final answer")
    sel.add_argument("--vouch-channels", action="store_true", default=None,
                     help="treat the Slack channel registry as operator-controlled")
    pol = ap.add_argument_group("policy switches (policy arm only)")
    for s in POLICY_SWITCHES:
        pol.add_argument("--" + s.replace("_", "-"), dest=s, action="store_true",
                         default=None, help=SWITCH_HELP[s])

    run = ap.add_argument_group("run settings")
    run.add_argument("--provider", default=TABLE_SETTINGS["provider"],
                     choices=("azure", "openai"))
    run.add_argument("--deployment", default=TABLE_SETTINGS["deployment"],
                     help="Azure deployment or OpenAI model")
    run.add_argument("--model-label", default=None,
                     help="AgentDojo model id the pipeline is named after. The attacks "
                          "address the victim model by the name they resolve from the "
                          "pipeline name, so this must be an id AgentDojo knows. "
                          f"Default: {TABLE_SETTINGS['model_label']} for the "
                          f"{TABLE_SETTINGS['deployment']} deployment; required with "
                          "any other --deployment, so two models never share result "
                          "directories")
    run.add_argument("--env-file", type=Path, default=REPO_ROOT / ".env-credentials",
                     help="KEY=value file read for provider credentials")
    run.add_argument("--attack", default=TABLE_SETTINGS["attack"],
                     help="AgentDojo attack; 'none' runs only the clean user tasks")
    run.add_argument("--version", default=TABLE_SETTINGS["version"],
                     help="AgentDojo benchmark version")
    run.add_argument("--suite", action="append", default=[], choices=SUITES,
                     help="restrict to a suite; repeatable (default: all four)")
    run.add_argument("--user-task", action="append", default=[],
                     help="restrict to a user task id; repeatable")
    run.add_argument("--injection-task", action="append", default=[],
                     help="restrict to an injection task id; repeatable")
    run.add_argument("--repeats", type=int, default=None,
                     help="run each arm N times, tagging sweeps <tag>-r<i> (also "
                          "<tag>-r1 for a single sweep) "
                          f"(default: {TABLE_SETTINGS['repeats']} with --preset, else 1)")
    run.add_argument("--threads", type=int, default=TABLE_SETTINGS["threads"])
    run.add_argument("--shard", type=parse_shard, default=None, metavar="K/N",
                     help="run only every N-th job starting at K")
    run.add_argument("--logdir", type=Path, default=None,
                     help="result directory (default: outputs/agentdojo/runs)")
    run.add_argument("--decisions", type=Path, default=None,
                     help="append every monitor decision to this JSONL file")
    run.add_argument("--force-rerun", action="store_true", help="ignore existing results")
    run.add_argument("--fail-open", action="store_true",
                     help="allow tool calls when the monitor is unreachable")
    run.add_argument("--dry-run", action="store_true",
                     help="print the arms and job counts without running anything")
    return ap


def list_presets() -> None:
    settings = " ".join(
        f"--{k.replace('_', '-')} {v}" for k, v in TABLE_SETTINGS.items()
    )
    print(f"shared settings: {settings}\n")
    for p in PRESETS.values():
        print(f"{p.name:<16} {p.row}")
        print(f"{'':<16} {' '.join(p.flags())}")
    for name, members in GROUPS.items():
        print(f"\n{name}: {', '.join(members)}")


def configs_from_args(ap: argparse.ArgumentParser, args) -> list:
    from sasy_agentdojo.runner import ArmConfig  # noqa: PLC0415

    if args.preset:
        given = [f"--{f.replace('_', '-')}" for f in ARM_FLAGS if getattr(args, f) is not None]
        if given:
            ap.error(f"{', '.join(given)} cannot be combined with --preset; a preset "
                     f"fixes the arm (see --list-presets)")
        try:
            presets = resolve(args.preset)
        except ValueError as exc:
            ap.error(str(exc))
        return [
            ArmConfig(arm=p.arm, tag=p.tag, defense=p.defense, switches=p.switches,
                      repeat_user_prompt=p.repeat_user_prompt)
            for p in presets
        ]

    arms = args.arm or ["sasy"]
    switches = frozenset(s for s in POLICY_SWITCHES if getattr(args, s))
    if args.defense and "sasy" in arms:
        ap.error("--defense applies to the baseline arm; use --repeat-user-prompt "
                 "to compose the repeat defense with the policy arm")
    if "sasy" not in arms and (switches or args.repeat_user_prompt or args.check_response
                               or args.vouch_channels):
        ap.error("policy switches apply to --arm sasy only")
    return [
        ArmConfig(arm=arm, tag=args.tag, defense=args.defense if arm == "baseline" else None,
                  switches=switches if arm == "sasy" else frozenset(),
                  repeat_user_prompt=bool(args.repeat_user_prompt) and arm == "sasy",
                  check_response=bool(args.check_response) and arm == "sasy",
                  vouch_channels=bool(args.vouch_channels) and arm == "sasy")
        for arm in arms
    ]


def main() -> None:
    ap = build_parser()
    args = ap.parse_args()
    if args.list_presets:
        list_presets()
        return

    model_label = args.model_label
    if model_label is None:
        # The label names the result directories, and a run whose file exists
        # is skipped. Defaulting the table's label for another deployment would
        # file that model's runs under the table model's name and skip tasks
        # on the strength of the other model's results.
        if args.deployment != TABLE_SETTINGS["deployment"]:
            ap.error(f"--deployment {args.deployment} is not the table's "
                     f"{TABLE_SETTINGS['deployment']}; give --model-label with the "
                     f"AgentDojo model id of that model")
        model_label = TABLE_SETTINGS["model_label"]

    configs = configs_from_args(ap, args)

    from sasy_agentdojo import runner  # noqa: PLC0415 — imports AgentDojo

    settings = runner.RunSettings(
        provider=args.provider,
        deployment=args.deployment,
        model_label=model_label,
        env_file=args.env_file,
        attack=None if args.attack in (None, "", "none") else args.attack,
        version=args.version,
        logdir=args.logdir or runner.DEFAULT_LOGDIR,
        threads=args.threads,
        force_rerun=args.force_rerun,
        fail_open=args.fail_open,
        decisions=args.decisions,
    )
    repeats = args.repeats or (TABLE_SETTINGS["repeats"] if args.preset else 1)

    if args.dry_run:
        jobs = runner.build_jobs(configs, settings, args.suite, args.user_task,
                                 args.injection_task, args.shard)
        print(f"logdir  : {settings.logdir}")
        print(f"repeats : {repeats}")
        for cfg in configs:
            n = sum(1 for j in jobs if j.config == cfg)
            sweep = "r1" if repeats == 1 else "r<i>"
            tag = f"{cfg.tag}-{sweep}" if cfg.tag else sweep
            name = runner.ArmConfig(cfg.arm, tag, cfg.defense).pipeline_name(settings.label)
            print(f"{name:<60} {n} runs per sweep  "
                  f"switches={','.join(sorted(cfg.switches)) or '-'}"
                  f"{'  repeat_user_prompt' if cfg.repeat_user_prompt else ''}")
        print(f"{len(jobs)} runs per sweep, {len(jobs) * repeats} in total")
        return

    failed = runner.run(configs, settings, repeats=repeats, suites=args.suite,
                        user_tasks=args.user_task, injection_tasks=args.injection_task,
                        shard=args.shard)
    # A sweep that lost runs is INCOMPLETE: failures are arm-specific, so the
    # arm that lost its hardest runs would look better and nothing in the
    # results would say so. Every job is still attempted, but the exit status
    # reports the loss.
    if failed:
        print(f"\n{failed} run(s) failed; the results are INCOMPLETE and not "
              f"comparable with a complete sweep", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
