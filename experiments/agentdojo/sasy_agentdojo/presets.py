"""The eight configurations of the paper's AgentDojo results table, by name.

The table has four defense rows, each run with AgentDojo's
`repeat_user_prompt` defense off and on:

    preset            tag          arm       defense / policy switches
    none              N-none       baseline  no defense
    none-rep          N-none-rep   baseline  repeat_user_prompt
    tool-filter       N-tf         baseline  tool_filter
    tool-filter-rep   N-tf-rep     baseline  tool_filter + repeat_user_prompt
    ifc               P-fides      sasy      strict_taint
    ifc-rep           P-fides-rep  sasy      strict_taint + repeat_user_prompt
    provenance        C-w2-norep   sasy      PROVENANCE_SWITCHES
    provenance-rep    C-w2         sasy      PROVENANCE_SWITCHES + repeat_user_prompt

Every cell shares `TABLE_SETTINGS`: all four suites, the Azure deployment
`gpt-4o` with the pipeline named after `gpt-4o-2024-08-06`, the
`tool_knowledge` attack, benchmark version v1.2.2, five sweeps.

The tag is what `analysis/make_table.py` matches on (`ROWS`), so a preset's
results land in run directories the table generator reads without any
renaming. This module imports nothing from AgentDojo, so the CLI can list the
presets and the tests can check them without the benchmark installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Policy switches the runner can turn on. Each maps one-to-one to a keyword of
#: `policy_metadata.build` (`no_target_exemption` is the negation of its
#: `target_exemption`), and to a `--kebab-case` CLI flag.
POLICY_SWITCHES = (
    "fetch_gate",
    "strict_taint",
    "confidentiality",
    "gate_unanchorable",
    "account_provenance",
    "browse_exception",
    "selection_check",
    "strict_consent",
    "no_target_exemption",
    "dictated_content",
)

#: The switches of the provenance policy rows. The intent rule, destination
#: anchoring, body-identifier anchoring, new-credential provenance and the
#: response check on business names are always on and need no switch.
PROVENANCE_SWITCHES = frozenset({
    "fetch_gate",
    "confidentiality",
    "gate_unanchorable",
    "browse_exception",
    "selection_check",
    "strict_consent",
    "dictated_content",
})

#: Run settings shared by every table cell.
TABLE_SETTINGS = {
    "provider": "azure",
    "deployment": "gpt-4o",
    "model_label": "gpt-4o-2024-08-06",
    "attack": "tool_knowledge",
    "version": "v1.2.2",
    "repeats": 5,
    "threads": 96,
}


@dataclass(frozen=True)
class Preset:
    name: str
    tag: str
    row: str
    arm: str
    defense: str | None = None
    switches: frozenset[str] = field(default_factory=frozenset)
    repeat_user_prompt: bool = False

    def flags(self) -> list[str]:
        """The CLI flags that select this configuration by hand."""
        out = ["--arm", self.arm, "--tag", self.tag]
        if self.defense:
            out += ["--defense", self.defense]
        out += ["--" + s.replace("_", "-") for s in POLICY_SWITCHES if s in self.switches]
        if self.repeat_user_prompt:
            out.append("--repeat-user-prompt")
        return out


_ROW_NONE = "No runtime defense"
_ROW_TF = "Tool-call filter"
_ROW_IFC = "Information flow control policy"
_ROW_PROV = "Provenance policy (ours)"

PRESETS: dict[str, Preset] = {
    p.name: p
    for p in (
        Preset("none", "N-none", _ROW_NONE, "baseline"),
        Preset("none-rep", "N-none-rep", _ROW_NONE, "baseline",
               defense="repeat_user_prompt"),
        Preset("tool-filter", "N-tf", _ROW_TF, "baseline", defense="tool_filter"),
        Preset("tool-filter-rep", "N-tf-rep", _ROW_TF, "baseline",
               defense="tool_filter+repeat_user_prompt"),
        Preset("ifc", "P-fides", _ROW_IFC, "sasy",
               switches=frozenset({"strict_taint"})),
        Preset("ifc-rep", "P-fides-rep", _ROW_IFC, "sasy",
               switches=frozenset({"strict_taint"}), repeat_user_prompt=True),
        Preset("provenance", "C-w2-norep", _ROW_PROV, "sasy",
               switches=PROVENANCE_SWITCHES),
        Preset("provenance-rep", "C-w2", _ROW_PROV, "sasy",
               switches=PROVENANCE_SWITCHES, repeat_user_prompt=True),
    )
}

#: Names that expand to several presets.
GROUPS: dict[str, tuple[str, ...]] = {"table": tuple(PRESETS)}


def resolve(names: list[str]) -> list[Preset]:
    """Presets for `names`, in order and without duplicates.

    A name is a preset name, a group name (`table`), or a preset's tag
    (`C-w2`), so a cell can be asked for by the label the table uses.
    """
    by_tag = {p.tag: p for p in PRESETS.values()}
    out: list[Preset] = []
    for name in names:
        if name in GROUPS:
            picked = [PRESETS[n] for n in GROUPS[name]]
        elif name in PRESETS:
            picked = [PRESETS[name]]
        elif name in by_tag:
            picked = [by_tag[name]]
        else:
            known = ", ".join([*GROUPS, *PRESETS])
            raise ValueError(f"unknown preset {name!r} (known: {known}, or a table tag)")
        out += [p for p in picked if p not in out]
    return out
