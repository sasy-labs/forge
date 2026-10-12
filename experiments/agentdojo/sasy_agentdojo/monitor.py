"""The reference monitor, as an AgentDojo pipeline element.

AgentDojo runs tools through `ToolsExecutor`, which walks the tool calls on the
last assistant message and appends one `ChatToolResultMessage` per call. The
monitor is a drop-in replacement for that element: before each call runs, it
records the conversation into the graph, asks the SASY reference monitor
whether the call is authorized, and executes it only if the answer is yes.

Replacing the executor rather than sitting beside it matters. A separate
element placed after `ToolsExecutor` could only complain about a call that had
already run; one placed before it cannot suppress a single call without
breaking the assistant-message-then-results ordering that every provider's
API expects, and without losing the `tool_call_id` pairing. Wrapping the
executor keeps the transcript well formed and puts the decision genuinely
before the effect.

A denied call comes back to the model as a tool result carrying the policy's
reason and its suggested fix, which is the same shape the agent would see from
any other tool failure. The model is free to try something else; what it
cannot do is have the effect.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Sequence
from typing import Any

from agentdojo.agent_pipeline.llms.google_llm import EMPTY_FUNCTION_NAME
from agentdojo.agent_pipeline.tool_execution import (
    ToolsExecutor,
    is_string_list,
    tool_result_to_str,
)
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionReturnType, FunctionsRuntime
from ast import literal_eval
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.types import (
    ChatAssistantMessage,
    ChatMessage,
    ChatToolResultMessage,
    text_content_block_from_string,
)

from sasy.reference_monitor import check_tool_call

from .graph import (
    MALFORMED_EMPTY,
    MALFORMED_UNKNOWN_PREFIX,
    REFUSAL_PREFIX,
    ConversationRecorder,
    did_not_run,
    message_text,
)
from .taxonomy import TAXONOMIES

logger = logging.getLogger(__name__)


#: Marks a call the policy asked about and that ran anyway, because this
#: harness has no approval channel. Scored separately — never a refusal.
#:
#: It goes in the tool result's CONTENT, not its `error`. AgentDojo renders a
#: tool message as `content = error or content_blocks`, so a marker in `error`
#: replaces the tool's return value with itself: the call runs, and the model is
#: shown the marker instead of the id or the listing it asked for. That breaks
#: any task needing a returned value, for a reason nothing to do with the
#: policy, and it silently understated the graded arm's utility.
ASK_MARKER = "[Approved by the user after a policy question.]"

#: The first line of every refusal. A tool result is recognised as a refusal by
#: this PREFIX, never by equality with it, because the reason and the suggestion
#: follow it in the same string — see `refusal_text` below for why they must.
DENIAL_TEMPLATE = (
    REFUSAL_PREFIX + "\n"
    "Reason: {reason}\n"
    "{suggestion}"
)


def is_refusal(message) -> bool:
    """Did the policy refuse the call this tool result belongs to?

    Narrower than `did_not_run`, which also covers a call the executor never
    dispatched. Refusal is a POLICY verdict; the others are the model naming a
    tool that does not exist.
    """
    return (message.get("error") or "").startswith(REFUSAL_PREFIX)


class PolicyDecision:
    """A verdict, which has three shapes and not two.

    `asked` marks the soft outcome the engine signals with an `ASK`-kind denial
    reason (the policy's `@ask` annotation): the policy will not decide alone
    and wants the person to. A host with an approval channel prompts; here there
    is nobody to prompt, so an asked call PROCEEDS and is counted.

    Counting it is the point. Under attack a human would stop it, so security is
    scored with asked calls treated as blocked; utility is scored with them
    treated as approved, which is the assumption the published human-in-the-loop
    work makes for its own numbers. The rate itself is the third axis — the
    human load a deny-only policy has no way to trade against.
    """

    __slots__ = ("authorized", "reasons", "suggestions", "asked")

    def __init__(self, authorized: bool, reasons: list[str], suggestions: list[str],
                 asked: bool = False):
        self.authorized = authorized
        self.reasons = reasons
        self.suggestions = suggestions
        self.asked = asked

    def as_tool_output(self) -> str:
        reason = "; ".join(self.reasons) or "not permitted"
        suggestion = ""
        if self.suggestions:
            suggestion = "Suggestion: " + " ".join(self.suggestions)
        return DENIAL_TEMPLATE.format(reason=reason, suggestion=suggestion).strip()


class SasyToolsExecutor(ToolsExecutor):
    """`ToolsExecutor` with a policy check in front of every call.

    Args:
        suite: which AgentDojo suite is running, used to resolve the
            operator's directory for the destination lookup.
        directory: normalised identifiers the operator vouches for. The monitor
            attaches the matching ones to the check as `ActionMetadata` facts,
            so the policy learns "this destination is one of ours" from the
            operator's own records rather than from the agent's context.
        fail_closed: what to do when the monitor cannot be reached. Defaults to
            refusing, since a reference monitor that fails open is not one.
    """

    def __init__(
        self,
        suite: str,
        recorder: ConversationRecorder,
        directory: frozenset[str] = frozenset(),
        tool_output_formatter: Callable[[FunctionReturnType], str] = tool_result_to_str,
        fail_closed: bool = True,
        on_decision: Callable[[str, dict[str, Any], PolicyDecision], None] | None = None,
    ) -> None:
        super().__init__(tool_output_formatter)
        self.suite = suite
        self.recorder = recorder
        self.directory = directory
        self.fail_closed = fail_closed
        self.on_decision = on_decision

    # ── directory lookup ─────────────────────────────────────────────

    def _known_destinations(self, function: str, args: dict[str, Any]) -> list[tuple[str, str, str]]:
        """Resolve the call's destination arguments against the operator's
        directory and emit one `known_destination` fact per hit.

        Only values the call actually carries are looked up, so the fact set
        stays a handful of tuples rather than the whole directory.
        """
        taxonomy = TAXONOMIES.get(self.suite)
        if taxonomy is None:
            return []
        fields = taxonomy.destination_fields.get(function, ())
        facts: list[tuple[str, str, str]] = []
        for field in fields:
            value = args.get(field)
            values = value if isinstance(value, (list, tuple, set)) else [value]
            for item in values:
                if item is None:
                    continue
                folded = "".join(c for c in str(item).lower() if c.isalnum())
                if folded and folded in self.directory:
                    facts.append(("known_destination", folded, "1"))
        return facts

    # ── the check ────────────────────────────────────────────────────

    def check(
        self,
        function: str,
        args: dict[str, Any],
        input_node_id: str | None,
        extra_metadata: list[tuple[str, str, str]] | None = None,
    ) -> PolicyDecision:
        try:
            result = check_tool_call(
                function,
                json.dumps(dict(args), default=str),
                input_node_ids=[input_node_id] if input_node_id else [],
                metadata=self._known_destinations(function, args) + (extra_metadata or []),
            )
        except Exception as exc:  # transport failure, not a policy answer
            logger.warning("reference monitor unreachable for %s: %s", function, exc)
            if self.fail_closed:
                return PolicyDecision(
                    False,
                    ["the security policy could not be consulted"],
                    ["Retry once the reference monitor is reachable."],
                )
            return PolicyDecision(True, [], [])
        # `ASK` = 5 in DenialReasonType: a soft denial the policy wants a person
        # to settle. Every reason must be an ask for the call to proceed — a
        # verdict carrying one hard refusal is a refusal.
        reasons_proto = list(result.denial_trace.reasons)
        asked = bool(reasons_proto) and all(r.reason_type == 5 for r in reasons_proto)
        if asked:
            return PolicyDecision(True, list(result.denial_reasons),
                                  list(result.suggestions), asked=True)
        return PolicyDecision(
            bool(result.authorized), list(result.denial_reasons), list(result.suggestions)
        )

    # ── pipeline element ─────────────────────────────────────────────

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        if len(messages) == 0 or messages[-1]["role"] != "assistant":
            return query, runtime, env, messages, extra_args
        pending = messages[-1].get("tool_calls")
        if not pending:
            return query, runtime, env, messages, extra_args

        # Everything the model could have read before choosing these calls has
        # to be in the graph before the first check.
        input_node_id = self.recorder.sync(messages)

        results: list[ChatToolResultMessage] = []
        for tool_call in pending:
            # Re-sync so this check sees what the EARLIER calls in this same
            # batch did. The model emits several calls in one assistant message,
            # and syncing only once per batch leaves every call after the first
            # reasoning about the state before any of them ran — so a rule about
            # what has already happened is blind to a repeat inside one batch.
            # `sync` is incremental and returns the last node, so this is a
            # no-op on the first iteration.
            if results:
                input_node_id = self.recorder.sync([*messages, *results])
            # Malformed calls are rejected the way the stock executor rejects
            # them, before the policy is consulted: there is no effect to
            # authorize, and a monitor that answered these would be answering
            # about a tool that does not exist.
            malformed = None
            if tool_call.function == EMPTY_FUNCTION_NAME:
                malformed = MALFORMED_EMPTY
            elif tool_call.function not in (t.name for t in runtime.functions.values()):
                malformed = f"{MALFORMED_UNKNOWN_PREFIX}{tool_call.function} provided."
            if malformed is not None:
                results.append(
                    ChatToolResultMessage(
                        role="tool",
                        content=[text_content_block_from_string("")],
                        tool_call_id=tool_call.id,
                        tool_call=tool_call,
                        error=malformed,
                    )
                )
                continue

            # Same list coercion the stock executor applies, so the policy sees
            # the arguments the tool will actually receive.
            for name, value in tool_call.args.items():
                if isinstance(value, str) and is_string_list(value):
                    tool_call.args[name] = literal_eval(value)

            decision = self.check(tool_call.function, dict(tool_call.args), input_node_id)
            if self.on_decision is not None:
                self.on_decision(tool_call.function, dict(tool_call.args), decision)

            if not decision.authorized:
                output = decision.as_tool_output()
                # The refusal is NOT recorded separately. The tool result below
                # carries the same text, the model sees exactly that one
                # message, and the re-sync before the next check records it — so
                # a synthetic node in addition would put two refusals in the
                # graph for one the conversation actually contained, inflating
                # message counts and provenance paths, and making a captured
                # scenario describe a conversation that never happened.
                #
                # `record_denial` is still used for a refused ANSWER, where
                # there is no tool message to carry it.
                results.append(
                    ChatToolResultMessage(
                        role="tool",
                        content=[text_content_block_from_string(output)],
                        tool_call_id=tool_call.id,
                        tool_call=tool_call,
                        # The DETAIL goes in `error`, not a bare marker.
                        # AgentDojo renders a tool message as
                        # `content = error or content`, so a short marker here
                        # replaces the reason and the suggestion with itself and
                        # the model is told only that something was refused —
                        # never which rule, and never what to do instead. The
                        # policy's whole remediation vocabulary was invisible.
                        error=output,
                    )
                )
                continue

            value, error = runtime.run_function(env, tool_call.function, tool_call.args)
            results.append(
                ChatToolResultMessage(
                    role="tool",
                    content=[text_content_block_from_string(self.output_formatter(value))],
                    tool_call_id=tool_call.id,
                    tool_call=tool_call,
                    error=error,
                )
            )
            # An asked call ran, because there is no one here to ask. The marker
            # rides alongside the real output so the run can be scored both ways
            # afterwards — blocked for security, approved for utility — without
            # taking the output away from the model.
            if decision.asked and not error:
                results[-1]["content"] = [
                    *(results[-1].get("content") or []),
                    text_content_block_from_string(ASK_MARKER),
                ]

        messages = [*messages, *results]
        # Register the results too, so the next call's backward slice contains
        # the content the model is about to read.
        self.recorder.sync(messages)
        return query, runtime, env, messages, extra_args


class ResponseMonitor(BasePipelineElement):
    """Checks the agent's own answer, after the tools have finished.

    A prompt injection does not have to reach a tool to succeed. It can simply
    get the agent to recommend a business, or repeat a claim, in the reply it
    gives the user — and every rule that gates tool calls is blind to that by
    construction. On the travel suite roughly a quarter of everything that
    survives the tool-level policy is of this kind.

    The answer is submitted to the same reference monitor as a call to the
    `respond` tool, so it travels the same path, lands in the same graph, and is
    judged by rules written in the same shape.

    A refusal is answered with redaction, not a blank. Replacing the whole reply
    is a bad trade: an injection that plants one sentence would cost the user the
    other nine they asked for, turning a security win into a utility loss. So the
    element re-submits the reply a sentence at a time and keeps the sentences the
    policy accepts, then verifies the result as a whole before returning it. The
    policy stays the authority on what may be said; the element only decides how
    little to remove.
    """

    RESPONSE_TOOL = "respond"

    #: Sentence boundaries, plus list items and headings, which agent answers
    #: are full of and which `.` alone does not separate.
    _SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")

    #: Beyond this an answer is not worth taking apart sentence by sentence;
    #: the whole thing is withheld instead.
    MAX_PARTS = 60

    def __init__(
        self,
        suite: str,
        recorder: ConversationRecorder,
        directory: frozenset[str] = frozenset(),
        fail_closed: bool = True,
        on_decision: Callable[[str, dict[str, Any], PolicyDecision], None] | None = None,
    ) -> None:
        self.checker = SasyToolsExecutor(
            suite=suite,
            recorder=recorder,
            directory=directory,
            fail_closed=fail_closed,
            on_decision=on_decision,
        )

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        if not messages or messages[-1]["role"] != "assistant":
            return query, runtime, env, messages, extra_args
        text = message_text(messages[-1])
        if not text.strip():
            return query, runtime, env, messages, extra_args

        node = self.checker.recorder.sync(messages)
        decision = self.checker.check(
            self.RESPONSE_TOOL, {"text": text}, node, self._vouched_for(text)
        )
        # The decision stream promises EVERY verdict, and the final answer is a
        # verdict like any other. Reporting only the tool calls left response
        # denials and ASK outcomes out of it, which understates both the
        # intervention count and the human load on response-checked runs.
        if self.checker.on_decision is not None:
            self.checker.on_decision(self.RESPONSE_TOOL, {"text": text}, decision)
        if decision.authorized:
            return query, runtime, env, messages, extra_args

        replacement = self._redact(text, node)
        self.checker.recorder.record_denial(
            "response redacted: " + ("; ".join(decision.reasons) or "refused"), node
        )
        messages = [
            *messages[:-1],
            ChatAssistantMessage(
                role="assistant",
                content=[text_content_block_from_string(replacement)],
                tool_calls=[],
            ),
        ]
        return query, runtime, env, messages, extra_args

    def _redact(self, text: str, node: str | None) -> str:
        """Keep the parts of the answer the policy accepts.

        Each part is checked on its own, which is sound for a rule that refuses
        an answer because of something it *contains*: a part that is refused
        alone would be refused in any answer containing it. The reassembled
        answer is checked once more, and if it still fails — a rule about the
        answer as a whole rather than its parts — nothing is returned.
        """
        parts = [p for p in self._SPLIT.split(text) if p.strip()]
        if not parts or len(parts) > self.MAX_PARTS:
            return self.WITHHELD

        kept = [
            p for p in parts
            if self.checker.check(
                self.RESPONSE_TOOL, {"text": p}, node, self._vouched_for(p)
            ).authorized
        ]
        if not kept:
            return self.WITHHELD

        rebuilt = " ".join(kept)
        if not self.checker.check(
            self.RESPONSE_TOOL, {"text": rebuilt}, node, self._vouched_for(rebuilt)
        ).authorized:
            return self.WITHHELD
        return rebuilt

    def _vouched_for(self, text: str) -> list[tuple[str, str, str]]:
        """Directory entries this text mentions, as `known_destination` facts.

        The argument path resolves the directory by looking up named
        destination *fields*, which a free-text answer does not have — so
        without this the response rule is handed no catalogue at all and
        refuses every business, including the ones the operator lists. Scanning
        the text for entries the operator already knows is the same lookup, run
        the only way prose allows.
        """
        folded = "".join(c for c in text.lower() if c.isalnum())
        return [
            ("known_destination", entry, "1")
            for entry in self.checker.directory
            if entry in folded
        ]

    WITHHELD = (
        "I can't answer that: the reply drew on content I could not verify, "
        "and removing it left nothing to say."
    )
