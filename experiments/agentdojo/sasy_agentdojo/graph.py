"""Record an AgentDojo conversation into the SASY observability graph.

The policy engine decides from the dependency graph, so every message the
agent sees has to exist there before a tool call is checked against it. This
module does the translation, and it is the whole of what "instrumenting an
AgentDojo agent" means:

    AgentDojo message            SASY event
    ─────────────────────────    ────────────────────────────────────────
    system                       Event(role=SYSTEM)
    user                         Event(role=USER)          ← the trusted request
    assistant (+ tool_calls)     Event(role=LLM, tools=[…])
    tool result                  Event(role=AGENT, derived_from=Tool(…))
                                 which the engine projects to
                                 ToolResult(id, fn_name, args)

Edges run from every earlier message to the new one, which is what makes
`CurrentDepends` the agent's context: anything the model could have read
before producing a call is reachable backwards from that call. The last edge
of each batch is marked `proximal`, matching what the tau2 instrumentation
records, so a policy can tell "the message being answered" from "something
said earlier".

Message identity is a UUID we mint and keep in a side table keyed by the
message's position in the list, because AgentDojo's `ChatMessage` is a
TypedDict and cannot carry an extra attribute the way tau2's pydantic models
can.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any
from uuid import uuid4

from agentdojo.types import ChatMessage, get_text_content_as_str

from sasy.observability.api import record_events_with_dependencies
from sasy.proto.observability_pb2 import Edge, Event, Role, Tool

_ROLE = {
    "system": Role.SYSTEM,
    "user": Role.USER,
    "assistant": Role.LLM,
    "tool": Role.AGENT,
}


def message_text(message: ChatMessage) -> str:
    content = message.get("content")
    if content is None:
        return ""
    if isinstance(content, str):  # pre-content-block traces
        return content
    return get_text_content_as_str(content)


def _tools(message: ChatMessage) -> list[Tool]:
    if message.get("role") != "assistant":
        return []
    return [
        Tool(name=tc.function, arguments=json.dumps(dict(tc.args), default=str))
        for tc in (message.get("tool_calls") or [])
    ]


#: The first line of the monitor's refusal text. Matched as a PREFIX because
#: the reason and the suggestion follow it in the same field — AgentDojo shows
#: the model `error`, so that is where the detail has to live.
REFUSAL_PREFIX = "This call was refused by the security policy."

#: The executor's replies to a call it never dispatched. AgentDojo's own
#: wording, kept as constants so both layers recognise them.
MALFORMED_EMPTY = "Empty function name provided. Provide a valid function name."
MALFORMED_UNKNOWN_PREFIX = "Invalid tool "


def did_not_run(message) -> bool:
    """Does this tool message record a call that never executed?

    Three ways a call fails to happen, and all of them look like a tool result
    to anything reading the message list: the policy refused it, the model named
    a tool that does not exist, or it named no tool at all. None is an effect,
    and a rule asking what has already happened must not count them.

    A tool that RAN and raised is deliberately not here. That call happened, and
    its failure is a fact about the world like any other result.
    """
    err = message.get("error") or ""
    return (
        err.startswith(REFUSAL_PREFIX)
        or err == MALFORMED_EMPTY
        or err.startswith(MALFORMED_UNKNOWN_PREFIX)
    )


def _derived_from(message: ChatMessage) -> Tool | None:
    """The tool whose RESULT this message is, or None.

    A call that never ran is deliberately excluded — refused by the policy, or
    naming a tool that does not exist. The message exists because the model has
    to be told, but nothing executed and there is no result, so projecting it as
    a `ToolResult` would state something false about the world.

    It matters beyond tidiness. Any rule that reasons about what has already
    happened — "an effect of this kind occurred", "this content entered the
    context" — would count the attempt as the thing it prevented.
    """
    if message.get("role") != "tool":
        return None
    call = message.get("tool_call")
    if call is None:
        return None
    if did_not_run(message):
        return None
    return Tool(name=call.function, arguments=json.dumps(dict(call.args), default=str))


class ConversationRecorder:
    """Keeps the graph in step with a growing message list.

    `sync` is called with the whole list each time the pipeline advances. It
    registers whatever is new and returns the id of the last message, which is
    what a tool-call check passes as its input node.
    """

    def __init__(self, agent_name: str = "agentdojo") -> None:
        self.agent_name = agent_name
        self.ids: list[str] = []
        #: How many of `ids` came from a MESSAGE. Denial nodes are appended to
        #: `ids` as well, so this is the only safe resume point for `sync`.
        self.recorded = 0
        #: A digest per recorded message, so a restart is detected by what the
        #: messages ARE and not by how many there are. See `sync`.
        self.digests: list[str] = []

    def reset(self) -> None:
        self.ids = []
        self.recorded = 0
        self.digests = []

    @staticmethod
    def _digest(message: ChatMessage) -> str:
        """Identity of a message for restart detection.

        Role, text and the tool call are enough to tell two attempts apart, and
        all three are cheap to read. Deliberately not the object id: AgentDojo
        rebuilds the message objects each attempt, so identity by reference
        would report every message as changed.
        """
        def one(call) -> str:
            return f"{call.function}:{sorted(dict(call.args).items())}"

        # BOTH fields, because they belong to different message kinds and the
        # case this exists for lives in the plural one. An assistant message
        # carries the calls it PROPOSES in `tool_calls`; a tool-result message
        # carries the single call it answers in `tool_call`. Reading only the
        # singular field left every assistant digest without its calls, so two
        # attempts that differ solely in what the model decided to do — the
        # ordinary shape of a retry — compared equal and the restart went
        # undetected.
        proposed = message.get("tool_calls") or []
        answered = message.get("tool_call")
        return "\x00".join((
            str(message.get("role")),
            message_text(message),
            "|".join(one(c) for c in proposed),
            "" if answered is None else one(answered),
            str(message.get("error") or ""),
        ))

    def _event(self, message: ChatMessage, event_id: str) -> Event:
        event = Event(
            text=message_text(message),
            role=_ROLE.get(str(message.get("role")), Role.AGENT),
            agent=self.agent_name,
            tools=_tools(message),
            id=event_id,
        )
        derived = _derived_from(message)
        if derived is not None:
            event.derived_from.CopyFrom(derived)
        return event

    def _prefix_changed(self, messages: Sequence[ChatMessage]) -> bool:
        """Detect changed messages at recorded positions, including retries
        whose initial message count matches the previous attempt.
        """
        for i in range(min(self.recorded, len(messages))):
            if self.digests[i] != self._digest(messages[i]):
                return True
        return False

    def sync(self, messages: Sequence[ChatMessage]) -> str | None:
        """Record new messages and return the last node ID, or None.

        Reset tracking when a conversation restarts or its recorded prefix
        changes. Resume from the recorded message count; graph IDs also
        include denial nodes that do not correspond to messages.
        """
        if len(messages) < self.recorded or self._prefix_changed(messages):
            self.reset()
        if len(messages) <= self.recorded:
            return self.ids[-1] if self.ids else None

        for position in range(self.recorded, len(messages)):
            message = messages[position]
            event_id = str(uuid4())
            inputs = list(self.ids)
            edges = [
                Edge(
                    source=source,
                    destination=event_id,
                    message_index=i,
                    proximal=(i == len(inputs) - 1),
                )
                for i, source in enumerate(inputs)
            ]
            record_events_with_dependencies([self._event(message, event_id)], edges)
            self.ids.append(event_id)
            self.digests.append(self._digest(message))
            self.recorded += 1

        return self.ids[-1]

    def record_denial(self, text: str, parent_id: str | None) -> str:
        """Register a policy denial as a node of its own, so a later call can
        see in the graph that this one was refused."""
        event_id = str(uuid4())
        event = Event(text=text, role=Role.AGENT, agent=self.agent_name, id=event_id)
        edges = (
            [Edge(source=parent_id, destination=event_id, message_index=0, proximal=True)]
            if parent_id
            else []
        )
        record_events_with_dependencies([event], edges)
        self.ids.append(event_id)
        return event_id


def call_signature(function: str, args: dict[str, Any]) -> tuple[str, str]:
    return function, json.dumps(dict(args), default=str, sort_keys=True)
