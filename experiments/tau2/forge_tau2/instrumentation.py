"""
Example-local observability instrumentation for Tau2 bench orchestration.

All communication uses gRPC (including cloud deployments via
Fly.io h2_backend).
"""

import contextvars
import json
from copy import deepcopy
from functools import lru_cache
from typing import Any, cast
from uuid import uuid4

from opentelemetry.trace import StatusCode
from sasy.capture import capture_logger, capture_text
from sasy.instrumentation.config import get_config
from sasy.instrumentation.feedback import (
    FeedbackScope,
    add_tool_denial,
)
from sasy.instrumentation.otel import get_tracer
from sasy.instrumentation.otel.context import (
    _current_input_ids,
)
from sasy.instrumentation.session import is_session_active
from wrapt import wrap_function_wrapper  # type: ignore[import-untyped]

# True while the airline/retail/etc env is replaying a trajectory in
# ``Environment.set_state`` — either the orchestrator's initial-state
# restore or, more importantly, the evaluator's predicted/gold replays
# in ``EnvironmentEvaluator.calculate_reward``.
#
# ``Environment.get_response`` is wrapped (see
# ``wrap_tool_message_response``) to gate every tool execution on a
# fresh policy check. That's correct during the live agent run, but
# during replay it makes the policy decision depend on the *current*
# session graph rather than the graph at the time of the original
# call. Sessions accumulate events as the run progresses, so a call
# the policy authorized at turn 5 may now be denied at eval time
# because an unrelated turn-12 event flipped a precondition — the
# replay would skip the call (line 332 of ``environment.py``) while
# the actual run executed it, and the predicted DB diverges from
# actual. ``get_user_details`` then returns different gift card /
# payment balances and ``set_state`` raises.
#
# When this flag is set, the wrapper short-circuits to the underlying
# ``Environment.get_response`` with no policy check and no event
# recording — replay reproduces actual exactly.
_replay_mode: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "sasy_tau2_replay_mode",
    default=False,
)

try:
    from pydantic import Extra
except ImportError:
    Extra = None  # type: ignore[misc,assignment]


def _import_grpc_deps() -> tuple:
    """Lazily import gRPC dependencies.

    Returns:
        Tuple of (record_events,
        record_events_with_dependencies, Edge, Event, Role,
        Tool, check_tool_call, ToolCallResponse).
    """
    from sasy.observability.api import (
        record_events,
        record_events_with_dependencies,
    )
    from sasy.proto.observability_pb2 import (
        Edge,
        Event,
        Role,
        Tool,
    )
    from sasy.proto.reference_monitor_pb2 import (
        ToolCallResponse,
    )
    from sasy.reference_monitor import check_tool_call
    return (
        record_events,
        record_events_with_dependencies,
        Edge,
        Event,
        Role,
        Tool,
        check_tool_call,
        ToolCallResponse,
    )


logger = capture_logger(__name__)


class Tau2InstrumentationError(RuntimeError):
    """The installed tau2 does not expose a point this module weaves.

    Raised by :func:`instrument` instead of silently skipping: a tau2
    run without these wrappers records no graph events and performs
    no tool-call policy checks, so a missing target must stop the run.
    """


# Every point :func:`instrument` wraps (or relies on), checked before
# any wrapper is installed so a partial install cannot happen. Each
# entry is ``(module, dotted attribute, leading parameter names after
# self)``; ``None`` skips the signature check and the "defined on this
# class itself" check (pydantic ``__init__`` is inherited). The parameter names
# matter because the wrappers read ``message`` / ``state`` either by
# keyword or by position. Layout matches tau2-bench v1.0.1.
_WEAVE_TARGETS: tuple[tuple[str, str, tuple[str, ...] | None], ...] = (
    ("tau2.agent.llm_agent", "LLMAgent.generate_next_message",
     ("message", "state")),
    ("tau2.agent.llm_agent", "LLMGTAgent.generate_next_message",
     ("message", "state")),
    ("tau2.agent.llm_agent", "LLMSoloAgent.generate_next_message",
     ("message", "state")),
    ("tau2.gym.gym_agent", "GymAgent.generate_next_message",
     ("message", "state")),
    ("tau2.gym.gym_agent", "GymUser.generate_next_message",
     ("message", "state")),
    ("tau2.gym.gym_agent", "GymAgentState", None),
    ("tau2.gym.gym_agent", "GymUserState", None),
    ("tau2.data_model.message", "SystemMessage.__init__", None),
    ("tau2.data_model.message", "AssistantMessage.__init__", None),
    ("tau2.data_model.message", "UserMessage.__init__", None),
    ("tau2.data_model.message", "ToolMessage.__init__", None),
    ("tau2.data_model.message", "MultiToolMessage.__init__", None),
    ("tau2.data_model.message", "ParticipantMessageBase.has_content", ()),
    ("tau2.data_model.message", "ParticipantMessageBase.is_tool_call", ()),
    ("tau2.user.user_simulator_base", "UserState.flip_roles", ()),
    ("tau2.user.user_simulator", "UserSimulator.generate_next_message",
     ("message", "state")),
    ("tau2.environment.environment", "Environment.get_response",
     ("message",)),
    ("tau2.environment.environment", "Environment.set_state",
     ("initialization_data", "initialization_actions", "message_history")),
)


def _verify_weave_targets() -> None:
    """Resolve every entry of :data:`_WEAVE_TARGETS` or raise.

    Collects all problems (missing module, missing attribute, changed
    leading parameters) into one :class:`Tau2InstrumentationError`.
    """
    import importlib
    import inspect

    problems: list[str] = []
    for module_name, dotted, params in _WEAVE_TARGETS:
        try:
            obj: Any = importlib.import_module(module_name)
        except ImportError as exc:
            problems.append(f"{module_name}: cannot import ({exc})")
            continue
        missing = False
        owner: Any = None
        for part in dotted.split("."):
            if not hasattr(obj, part):
                problems.append(f"{module_name}.{dotted}: missing")
                missing = True
                break
            owner, obj = obj, getattr(obj, part)
        if missing or params is None:
            continue
        # Methods must be defined on the named class itself, not
        # inherited: the wrappers replicate that exact implementation
        # (e.g. which incoming messages it appends to state).
        if isinstance(owner, type) and dotted.split(".")[-1] not in vars(owner):
            problems.append(
                f"{module_name}.{dotted}: no longer defined on"
                f" {owner.__name__} itself (inherited)"
            )
            continue
        try:
            names = [
                p.name
                for p in inspect.signature(obj).parameters.values()
            ]
        except (TypeError, ValueError) as exc:
            problems.append(
                f"{module_name}.{dotted}: no signature ({exc})"
            )
            continue
        if names and names[0] == "self":
            names = names[1:]
        if tuple(names[: len(params)]) != params:
            problems.append(
                f"{module_name}.{dotted}: expected leading parameters"
                f" {params}, found {tuple(names)}"
            )
    if problems:
        raise Tau2InstrumentationError(
            "sasy tau2 instrumentation cannot attach to the installed"
            " tau2 (expected the tau2-bench v1.0.1 layout); refusing to"
            " run uninstrumented:\n  " + "\n  ".join(problems)
        )


def _missing_sasy_tau2_patches() -> list[str]:
    """Return the SASY tau2 patches the installed tau2 lacks.

    The SDK attaches to upstream tau2 too, but two SASY patches to the
    vendored tree (``experiments/tau2-bench/PATCHES.md``) keep the
    measurement correct: ``ToolMessage.policy_denial`` (denials do not
    count toward ``max_errors``) and ``Environment.set_state`` skipping
    recorded denials during replay. The replay check runs the installed
    ``set_state`` (see :func:`_probe_set_state_skips_denials`); if the
    probe cannot run, set_state is reported as unverified.
    """
    import importlib

    missing: list[str] = []
    try:
        message_mod = importlib.import_module("tau2.data_model.message")
        env_mod = importlib.import_module("tau2.environment.environment")
    except ImportError:
        return missing  # _verify_weave_targets reports this
    if "policy_denial" not in getattr(
        message_mod.ToolMessage, "model_fields", {}
    ):
        missing.append(
            "ToolMessage.policy_denial is absent: policy denials count"
            " toward max_errors, so correct denials can end a trial"
            " as too_many_errors"
        )
    problem = _probe_set_state_skips_denials(message_mod, env_mod)
    if problem is not None:
        missing.append(problem)
    return missing


_PROBE_TOOL = "sasy_probe_mutating_tool"


def _probe_set_state_skips_denials(message_mod: Any, env_mod: Any) -> str | None:
    """Run the installed ``Environment.set_state`` on a recorded denial.

    The probe replays, with ``strict=True``, a short history against a
    stub environment whose only tool is a mutating one:

    1. a call answered by a denial in the legacy shape (``error=True``
       and ``[BLOCKED]`` content, no ``policy_denial``), as results saved
       before that field existed deserialize, which a patched set_state
       must skip;
    2. where ``ToolMessage.policy_denial`` exists, a call answered by a
       denial in the current shape (``policy_denial=True`` as well),
       which it must also skip. The legacy denial is a separate message
       so that a set_state skipping only on ``policy_denial`` is caught;
    3. a plain call with a matching recorded result, which any set_state
       must replay. It shows the probe reached the replay loop, so a
       set_state that replays nothing does not pass.

    Only the tool dispatch (``_has_tool``, ``_is_mutating_tool``,
    ``get_response``, ``sync_tools``) is stubbed; set_state's own code
    decides which calls to replay. No domain data is read.

    Returns None when every denial is skipped and the plain call replayed,
    else a message for :func:`_missing_sasy_tau2_patches`: a denial was
    replayed, or the probe could not run and set_state is unverified.
    """
    import inspect

    unverified = (
        "Environment.set_state unverified: cannot confirm that replay"
        " skips recorded policy denials"
    )
    try:
        tool_message_cls = message_mod.ToolMessage
        # (call id, extra ToolMessage fields) for each recorded denial.
        denial_shapes: list[tuple[str, dict[str, Any]]] = [
            ("sasy-probe-denied-legacy", {"error": True}),
        ]
        if "policy_denial" in getattr(tool_message_cls, "model_fields", {}):
            denial_shapes.append(
                ("sasy-probe-denied", {"error": True, "policy_denial": True})
            )
        denied_ids = [call_id for call_id, _ in denial_shapes]
        control = message_mod.ToolCall(
            id="sasy-probe-control", name=_PROBE_TOOL, arguments={}
        )
        history: list[Any] = []
        for call_id, fields in denial_shapes:
            call = message_mod.ToolCall(id=call_id, name=_PROBE_TOOL, arguments={})
            history.append(
                message_mod.AssistantMessage(role="assistant", tool_calls=[call])
            )
            history.append(
                tool_message_cls(
                    id=call_id,
                    role="tool",
                    content="[BLOCKED] sasy probe: policy denial",
                    requestor="assistant",
                    **fields,
                )
            )
        history += [
            message_mod.AssistantMessage(role="assistant", tool_calls=[control]),
            tool_message_cls(
                id=control.id,
                role="tool",
                content='{"ok": true}',
                requestor="assistant",
            ),
        ]
        set_state = inspect.unwrap(env_mod.Environment.set_state)

        class _ProbeEnvironment(env_mod.Environment):  # type: ignore[misc,name-defined]
            def __init__(self) -> None:  # no domain, no tools, no DB
                self.solo_mode = False
                self.replayed: list[str] = []

            def _has_tool(self, tool_name: str) -> bool:
                return tool_name == _PROBE_TOOL

            def _is_mutating_tool(self, tool_name: str) -> bool:
                return tool_name == _PROBE_TOOL

            def get_response(self, message: Any) -> Any:
                self.replayed.append(message.id)
                return tool_message_cls(
                    id=message.id,
                    role="tool",
                    content='{"ok": true}',
                    requestor="assistant",
                )

            def sync_tools(self) -> None:
                return None

        env = _ProbeEnvironment()
    except Exception as exc:
        return f"{unverified} (the probe could not be built: {exc!r})"

    try:
        set_state(env, None, None, history, strict=True)
    except Exception as exc:
        # strict=True raises on a replayed denial's content mismatch;
        # that case is reported below as not skipped.
        if not any(call_id in env.replayed for call_id in denied_ids):
            return f"{unverified} (the probe raised {exc!r})"
    if any(call_id in env.replayed for call_id in denied_ids):
        shape = (
            "legacy (error=True, [BLOCKED], no policy_denial)"
            if denied_ids[0] in env.replayed
            else "policy_denial=True"
        )
        return (
            "Environment.set_state does not skip recorded policy denials:"
            " replay re-executes denied state-changing calls (strict=True"
            " raises; strict=False mis-scores the database); replayed"
            f" a {shape} denial"
        )
    if control.id not in env.replayed:
        return (
            f"{unverified} (the probe's plain state-changing call was not"
            " replayed either, so the probe did not reach the replay)"
        )
    return None


def _positional_or_keyword(
    args: tuple, kwargs: dict, index: int, name: str
) -> Any:
    """Read a wrapped call's argument by keyword, else by position."""
    if name in kwargs:
        return kwargs[name]
    if len(args) > index:
        return args[index]
    return None


@lru_cache
def instrument() -> None:
    """Observability instrumentation for Tau2 benchmark.

    Wraps Tau2 LLMAgent methods to:

    1. Record messages and dependencies to the
       observability graph
    2. Create OpenTelemetry spans for each responder
    3. Link spans to messages via special attributes
    4. Accumulate authorization feedback per-responder
       scope

    Raises:
        Tau2InstrumentationError: tau2 is not installed, or a weave
            target is missing or has changed shape. Nothing is
            wrapped in that case.
    """
    # Initialize OTel (configures if not already done)
    TRACER = "instrumentation.tau2"
    get_tracer(TRACER)

    # Fail loud before wrapping anything: a silently skipped wrapper
    # means tool calls run with no policy check at all.
    _verify_weave_targets()
    # Upstream (unpatched) tau2 still works, but mis-measures denials:
    # say so instead of attaching silently.
    missing_patches = _missing_sasy_tau2_patches()
    if missing_patches:
        import warnings

        text = (
            "sasy tau2 instrumentation: the installed tau2 lacks SASY"
            " patches from experiments/tau2-bench/PATCHES.md; results that"
            " involve policy denials will be mis-measured:\n  "
            + "\n  ".join(missing_patches)
        )
        logger.warning(text)
        warnings.warn(text, RuntimeWarning, stacklevel=2)

    try:
        import tau2.agent.base_agent as agent_base
        import tau2.agent.llm_agent as llm_agent
        import tau2.data_model.message as message
        import tau2.gym.gym_agent as gym_agent
        import tau2.user.user_simulator_base as user_base
        from tau2.data_model.message import (
            Message,
            ParticipantMessageBase,
        )

        # Import gRPC dependencies and proto types
        (
            grpc_record_events,
            grpc_record_events_with_deps,
            Edge,
            Event,
            Role,
            Tool,
            grpc_check_tool_call,
            ToolCallResponse,
        ) = _import_grpc_deps()

        # Map tool call IDs to their originating output message ID.
        # Needed because tau2 creates ToolMessage objects from ToolCalls,
        # and extra attributes (_derived_from) don't transfer between them.
        _tool_call_origin: dict[str, str | None] = {}

        def translate_roles(role: str) -> Any:
            """Convert Tau2 roles into proto Roles."""
            match role:
                case "system":
                    return Role.SYSTEM
                case "assistant":
                    return Role.LLM
                case "tool":
                    return Role.AGENT
                case _:
                    return Role.USER

        def get_id(msg: Message) -> str | None:
            """
            Extract the Observability ID from the message.
            """
            if hasattr(msg, "_observability_id") and (
                id := getattr(msg, "_observability_id")
            ):
                return cast(str, id)

            return None

        def get_tools(
            msg: Message,
        ) -> list:  # list[Tool]
            """Extracts the tools from a Message."""
            if isinstance(msg, ParticipantMessageBase):
                return [
                    Tool(name=tc.name, arguments=json.dumps(tc.arguments))
                    for tc in msg.tool_calls or []
                ]
            return []

        def record_events_and_dependencies(
            message_history: list,
            record_index: bool = True,
        ) -> list[str]:
            """Record events and dependency edges via gRPC (atomic)."""
            input_ids = [
                e.id for e in message_history[:-1] if e.id
            ]
            output_id = message_history[-1].id
            dependencies = [
                Edge(
                    source=id,
                    destination=output_id,
                    message_index=(
                        i if record_index else None
                    ),
                    proximal=i == len(input_ids) - 1,
                )
                for i, id in enumerate(input_ids)
            ] if input_ids and output_id else []
            return grpc_record_events_with_deps(
                message_history, dependencies
            )

        def record_new_output_with_fanin(
            message_history: list,
            record_index: bool = True,
        ) -> list[str]:
            """Register only the last event in ``message_history`` and
            fan-in edges from every prior event id.

            Equivalent to :func:`record_events_and_dependencies` but
            assumes every event in ``message_history[:-1]`` is already
            in the graph. The server side
            (``GraphStore::merge_dependencies_unlocked``) silently
            skips edges whose source or destination is not present
            (logs a ``warn!``), so this trim is unsafe to use unless
            pre-registration is guaranteed by the caller.

            Used by the post-LLM history dump in ``llm_gen_wrapper``,
            where ``_record_events_dispatch(pre_events)`` runs
            synchronously before the LLM call and propagates any
            failure — so by the time we hit the post-LLM step, every
            input message is in the graph. Drops the per-call event
            payload from O(turn-index) to 1, reducing both wire bytes
            and server-side write-lock pressure (each batched event
            costs an upsert under the global graph RwLock; trimming
            shrinks each post-LLM batch by ~Nx where N is the
            conversation length).
            """
            if not message_history:
                return []
            input_ids = [
                e.id for e in message_history[:-1] if e.id
            ]
            output_event = message_history[-1]
            output_id = output_event.id
            dependencies = [
                Edge(
                    source=src,
                    destination=output_id,
                    message_index=(i if record_index else None),
                    proximal=i == len(input_ids) - 1,
                )
                for i, src in enumerate(input_ids)
            ] if input_ids and output_id else []
            return grpc_record_events_with_deps(
                [output_event], dependencies
            )

        def _record_events_dispatch(events: list) -> list[str]:
            """Record events via gRPC."""
            return grpc_record_events(events)

        def _record_events_with_deps_dispatch(
            events: list,
            dependencies: list,
        ) -> list[str]:
            """Record events + deps via gRPC."""
            return grpc_record_events_with_deps(
                events, dependencies
            )

        def llm_gen_wrapper(agent_name: str, user_simulator: bool = False):
            """Patched LLM generation step."""
            def wrapper(wrapped, instance, args, kwargs):
                # Extract the message argument from the responder call
                if not is_session_active():
                    return wrapped(*args, **kwargs)
                msg: agent_base.ValidAgentInputMessage | None = (
                    _positional_or_keyword(args, kwargs, 0, "message")
                )
                state: (
                    llm_agent.LLMAgentState | user_base.UserState | gym_agent.GymUserState | gym_agent.GymAgentState
                ) = deepcopy(_positional_or_keyword(args, kwargs, 1, "state")) # Gym agents may have state manipulation during the responder execution

                # Replicate the responder's own "append incoming message
                # to state" logic so the recorded input ids are exactly
                # the messages the LLM sees (tau2 v1.0.1):
                # - ``None`` (LLMSoloAgent / Gym first turn) appends
                #   nothing;
                # - MultiToolMessage appends each tool message;
                # - UserSimulator skips an incoming non-tool message
                #   with no content and no tool calls (empty streaming
                #   chunks); agents append unconditionally.
                message_in: list
                if msg is None:
                    message_in = []
                elif isinstance(msg, message.MultiToolMessage):
                    message_in = list(msg.tool_messages)
                elif (
                    user_simulator
                    and not isinstance(msg, message.ToolMessage)
                    and not (msg.has_content() or msg.is_tool_call())
                ):
                    message_in = []
                else:
                    message_in = [msg]

                # Collect input message IDs before execution (filter out None values)
                if user_simulator:
                    # Track flipped IDs
                    state_copy = user_base.UserState(
                        system_messages=state.system_messages,
                        messages=state.messages + message_in,
                    )
                    messages_in = state_copy.system_messages + state_copy.flip_roles()
                elif any(isinstance(state, c) for c in [gym_agent.GymAgentState, gym_agent.GymUserState]):
                    # Only uses messages, no separate system messages field
                    messages_in = state.messages + message_in
                else:
                    messages_in = state.system_messages + state.messages + message_in

                # Ensure all input messages have observability IDs
                # (fallback if monkey patching didn't apply due to import order)
                for msg in messages_in:
                    if not hasattr(msg, "_observability_id") or not msg._observability_id:
                        msg._observability_id = str(uuid4())

                input_ids = [
                    id
                    for msg in messages_in
                    if (id := get_id(msg)) is not None
                ]

                # Create span name
                span_name = "tau2.user_simulator" if user_simulator else "tau2.llm_gen"

                with get_tracer(TRACER).start_as_current_span(span_name) as span:
                    # Set attributes BEFORE execution so they're available
                    span.set_attribute("agent.name", agent_name)
                    span.set_attribute("responder.type", "llm_response")
                    if input_ids:
                        span.set_attribute(
                            "_input_message_ids", ",".join(input_ids)
                        )
                        # Propagate input IDs to nested calls via ContextVar
                        token = _current_input_ids.set(input_ids)
                    else:
                        token = None

                    try:
                        try:
                            # Pre-record input events so
                            # they exist before the LLM
                            # call fires.
                            if input_ids:
                                pre_events = [
                                    Event(
                                        text=msg.content,
                                        role=translate_roles(
                                            msg.role
                                        ),
                                        agent=agent_name,
                                        tools=get_tools(msg),
                                        id=get_id(msg),
                                    )
                                    for msg in messages_in
                                    if get_id(msg)
                                ]
                                if pre_events:
                                    _record_events_dispatch(
                                        pre_events
                                    )

                            # Generate edges from user simulator gen step
                            if not user_simulator:
                                for msg in messages_in:
                                    if getattr(msg, "_generated_by_user_simulator", False) and hasattr(msg, "_flipped_observability_id"):
                                        current_agent_msg = Event(
                                            text=msg.content,
                                            role=translate_roles(msg.role),
                                            agent=agent_name,
                                            tools=get_tools(msg),
                                            id=get_id(msg),
                                        )
                                        flipped_msg = Event(
                                            text=msg.content,
                                            role=Role.LLM,
                                            agent="UserSimulator",
                                            tools=get_tools(msg),
                                            id=msg._flipped_observability_id,
                                        )
                                        record_events_and_dependencies([flipped_msg, current_agent_msg], record_index=False)

                            # FeedbackScope accumulates auth feedback during execution
                            with FeedbackScope(agent_name, "llm") as feedback_scope:
                                output_message, output_state = wrapped(*args, **kwargs)

                            if user_simulator and not hasattr(output_message, "_flipped_observability_id"):
                                output_message._flipped_observability_id = str(uuid4())
                                output_message._generated_by_user_simulator = True

                            # Process feedback after execution (callback can modify output)
                            # Pass the state to the "agent" field
                            feedback_scope.process(output_message, output_state)

                            # Ensure output_message has an observability ID
                            # (fallback if monkey patching didn't apply due to import order)
                            if not hasattr(output_message, "_observability_id") or not output_message._observability_id:
                                output_message._observability_id = str(uuid4())

                            if output_message.tool_calls is not None:
                                for tc in output_message.tool_calls:
                                    tc._derived_from = get_id(output_message)
                                    if hasattr(tc, "id") and tc.id:
                                        _tool_call_origin[tc.id] = get_id(output_message)

                        except Exception as e:
                            # Record exception on span and set error status
                            span.record_exception(e)
                            span.set_status(StatusCode.ERROR, str(e))
                            logger.error(str(e))
                            raise

                        if output_message and output_state:
                            history = [
                                Event(
                                    text=msg.content,
                                    role=translate_roles(msg.role),
                                    agent=agent_name,
                                    tools=get_tools(msg),
                                    id=get_id(msg),
                                )
                                for msg in messages_in + [output_message]
                            ]

                            # Record the actual generation step
                            if user_simulator:
                                history[-1].id = output_message._flipped_observability_id
                                history[-1].role = Role.LLM

                            # Record events and dependencies. Every
                            # entry in ``messages_in`` was already
                            # registered above via
                            # ``_record_events_dispatch(pre_events)``
                            # (synchronous, propagates failures), so
                            # only ``output_message`` is new at this
                            # point. Use the trimmed path that sends
                            # one event + the fan-in edges instead of
                            # re-sending the full history each turn.
                            ids = record_new_output_with_fanin(
                                history, record_index=True,
                            )

                            if ids:
                                # The ID of the output message
                                span.set_attribute("_output_message_id", get_id(output_message) if user_simulator else ids[-1])
                                if user_simulator:
                                    # The ID of the generated message for user simulators
                                    span.set_attribute("_output_message_flipped_id", ids[-1])

                        span.set_status(StatusCode.OK)
                        return output_message, output_state
                    finally:
                        # Reset ContextVar to previous value
                        if token is not None:
                            _current_input_ids.reset(token)

            return wrapper

        for class_name in ["LLMAgent", "LLMGTAgent", "LLMSoloAgent"]:
            wrap_function_wrapper(
                "tau2.agent.llm_agent",
                f"{class_name}.generate_next_message",
                llm_gen_wrapper(class_name),
            )

        for class_name in ["GymAgent", "GymUser"]:
            wrap_function_wrapper(
                "tau2.gym.gym_agent",
                f"{class_name}.generate_next_message",
                llm_gen_wrapper(class_name),
            )

        def track_id_wrapper(wrapped, instance, args, kwargs):
            """Track unique ids in Messages."""
            if not is_session_active():
                return wrapped(*args, **kwargs)
            wrapped(*args, **kwargs)
            if hasattr(instance, "model_config"):
                instance.model_config["extra"] = "allow"
            elif Extra is not None:
                # Fallback for Pydantic v1
                instance.__config__.extra = Extra.allow

            instance._observability_id = str(uuid4())


        for msg_class in [
            "SystemMessage",
            "AssistantMessage",
            "UserMessage",
            "ToolMessage",
            "MultiToolMessage",
        ]:
            wrap_function_wrapper(
                "tau2.data_model.message",
                f"{msg_class}.__init__",
                track_id_wrapper
            )


        def wrap_flip_roles(wrapped, instance: user_base.UserState, args, kwargs):
            if not is_session_active():
                return wrapped(*args, **kwargs)
            for msg in instance.messages:
                if not hasattr(msg, "_observability_id"):
                    msg._observability_id = str(uuid4())
                if not hasattr(msg, "_flipped_observability_id"):
                    msg._flipped_observability_id = str(uuid4())
            flipped_messages = wrapped(*args, **kwargs)

            # The pairing below assumes flip_roles maps messages 1:1
            # (true in tau2 v1.0.1: every branch appends one message
            # or raises). A dropped or added message would silently
            # attach ids to the wrong turns, so refuse instead.
            if len(flipped_messages) != len(instance.messages):
                raise Tau2InstrumentationError(
                    "UserState.flip_roles returned"
                    f" {len(flipped_messages)} messages for"
                    f" {len(instance.messages)} inputs; sasy tau2"
                    " instrumentation requires a 1:1 mapping"
                )

            # Record the derivation of the flipped copies
            # (MERGE in Neo4j prevents duplicate edges)
            for flipped, orig in zip(flipped_messages, instance.messages):
                flipped._observability_id = orig._flipped_observability_id
                for i, tc in enumerate(getattr(flipped, "tool_calls", []) or []):
                    if hasattr(orig.tool_calls[i], "_derived_from"):
                       tc._derived_from = orig.tool_calls[i]._derived_from
            in_messages = [
                Event(
                    text=msg.content,
                    role=translate_roles(msg.role),
                    tools=get_tools(msg),
                    id=get_id(msg),
                )
                for msg in instance.messages
            ]
            out_messages = [
                Event(
                    text=msg.content,
                    role=translate_roles(msg.role),
                    agent="UserSimulator",
                    tools=get_tools(msg),
                    id=get_id(msg),
                )
                for msg, in_msg in zip(flipped_messages, instance.messages)
            ]
            dependencies = [
                Edge(
                    source=in_msg.id,
                    destination=out_msg.id,
                )
                for in_msg, out_msg in zip(in_messages, out_messages)
                if in_msg.id and out_msg.id
            ]
            _record_events_with_deps_dispatch(
                in_messages + out_messages, dependencies
            )
            return flipped_messages

        wrap_function_wrapper(
            "tau2.user.user_simulator_base",
            "UserState.flip_roles",
            wrap_flip_roles,
        )

        wrap_function_wrapper(
            "tau2.user.user_simulator",
            "UserSimulator.generate_next_message",
            llm_gen_wrapper("UserSimulator", user_simulator=True),
        )

        # Tool call policy enforcement wrappers

        def _extract_denial_message(response: Any) -> str:
            """Extract denial message from gRPC ToolCallResponse."""
            if response.denial_trace:
                msg = response.denial_trace.action_description
                if response.denial_trace.reasons:
                    reason_parts = [
                        (
                            reason.details
                            if reason.details
                            else str(reason.reason_type)
                        )
                        for reason in (
                            response.denial_trace.reasons
                        )
                    ]
                    msg += ": " + "; ".join(reason_parts)
                return msg
            return "Tool call not authorized"

        def _extract_suggestions(
            response: Any,
        ) -> list[str]:
            """Extract suggestions from gRPC ToolCallResponse."""
            if (
                response.denial_trace
                and response.denial_trace.suggested_fixes
            ):
                return list(
                    response.denial_trace.suggested_fixes
                )
            return []

        def _format_denial(
            fn_name: str,
            denial_msg: str,
            suggestions: list[str],
        ) -> str:
            """Format a denial content string."""
            content = f"[BLOCKED] {fn_name}: {capture_text(denial_msg)}"
            if suggestions:
                if len(suggestions) == 1:
                    content += (
                        f"\nRequired action:"
                        f" {suggestions[0]}"
                    )
                else:
                    content += (
                        "\nRequired actions:\n"
                        + "\n".join(
                            f"  - {s}" for s in suggestions
                        )
                    )
            return content

        def _make_policy_denial_message(
            message_mod,
            tool_id: str,
            content: str,
            requestor,
        ):
            """Construct a tau2 ``ToolMessage`` for a policy denial.

            Sets ``error=True`` so existing rendering / serialization
            paths (CLI view, JSON dump, agent prompt formatting)
            continue to surface the denial. ALSO sets
            ``policy_denial=True`` when the local tau2 version supports
            the field — the upstream patch in
            ``experiments/tau2-bench/.../message.py`` adds it. The
            orchestrator consults that flag to decide whether to count
            the message toward ``max_errors``.

            Skips ``policy_denial=`` when the field is absent in the
            installed tau2, so this instrumentation keeps working
            against the upstream tau2; :func:`instrument` warns in that
            case (see :func:`_missing_sasy_tau2_patches`).
            """
            kwargs = dict(
                id=tool_id,
                content=content,
                requestor=requestor,
                role="tool",
                error=True,
            )
            ToolMsg = message_mod.ToolMessage
            if "policy_denial" in ToolMsg.model_fields:
                kwargs["policy_denial"] = True
            return ToolMsg(**kwargs)

        def _record_tool_denial_event(
            out: Any,
            tool_id: str | None,
        ) -> None:
            """Record a denial event + edge."""
            denied_event = Event(
                text=out.content,
                role=translate_roles(out.role),
                id=get_id(out),
            )
            if tool_id and get_id(out):
                edge = Edge(
                    source=tool_id,
                    destination=get_id(out),
                    proximal=True,
                )
                _record_events_with_deps_dispatch(
                    [denied_event], [edge]
                )
            else:
                _record_events_dispatch([denied_event])

        def wrap_tool_message_response(
            wrapped, instance, args, kwargs
        ):
            """Check tool call authorization before executing."""
            if not is_session_active():
                return wrapped(*args, **kwargs)
            tool = kwargs.get(
                "message", args[0] if args else None
            )
            if tool is None:
                return wrapped(*args, **kwargs)

            # During ``Environment.set_state`` replay (eval-time gold
            # / predicted DB reconstruction, or orchestrator initial-
            # state restore), bypass the policy check and event
            # recording entirely. See the ``_replay_mode`` docstring
            # at the top of this file for why.
            if _replay_mode.get():
                return wrapped(*args, **kwargs)

            fn_name = tool.name
            fn_args = json.dumps(tool.arguments)

            tool_call_id = getattr(tool, "id", None)
            tool_id = getattr(
                tool, "_derived_from", None
            ) or (
                _tool_call_origin.get(tool_call_id)
                if tool_call_id is not None else None
            )
            input_ids = (
                [tool_id]
                if tool_id is not None
                else []
            )
            config = get_config()

            span_name = "tau2.tool_call"

            with get_tracer(TRACER).start_as_current_span(
                span_name
            ) as span:
                span.set_attribute("fn.name", fn_name)
                span.set_attribute("fn.args", fn_args)
                try:
                    response = grpc_check_tool_call(
                        fn_name, fn_args, input_ids
                    )
                    if not response.authorized:
                        denial_msg = (
                            _extract_denial_message(
                                response
                            )
                        )
                        suggestions = (
                            _extract_suggestions(
                                response
                            )
                        )
                        add_tool_denial(
                            fn_name,
                            denial_msg,
                            fn_args,
                            suggestions,
                        )
                        content = _format_denial(
                            fn_name,
                            denial_msg,
                            suggestions,
                        )
                        if config.log_policy_decisions:
                            print(
                                "\033[91m\033[1m"
                                "[POLICY] DENIED"
                                f" {fn_name}\033[0m"
                            )
                            if denial_msg:
                                print(
                                    "\033[91m"
                                    "  Reason:"
                                    f" {capture_text(denial_msg)}"
                                    "\033[0m"
                                )
                            for s in suggestions:
                                print(
                                    "\033[93m"
                                    "  Suggestion:"
                                    f" {capture_text(s)}\033[0m"
                                )
                        logger.info(
                            "Tool call denied:"
                            f" {fn_name} -"
                            f" {capture_text(denial_msg)}"
                        )
                        out = _make_policy_denial_message(
                            message,
                            tool.id,
                            content,
                            tool.requestor,
                        )
                        _record_tool_denial_event(
                            out, tool_id
                        )
                        return out

                except Exception as e:
                    if config.tool_policy_fail_closed:
                        logger.warning(
                            "Tool call denied (RM"
                            " unavailable,"
                            " fail_closed=True):"
                            f" {fn_name} - {e}"
                        )
                        add_tool_denial(
                            fn_name,
                            "Reference monitor unavailable",
                            fn_args,
                            [],
                        )
                        out = _make_policy_denial_message(
                            message,
                            tool.id,
                            (
                                f"[BLOCKED] {fn_name}:"
                                " Reference monitor"
                                " unavailable"
                            ),
                            tool.requestor,
                        )
                        return out
                    else:
                        logger.debug(
                            "Tool policy check failed"
                            f" (passthrough): {e}"
                        )

                if config.log_policy_decisions:
                    print(
                        "\033[92m\033[1m"
                        "[POLICY] AUTHORIZED"
                        f" {fn_name}\033[0m"
                    )

                out = wrapped(*args, **kwargs)

                result_event = Event(
                    text=out.content,
                    role=translate_roles(out.role),
                    derived_from=Tool(
                        name=fn_name,
                        arguments=fn_args,
                    ),
                    id=get_id(out),
                )
                if tool_id and get_id(out):
                    edge = Edge(
                        source=tool_id,
                        destination=get_id(out),
                        proximal=True,
                    )
                    _record_events_with_deps_dispatch(
                        [result_event], [edge]
                    )
                else:
                    _record_events_dispatch(
                        [result_event]
                    )

                return out

        wrap_function_wrapper(
            "tau2.environment.environment",
            name="Environment.get_response",
            wrapper=wrap_tool_message_response,
        )

        def wrap_set_state(wrapped, instance, args, kwargs):
            """Flip ``_replay_mode`` for the duration of set_state.

            ``Environment.set_state`` walks a recorded trajectory and
            calls ``self.get_response(tool_call)`` for each non-blocked
            entry to reconstruct the env's DB. Those calls are pure
            replay — they should reproduce the actual run's responses
            exactly, not be re-checked against a session graph that
            has moved on since the original call.

            All arguments pass through unchanged, including the
            ``strict`` keyword tau2 v1.0.1 added.
            """
            if not is_session_active():
                return wrapped(*args, **kwargs)
            token = _replay_mode.set(True)
            try:
                return wrapped(*args, **kwargs)
            finally:
                _replay_mode.reset(token)

        wrap_function_wrapper(
            "tau2.environment.environment",
            name="Environment.set_state",
            wrapper=wrap_set_state,
        )

    except ImportError as exc:
        raise Tau2InstrumentationError(
            f"sasy tau2 instrumentation failed to import a dependency: {exc}"
        ) from exc
