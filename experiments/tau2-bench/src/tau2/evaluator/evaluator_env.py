from typing import Callable, Optional

from loguru import logger

from tau2.data_model.message import (
    AssistantMessage,
    Message,
    Tick,
    UserMessage,
)
from tau2.data_model.simulation import DBCheck, EnvAssertionCheck, RewardInfo
from tau2.data_model.tasks import Action, RewardType, Task
from tau2.environment.environment import Environment
from tau2.environment.toolkit import ToolType, get_tool_types
from tau2.evaluator.evaluator_base import EvaluatorBase


# SASY patch: shared helper for replaying golden actions with failure tracking.
# A failed golden WRITE must not silently produce a successful DB score.
def _replay_golden_actions(
    gold_environment: Environment,
    golden_actions: list[Action],
    tool_types: Optional[dict[str, ToolType]] = None,
) -> list[dict]:
    """Replay golden actions and return structured failure records.

    Returns a list of dicts, one per failed action, each with keys:
      - name: the tool name
      - arguments: the tool arguments
      - error: the exception message
      - tool_type: the ToolType if known, else None
    """
    failures = []
    for action in golden_actions:
        try:
            gold_environment.make_tool_call(
                tool_name=action.name,
                requestor=action.requestor,
                **action.arguments,
            )
        except Exception as e:
            tt = tool_types.get(action.name) if tool_types else None
            failures.append(
                {
                    "name": action.name,
                    "arguments": action.arguments,
                    "error": str(e),
                    "tool_type": tt.value if tt else None,
                }
            )
            logger.warning(
                f"Error in golden actions {action.name}({action.arguments}): {e}"
            )
    return failures


def _has_invalid_reference(
    failures: list[dict],
    tool_types: Optional[dict[str, ToolType]] = None,
) -> bool:
    """True when a golden-action failure invalidates the reference.

    Fail-closed: a failed action whose type is WRITE or unknown
    invalidates the reference. Only a known READ (or THINK/GENERIC)
    failure is treated as an expected probe that does not invalidate.
    """
    for f in failures:
        tt = f.get("tool_type")
        if tt is None and tool_types:
            resolved = tool_types.get(f["name"])
            tt = resolved.value if resolved else None
        if tt == ToolType.READ.value:
            continue
        if tt in (ToolType.THINK.value, ToolType.GENERIC.value):
            continue
        # WRITE or unknown → invalid reference (fail-closed)
        return True
    return False


def _env_evaluate_common(
    environment_constructor: Callable[..., Environment],
    task: Task,
    predicted_message_history: list[Message],
    solo_mode: bool = False,
    env_kwargs: dict = None,
    strict_replay: bool = True,
) -> RewardInfo:
    """Shared evaluation logic for both half-duplex and full-duplex modes."""
    if task.evaluation_criteria is None:
        return RewardInfo(
            reward=1.0,
            info={"note": "No evaluation criteria"},
        )
    expected_actions = task.evaluation_criteria.actions
    env_assertions = task.evaluation_criteria.env_assertions
    if expected_actions is None and env_assertions is None:
        return RewardInfo(
            reward=1.0,
            db_check=DBCheck(db_match=True, db_reward=1.0),
            info={"note": "No expected actions or env assertions"},
        )

    initialization_data = None
    if (
        task.initial_state is not None
        and task.initial_state.initialization_data is not None
    ):
        initialization_data = task.initial_state.initialization_data

    initialization_actions = None
    if (
        task.initial_state is not None
        and task.initial_state.initialization_actions is not None
    ):
        initialization_actions = task.initial_state.initialization_actions

    message_history = []
    if (
        task.initial_state is not None
        and task.initial_state.message_history is not None
    ):
        message_history = task.initial_state.message_history

    if env_kwargs is None:
        env_kwargs = {}

    predicted_environment = environment_constructor(solo_mode=solo_mode, **env_kwargs)
    predicted_environment.set_state(
        initialization_data=initialization_data,
        initialization_actions=initialization_actions,
        message_history=predicted_message_history,
        strict=strict_replay,
    )

    # Setting up gold environment
    gold_environment = environment_constructor(**env_kwargs)
    gold_environment.set_state(
        initialization_data=initialization_data,
        initialization_actions=initialization_actions,
        message_history=message_history,
        strict=strict_replay,
    )

    # SASY patch: get tool types for failure classification
    tool_types: Optional[dict[str, ToolType]] = None
    try:
        if gold_environment.tools is not None:
            tool_types = get_tool_types(gold_environment.tools)
    except Exception:
        pass

    golden_actions = task.evaluation_criteria.actions or []
    gold_failures = _replay_golden_actions(gold_environment, golden_actions, tool_types)

    # Comparing the environments
    agent_db_hash = gold_environment.get_db_hash()
    user_db_hash = gold_environment.get_user_db_hash()
    predicted_agent_db_hash = predicted_environment.get_db_hash()
    predicted_user_db_hash = predicted_environment.get_user_db_hash()
    agent_db_match = agent_db_hash == predicted_agent_db_hash
    user_db_match = user_db_hash == predicted_user_db_hash

    # SASY patch: a failed golden WRITE invalidates the DB comparison.
    # The gold environment didn't reach the intended state, so a hash
    # match is coincidental (the predicted env also didn't do the write).
    has_invalid_ref = _has_invalid_reference(gold_failures, tool_types)

    if has_invalid_ref:
        db_reward = 0.0
        db_match = False
        logger.error(
            "Golden reference has a failed WRITE action; "
            "DB score forced to 0 (invalid_reference)"
        )
    elif agent_db_match and user_db_match:
        db_reward = 1.0
        db_match = True
    else:
        db_reward = 0.0
        db_match = False

    db_check = DBCheck(db_match=db_match, db_reward=db_reward)

    # Run env assertions
    env_assertions_list = task.evaluation_criteria.env_assertions or []
    env_assertion_checks = []
    env_assertion_reward = 1.0
    for env_assertion in env_assertions_list:
        success = predicted_environment.run_env_assertion(
            env_assertion,
            raise_assertion_error=False,
        )
        res = EnvAssertionCheck(
            env_assertion=env_assertion,
            met=success,
            reward=1.0 if success else 0.0,
        )
        env_assertion_checks.append(res)
        env_assertion_reward *= res.reward

    reward = 1.0
    reward_breakdown = {}
    if RewardType.DB in task.evaluation_criteria.reward_basis:
        reward_breakdown[RewardType.DB] = db_reward
        reward *= db_reward
    if RewardType.ENV_ASSERTION in task.evaluation_criteria.reward_basis:
        reward_breakdown[RewardType.ENV_ASSERTION] = env_assertion_reward
        reward *= env_assertion_reward

    # SASY patch: include gold-action failures and invalid_reference in info
    info = {}
    if gold_failures:
        info["gold_action_failures"] = gold_failures
    if has_invalid_ref:
        info["invalid_reference"] = True

    return RewardInfo(
        reward=reward,
        db_check=db_check,
        env_assertions=env_assertion_checks,
        reward_basis=task.evaluation_criteria.reward_basis,
        reward_breakdown=reward_breakdown,
        info=info if info else None,
    )


class EnvironmentEvaluator(EvaluatorBase[Message]):
    """
    Evaluator focuses on endstate of the simulation environment.
    """

    @classmethod
    def calculate_reward(
        cls,
        environment_constructor: Callable[[], Environment],
        task: Task,
        full_trajectory: list[
            Message
        ],  # FIXME: It would be better to be able to get only the messages that are after the initial state
        solo_mode: bool = False,
        env_kwargs: dict = None,
        strict_replay: bool = True,
    ) -> RewardInfo:
        """
        Calculate the reward for the simulation.
        Args:
            environment_constructor: Callable[[], Environment]
            task: Task
            full_trajectory: list[Message] (Must include the message history from task initial state)
            solo_mode: bool
            strict_replay: forwarded to Environment.set_state(strict=...). Set
                False when re-grading historical trajectories whose recorded
                tool outputs may cosmetically differ from current tool code.
        Returns:
            RewardInfo
        """
        return _env_evaluate_common(
            environment_constructor=environment_constructor,
            task=task,
            predicted_message_history=list(full_trajectory),
            solo_mode=solo_mode,
            env_kwargs=env_kwargs,
            strict_replay=strict_replay,
        )


class FullDuplexEnvironmentEvaluator(EvaluatorBase[Tick]):
    """
    Evaluator focuses on endstate of the simulation environment.
    """

    @classmethod
    def ticks_to_message_history(cls, ticks: list[Tick]) -> list[Message]:
        """
        Convert a list of Ticks to a message history suitable for Environment.set_state().

        The order follows the execution order in FullDuplexOrchestrator:
        - User tool calls are processed before agent tool calls within each tick
        - Each tool call message is followed by its corresponding tool results

        Args:
            ticks: List of Tick objects from full-duplex simulation.

        Returns:
            List of Messages in the format expected by Environment.set_state():
            [UserMessage with tool_calls, ToolMessage results, AssistantMessage with tool_calls, ToolMessage results, ...]
        """
        messages: list[Message] = []

        for tick in ticks:
            # 1. User tool calls first (processed before agent in orchestrator)
            if tick.user_tool_calls:
                user_msg = UserMessage(
                    role="user",
                    content=tick.user_chunk.content if tick.user_chunk else None,
                    tool_calls=tick.user_tool_calls,
                    timestamp=(
                        tick.user_chunk.timestamp if tick.user_chunk else tick.timestamp
                    ),
                    contains_speech=(
                        tick.user_chunk.contains_speech if tick.user_chunk else False
                    ),
                )
                messages.append(user_msg)
                messages.extend(tick.user_tool_results)

            # 2. Agent tool calls second
            if tick.agent_tool_calls:
                agent_msg = AssistantMessage(
                    role="assistant",
                    content=tick.agent_chunk.content if tick.agent_chunk else None,
                    tool_calls=tick.agent_tool_calls,
                    timestamp=(
                        tick.agent_chunk.timestamp
                        if tick.agent_chunk
                        else tick.timestamp
                    ),
                    contains_speech=(
                        tick.agent_chunk.contains_speech if tick.agent_chunk else False
                    ),
                )
                messages.append(agent_msg)
                messages.extend(tick.agent_tool_results)

        return messages

    @classmethod
    def calculate_reward(
        cls,
        environment_constructor: Callable[[], Environment],
        task: Task,
        full_trajectory: list[Tick],
        solo_mode: bool = False,
        env_kwargs: dict = None,
        strict_replay: bool = True,
    ) -> RewardInfo:
        """
        Calculate the reward for the simulation.
        Args:
            environment_constructor: Callable[[], Environment]
            task: Task
            full_trajectory: list[Tick]
            solo_mode: bool
            env_kwargs: dict
            strict_replay: forwarded to Environment.set_state(strict=...). Set
                False when re-grading historical trajectories whose recorded
                tool outputs may cosmetically differ from current tool code.
        Returns:
            RewardInfo
        """
        predicted_message_history = cls.ticks_to_message_history(full_trajectory)
        return _env_evaluate_common(
            environment_constructor=environment_constructor,
            task=task,
            predicted_message_history=predicted_message_history,
            solo_mode=solo_mode,
            env_kwargs=env_kwargs,
            strict_replay=strict_replay,
        )
