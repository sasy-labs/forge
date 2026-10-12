import logging

import langroid as lr
from langroid.agent.chat_document import ChatDocument, StatusCode
from langroid.mytypes import Entity
from langroid.utils.constants import USER_QUIT_STRINGS
from sasy.instrumentation.langroid import record_message_update, recorded_version_id
from sasy.instrumentation.session import is_session_active

from malade.utils.constants import DONE

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


Responder = lr.Entity | type["Task"]

# Modified from langroid.agent.Task source
class Task(lr.Task):
    def __init__(
            self,
            agent: lr.Agent | None = None,
            recipient: str | None = None,
            max_cost=10, # catches potential infinite loops
            max_tokens=1_000_000,
            error_on_over_cost=True,
            **kwargs
    ):
        super().__init__(agent=agent, **kwargs)
        self.recipient = recipient # If set, forces a specific recipient for all results
        self.max_cost = max_cost
        self.max_tokens = max_tokens
        self.error_on_over_cost = error_on_over_cost

    def result(self, status: StatusCode | None = None) -> ChatDocument | None:
        """
        Get result of task. This is the default behavior.
        Derived classes can override this.

        Note the result of a task is returned as if it is from the User entity.

        Args:
            status (StatusCode): status of the task when it ended
        Returns:
            ChatDocument: result of task
        """
        # Keep Langroid's current tool-call fields, agent-origin metadata and
        # taint propagation. A copied legacy result constructor loses those
        # fields and newer Langroid then rejects the critic's feedback handoff.
        result_doc = super().result(status)
        if result_doc is None:
            return None
        edited = False
        if self.pending_message is not None and DONE in self.pending_message.content:
            # Langroid strips its `DONE` marker, which would leave `<>` from
            # MALADE's `<DONE>`. Keep the original MALADE text cleanup.
            content = self.pending_message.content.replace(DONE, "").strip()
            edited = result_doc.content != content
            result_doc.content = content
        if self.recipient is not None and result_doc.metadata.recipient != self.recipient:
            result_doc.metadata.recipient = self.recipient
            edited = True
        # The instrumented superclass already recorded the handoff. Cleanup
        # and a forced recipient are edits of that result, not a second result.
        if edited and is_session_active() and recorded_version_id(result_doc) is not None:
            record_message_update(result_doc, input_ids=[], agent=self.agent)

        return result_doc

    def done(
        self, result: ChatDocument | None = None, r: Responder | None = None
    ) -> tuple[bool, StatusCode]:
        """
        Check if task is done. This is the default behavior.
        Derived classes can override this.
        Args:
            result (ChatDocument|None): result from a responder
            r (Responder|None): responder that produced the result
                Not used here, but could be used by derived classes.
        Returns:
            bool: True if task is done, False otherwise
            StatusCode: status code indicating why task is done
        """
        if self._is_kill():
            return (True, StatusCode.KILL)
        result = result or self.pending_message
        user_quit = (
            result is not None
            and (result.content in USER_QUIT_STRINGS or DONE in result.content)
            and result.metadata.sender == Entity.USER
        )
        if self._level == 0 and self.interactive and self.only_user_quits_root:
            # for top-level task, in interactive mode, only user can quit out
            return (user_quit, StatusCode.USER_QUIT if user_quit else StatusCode.OK)

        if self.is_done:
            return (True, StatusCode.DONE)

        if self.n_stalled_steps >= self.max_stalled_steps:
            # we are stuck, so bail to avoid infinite loop
            logger.warning(
                f"Task {self.name} stuck for {self.max_stalled_steps} steps; exiting."
            )
            return (True, StatusCode.STALLED)

        if self.max_cost > 0 and self.agent.llm is not None:
            try:
                if self.agent.llm.tot_tokens_cost()[1] > self.max_cost:
                    logger.warning(
                        f"Task {self.name} cost exceeded {self.max_cost}; exiting."
                    )
                    return (True, StatusCode.MAX_COST)
            except Exception:
                pass

        if self.max_tokens > 0 and self.agent.llm is not None:
            try:
                if self.agent.llm.tot_tokens_cost()[0] > self.max_tokens:
                    logger.warning(
                        f"Task {self.name} uses > {self.max_tokens} tokens; exiting."
                    )
                    return (True, StatusCode.MAX_TOKENS)
            except Exception:
                pass
        final = (
            # no valid response from any entity/agent in current turn
            result is None
            # An entity decided task is done
            or DONE in result.content
            or (  # current task is addressing message to caller task
                self.caller is not None
                and self.caller.name != ""
                and result.metadata.recipient == self.caller.name
            )
            or user_quit
        )
        return (final, StatusCode.OK)

    def _is_done_response(
        self, result: str | None | ChatDocument, responder: Responder
    ) -> bool:
        """Is the task done based on the response from the given responder?"""

        response_says_done = result is not None and (
            (isinstance(result, str) and DONE in result)
            or (isinstance(result, ChatDocument) and DONE in result.content)
        )
        return (
            (
                responder.value in self.done_if_response
                and not self._is_empty_message(result)
            )
            or (
                responder.value in self.done_if_no_response
                and self._is_empty_message(result)
            )
            or (not self._is_empty_message(result) and response_says_done)
        )
