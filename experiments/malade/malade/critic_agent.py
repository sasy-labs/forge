import langroid as lr
from langroid.utils.constants import PASS
from malade.utils.constants import DONE
from malade.utils.formatting import format_list
from malade.tools.handling import handle_with
from typing import Optional, Any
from pydantic import ConfigDict

class FinalAnswerTool(lr.ToolMessage):
    request: str = "final_answer"
    purpose: str = "To express your answer to a question and the reasoning steps used to derive it"
    question: str
    steps: list[str]
    answer: str

    def handle(self) -> str:
        reasoning_steps = format_list(self.steps, numbered=True)
            
        reasoning = f"""
        Question: {self.question}
        -----
        Reasoning:
        {reasoning_steps}
        -----
        Final answer: {self.answer}
        """

        return reasoning 

class FeedbackTool(lr.ToolMessage):
    request: str = "feedback"
    purpose: str = "To express your feedback of the reasoning process"
    critique: str
    valid: bool

    model_config = ConfigDict(
        arbitrary_types_allowed=False,
        validate_assignment=True,
        json_schema_extra={"exclude": {"purpose", "id"}},
        extra="allow"
    )

    def handle(self) -> str:
        if self.valid:
            validity_statement = "Your reasoning is valid, submit your answer."
        else:
            validity_statement = "Please try again; make sure to correct these flaws in your reasoning."

        return f"""
        Feedback: {self.critique}

        {validity_statement}
        """

class CriticConfig(lr.ChatAgentConfig):
    system_message: str = """
    You are an expert in logical reasoning whose task is to act as a
    critic, who identifies any flaws in the user's reasoning.

    {user_message}

    First, you will take time to think and determine the flaws in the
    reasoning; explain your thought process. Once you are done, you
    will use the `feedback` tool/function to respond with your
    criticism, if any. Do not begin composing the `feedback`
    message until you are certain whether or not you want the
    user to try again due to flawed reasoning.

    If the reasoning provided is correct and/or you cannot identify
    any specific flaws, or in general when the user should not be
    required to make another attempt, `valid` must be true.
    """
    user_message: str = ""
    answer_attribute: str = "answer"
    criticized: type[lr.ToolMessage] = FinalAnswerTool
    max_attempts: Optional[int] = 5

class Critic(lr.ChatAgent):
    def __init__(self, config: CriticConfig):
        super().__init__(config)
        self.system_message = config.system_message.format(
            user_message=config.user_message,
        )
        self.user_message = None
        self.enable_message(
            config.criticized,
            handle=True,
            use=False,
        )
        self.enable_message(FeedbackTool, handle=True, use=True)
        self.answer: Any = None
        self.attempts: int  = 0
        self.max_attempts = config.max_attempts

        def handle_attempt(msg: config.criticized) -> Optional[lr.ChatDocument | str]:
            self.attempts += 1
            self.answer = getattr(msg, config.answer_attribute)
            return msg.handle()

        handle_with(self, config.criticized, handle_attempt)

    def handle_message_fallback(
        self, msg: str | lr.ChatDocument
    ) -> str | lr.ChatDocument | None:
        if isinstance(msg, lr.ChatDocument) and msg.metadata.sender == lr.Entity.LLM:
            return """
            You MUST use the `feedback` tool/function in your next
            message.  If possible, send the critique you intended for
            the user's provided reasoning. If that is not possible or
            if you would simply acknowledge this message, you must
            instead use the `feedback` tool with `critique` set to ""
            and mark the message as "valid".

            You must ALWAYS specify that the field `request` is
            `feedback` in your tool/function call. The fact that this
            is the default is NOT enough.

            Be sure to submit the actual JSON formatted request, not
            the schema, and be sure to specifically state that you are
            using the `feedback` tool, this MUST be in the `request` field.
            """

    def feedback(self, msg: FeedbackTool) -> str:
        setattr(msg, "answer", self.answer)

        if self.max_attempts is not None and self.attempts >= self.max_attempts:
            msg.critique = "No feedback provided."
            msg.valid = True

        return DONE + " " + PASS

