import langroid as lr
from langroid.agent.base import NO_ANSWER
import langroid.language_models as lm
from malade.doc.fda_handler import get_fda_handler
from malade.utils.formatting import format_list
from malade.utils.task import Task
from malade.tools.handling import handle_with
from malade.utils.constants import DONE
from pydantic import BaseModel
from malade.tools.passing import override_fallback_recipient_tool

llm = lm.OpenAIGPTConfig(
    chat_model="ollama/llama3:70b",
    chat_context_length=128000,
)
embed = "BAAI/bge-large-en-v1.5"
use_fn_api = False
override_fda_handler_llm = True

if override_fda_handler_llm:
    fda_handler_llm = lm.AzureConfig()
    embed = "openai"
else:
    fda_handler_llm = llm

def to_str(langroid_response: str | lr.ChatDocument | None) -> str:
    if isinstance(langroid_response, str):
        return langroid_response
    elif isinstance(langroid_response, lr.ChatDocument):
        return langroid_response.content
    else:
        return NO_ANSWER

def typed_agent_response(prompt: str, output_type: type, name: str):
    base_agent = lr.ChatAgent(
        lr.ChatAgentConfig(
            name=name,
            system_message=prompt,
            llm=llm,
            use_functions_api=use_fn_api,
            use_tools=not use_fn_api,
        )
    )
    class Response(lr.ToolMessage):
        request: str = "response"
        purpose: str = "To submit your response."
        response: output_type

    answer: output_type = None

    def handle(msg: Response) -> str:
        nonlocal answer
        answer = msg.response

        return DONE

    base_agent.enable_message(Response, use=True, handle=True, force=True)
    handle_with(base_agent, Response, handle)

    base_agent.enable_message(override_fallback_recipient_tool(
        """
        You MUST respond with your answer using `response` tool/function.
        """
    ))

    task = Task(
        base_agent,
        name=name,
        interactive=False,
    )

    task.run()
    return answer
            
def get_prompt(
        question: str,
        qa_pairs: list[tuple[str, str]],
        attempt_feedback_pairs: list[tuple[str, str]],
        instructions: str,
        most_recent_attempt_only: bool=False,
) -> str:
    formatted_qa_pairs = format_list(
        [
            f"""
            Question: {q}
            Response from FDAHandler: {a}
            """
            for q, a in qa_pairs
        ],
        numbered=True,
    )
    formatted_attempts = format_list(
        [
            f"""
            Attempt: {a.answer}
            Justification: {a.justification}
            Feedback from critic: {f}
            """
            for a, f in (attempt_feedback_pairs[-1:] if most_recent_attempt_only else attempt_feedback_pairs)
        ],
        numbered=True,
    )

    qa_prompt = "" if len(qa_pairs) == 0 else f"""
    You have received the following answers for past questions
    posed:

    {formatted_qa_pairs}
    """

    attempts_prompt = "" if len(attempt_feedback_pairs) == 0 else f"""
    You have attempted to submit the following answers, but the critic
    did not accept them, and provided the attached feedback.

    {formatted_attempts}

    Take this feedback into account when considering your response.
    """

    return f"""
    You are experienced in general medical reasoning, but must
    consult references for any specific medical knowledge
    required to answer my questions.

    You wish to answer the following question: {question}

    You have access to an assistant, FDAHandler, who finds
    answers, if possible, to your questions using FDA label
    data, and signals failure with an asnwer of {NO_ANSWER}.

    {qa_prompt}

    {attempts_prompt}

    {instructions}
    """
    
def compose_question(
        question: str,
        qa_pairs: list[tuple[str, str]],
        attempt_feedback_pairs: list[tuple[str, str]],
        most_recent_attempt_only: bool=False,
) -> str:
    prompt = get_prompt(
        question=question,
        qa_pairs=qa_pairs,
        attempt_feedback_pairs=attempt_feedback_pairs,
        most_recent_attempt_only=most_recent_attempt_only,
        instructions="""
        To help you answer the original question, you need to ask
        FDAHandler for more information. Submit your question to
        FDAHandler using the `response` tool/function.
        """
    )

    return typed_agent_response(prompt, str, "ComposeQuestion")

def query_fda(query: str) -> str:
    return to_str(
        get_fda_handler(
            llm=fda_handler_llm,
            embed=embed,
            collection_name="malade_localembed" if embed != "openai" else "malade",
        ).run(query)
    )

def can_answer(
        question: str,
        qa_pairs: list[tuple[str, str]],
        attempt_feedback_pairs: list[tuple[str, str]],
        most_recent_attempt_only: bool=False,
        force_query: bool = True,
) -> bool:
    if len(qa_pairs) == 0 and force_query:
        return False

    prompt = get_prompt(
        question=question,
        qa_pairs=qa_pairs,
        attempt_feedback_pairs=attempt_feedback_pairs,
        most_recent_attempt_only=most_recent_attempt_only,
        instructions=f"""
        Do you have enough information to submit your{' revised' if len(attempt_feedback_pairs) > 0 else ''}
        answer? Submit your answer using the `response` tool/function.
        """
    )
    return typed_agent_response(prompt, bool, "CanAnswer")

def compose_answer(
        question: str,
        qa_pairs: list[tuple[str, str]],
        attempt_feedback_pairs: list[tuple[str, str]],
        most_recent_attempt_only: bool=False,
) -> str:
    prompt = get_prompt(
        question=question,
        qa_pairs=qa_pairs,
        attempt_feedback_pairs=attempt_feedback_pairs,
        most_recent_attempt_only=most_recent_attempt_only,
        instructions=f"""
        You need to compose your{' revised' if len(attempt_feedback_pairs) > 0 else ''}
        answer. Submit your answer using the `response` tool/function together with
        justification.

        Your answer provided in `response` must include justification, and any relevant details provided
        by FDAAgent. It is NOT enough to express this separately.
        """
    )
    return typed_agent_response(prompt, str, "ComposeAnswer")

class Feedback(BaseModel):
    critique: str
    valid: bool

def critic(
        question: str,
        answer: str,
        qa_pairs: list[tuple[str, str]],
) -> Feedback:
    formatted_qa_pairs = format_list(
        [
            f"""
            Question: {q}
            Response from FDAHandler: {a}
            """
            for q, a in qa_pairs
        ],
        numbered=True,
    )

    prompt = f"""
    You are an expert in logical reasoning whose task is to act as a
    critic, who identifies any flaws in the user's reasoning. You are
    also experienced in medical reasoning, and have general medical
    knowledge. 

    Question: {question}

    The user's answer: {answer}

    The user has the following evidence from FDAHandler, expressed as
    a series of questions and answers. Unless the responses are
    inconsistent with your medical (or common-sense) knowledge, you
    generally trust responses from FDAHandler.

    {formatted_qa_pairs}

    The answer should express the strength of evidence for
    the answer and the magnitude of the effect. If the
    user states that FDAAgent does not have this information,
    you should accept it.

    The answer must furthermore contain a justification and
    relevant information (for example, it should specify the
    strength of evidence and summarize the key related information
    from the FDAHandler responses).

    If the answer given contains {NO_ANSWER}, accept it as 
    long as the responses from FDAHandler do not contain
    enough information to answer the question, for
    instance if FDAHandler responded with {NO_ANSWER}.

    First, you will take time to think and determine the flaws in the
    reasoning; explain your thought process. Once you are done, you
    will use the `response` tool/function to respond with your
    criticism, if any. Do not begin composing the `response`
    message until you are certain whether or not you want the
    user to try again due to flawed reasoning.

    If the reasoning provided is correct and/or you cannot identify
    any specific flaws, or in general when the user should not be
    required to make another attempt, `valid` must be true.
    """

    return typed_agent_response(prompt, Feedback, "Critic")


def drug_outcome_info_agent(drug: str, condition: str, max_iters: int = 5) -> str:
    qa_pairs, attempt_feedback_pairs = [], []
    question = f"Does {drug} increase or decrease the risk of {condition}?"

    for _ in range(max_iters):
        while not can_answer(question, qa_pairs, attempt_feedback_pairs):
            query = compose_question(question, qa_pairs, attempt_feedback_pairs)
            response = query_fda(query)

            qa_pairs.append((query, response))

        attempt = compose_answer(question, qa_pairs, attempt_feedback_pairs)
        feedback = critic(question, attempt, qa_pairs)
        valid, critique = feedback.valid, feedback.critique

        if not valid:
            attempt_feedback_pairs.append((attempt, critique))
        else:
            return attempt

    return attempt_feedback_pairs[-1][0]


if __name__ == "__main__":
    drug_outcome_info_agent("Lisinopril", "Angioedema")
