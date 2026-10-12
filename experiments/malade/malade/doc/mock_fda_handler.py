from rich.prompt import Prompt
import langroid as lr
import langroid.language_models as lm
from malade.utils.task import Task
from malade.utils.constants import DONE
from typing import Optional
from langroid.utils.constants import NO_ANSWER

def get_mock_fda_handler(
        llm: lm.OpenAIGPTConfig=lm.OpenAIGPTConfig(),
        recipient: Optional[str] = "LLM",
) -> Task:
    config = lr.ChatAgentConfig(
        system_message=f"""
        You will try your best to answer my questions, in this order
        of preference:
        
        If given a question asking about a drug "X and Y", this is a
        combination drug, so your answer should be for "X and Y" not
        "X" or "Y".

        If asked a question about drugs in broad category, make to
        consider your knowledge of EVERY drug in the category, and in
        particular, if the question asks for which drugs in the
        category something is true, make CERTAIN that your answer
        correctly lists ALL drugs in the category where the condition
        holds.
        
        ANSWER FORMAT:
        
        ALWAYS present your answer in one of the below 2 formats:
        
        1. In case you DO NOT know the answer:
        
        {DONE} {NO_ANSWER}  
        
        2. In case you DO know the answer: 
        
        {DONE}
        ANSWER: [Your concise answer, with a brief summary of
                 necessary context.  ALWAYS clarify the level of
                 certainty of the information.  If applicable, ALWAYS
                 express the magnitude of any increase or decrease in
                 risk and any associated information.]
        """,
        llm=llm,
    )

    task = Task(
        lr.ChatAgent(config),
        name="FDAHandler",
        interactive=False,
        single_round=True,
        recipient=recipient,
    )
    return task

if __name__ == "__main__":
    task = get_mock_fda_handler()
    while True:
        query = Prompt.ask("[blue]How can I help?")
        if query in ["x", "q"]:
            break
        task.run(query)
