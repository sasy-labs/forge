import langroid as lr
import langroid.language_models as lm
from typing import Literal, Callable, Optional
from pydantic import BaseModel
from langroid.utils.constants import NO_ANSWER
from malade.utils.formatting import format_list, snake_case
from malade.doc.fda_handler import get_fda_handler
from malade.doc.mock_fda_handler import get_mock_fda_handler
from malade.utils.constants import DONE
from malade.utils.task import Task
from malade.tools.passing import handle_with_pass_to,override_fallback_recipient_tool
from malade.tools.handling import handle_with
from malade.utils.pydantic import load, save, _resolve_path
from malade.critic_agent import (
    FeedbackTool,
    FinalAnswerTool,
    CriticConfig,
    Critic,
)
from fire import Fire
import logging
from malade.omop import omop_drugs, conditions, printing_name
from malade.omop_interactions import Interactions
from malade.drug_categories import get_drugs_by_category

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

lr.agent.special.doc_chat_agent.apply_nest_asyncio()

omop_drug_values: list[str] = list(map(lambda v: v.value, omop_drugs)) # type: ignore
condition_values = list(map(lambda v: v.value, conditions))
omop_conditions = conditions

class CatEffectOutputs(BaseModel):
    label: Literal["increase", "decrease", "no-effect"]
    confidence: float
    probability: float
    frequency: Literal["none", "rare", "common"]
    evidence: Literal["none", "weak", "strong"]
    justification: str

    @classmethod
    def default(cls):
        return cls(
            label="no-effect",
            confidence=0,
            probability=0,
            frequency="none",
            evidence="none",
            justification="",
        )

    def __str__(self):
        return f"""
        Answer: {self.label}

        Justification: {self.justification}
        """

class DrugCategoryConditionInteractionsAblation(BaseModel):
    raw_outputs: str = ""
    effect_computed: bool = False
    initial_effect_computed: bool = False
    initial_effect: CatEffectOutputs = CatEffectOutputs.default()
    effect: CatEffectOutputs = CatEffectOutputs.default()

class DrugCategoryInteractionsAblation(BaseModel):
    conditions: dict[str, DrugCategoryConditionInteractionsAblation] = {}
    initial_conditions: dict[str, DrugCategoryConditionInteractionsAblation] = {}
    initial_conditions_no_rag: dict[str, DrugCategoryConditionInteractionsAblation] = {}
    conditions_no_rag: dict[str, DrugCategoryConditionInteractionsAblation] = {}
    representative_drugs: list[str] = []

class InteractionsAblation(BaseModel):
    categories: dict[str, DrugCategoryInteractionsAblation] = {}

def to_new_version(cache_path="outputs/interactions.json", out_path="outputs/interactions_ablation.json"):
    interactions = load(Interactions, cache_path)()
    output = InteractionsAblation(
        categories={
            dc: DrugCategoryInteractionsAblation(
                representative_drugs=cat_inters.representative_drugs,
                conditions={
                    condition: DrugCategoryConditionInteractionsAblation(
                        raw_outputs=inters.raw_outputs,
                        effect_computed=True,
                        effect=CatEffectOutputs(
                            label=inters.net_effect,
                            confidence=inters.confidence,
                            probability=inters.probability,
                            frequency=inters.frequency,
                            evidence=inters.evidence,
                            justification=inters.net_effect_justification,
                        )
                    )
                    for condition, inters in cat_inters.conditions.items()
                }
            )
            for dc, cat_inters in interactions.categories.items()
        }
    )
    save(InteractionsAblation, out_path)(output)

def gen_drug_condition_interaction_task(
        name: str,
        use_fn_api: bool,
        llm: lr.language_models.OpenAIGPTConfig,
        handler: Callable[[str, str, int], None],
        embed: str,
        rag: bool=True,
        computed_final: bool=True,
        max_attempts: int = 5,
) -> Task:
    orchestrator_agent = lr.ChatAgent(
        lr.ChatAgentConfig(
            llm=llm,
            system_message=f"""
            You will recieve questions involving medical data.  You
            are experienced in general medical reasoning, but must
            consult references for any specific medical knowledge
            required to answer my questions.

            You have access to `FDAHandler`, who will answer questions
            you ask about specific drugs using FDA data. You must use
            the `recipient_message` tool/function to ask these
            questions, and the `intended_recipient` MUST be
            `FDAHandler` anytime you use this tool. Ensure that
            you ask FDAHandler for the specific information
            you need.

            As some potential complications are listed in FDA labels
            as lacking a verified causal relationship, make certain
            that your final answer expresses the degree of reliability
            of your answer. Similarly, make sure to clearly express
            the degree of risk associated (i.e. is the condition a
            rare or a common side effect, or does a drug rarely or
            frequently result in reduced risk of a condition).

            If FDAHandler cannot answer your question then your answer
            should be {NO_ANSWER}, because the FDA label data does not
            specify the answer. If FDAHandler answers with {NO_ANSWER}
            that means that the FDA label for the drug does not
            contain the information requested (and, in particular, it
            means that it does not mention the condition); hence, your
            answer should be {NO_ANSWER}. This indicates that there
            may not be any effect on the risk of the condition, make
            sure to explain this in your justification.

            IMPORTANT: if multiple attempts fail to retrieve any
            relevant information, there is no need to continue asking
            questions to FDAHandler; assume that the information is
            not in the FDA labels and so FDAHandler cannot answer.

            You MUST specifically tell the critic why you could not
            find an answer to the question; be sure to specify that
            the FDAHandler answered with {NO_ANSWER} if that is the
            reason.

            You must provide your final answer with the `final_answer`
            tool/function; make sure to clearly state my question, the
            reasoning used to derive the answer, including the
            questions asked to FDAHandler and a summary of the
            results, as well as your final answer in the `answer`
            field. `answer` must include any relevant details provided
            by FDAAgent, it is not enough to express these in `steps`.
            """,
        )
    )
    orchestrator_agent.enable_message(FinalAnswerTool)
    handle_with_pass_to(orchestrator_agent, FinalAnswerTool, "Critic")

    orchestrator_agent.enable_message(override_fallback_recipient_tool(
        f"""
        Do one of the following:

        1. If you intend to ask a question to FDAAgent:
            Please use the 'add_recipient' tool/function-call
            and specify FDAAgent as your `intended_recipient`.
            DO NOT REPEAT your original message; ONLY specify the 
            `intended_recipient` via this tool/function-call.
        2. If you are ready to have the critic check your answer,
            do so with the `final_answer` tool/function, as I have requested.

        DO NOT simply acknowldedge these instructions. If you have nothing
        else to say, you MUST submit your answer to the critic; if the critic
        does not approve your answer, submit {NO_ANSWER} as your answer, explaining
        that you could not answer with sufficient certainty.
        """
    ))
    orchestrator_agent.enable_message(FeedbackTool, handle=True, use=False)

    first_attempt = None
    attempts = 0

    def handle(msg: FeedbackTool) -> str:
        nonlocal first_attempt, attempts
        attempts += 1
        answer = getattr(msg, "answer")

        if first_attempt is None:
            first_attempt = answer

        if msg.valid or computed_final or attempts >= max_attempts:
            handler(first_attempt, answer, attempts)
            return DONE

        return msg.handle()

    handle_with(orchestrator_agent, FeedbackTool, handle)

    task = Task(
        orchestrator_agent,
        name= f"DOIA-{snake_case(name)}",
        interactive=False,
        erase_substeps=True,
    )

    critic_agent = Critic(
        CriticConfig(
            user_message=f"""
            You are also experienced in medical reasoning, and
            have general medical knowledge. Unless the responses
            are inconsistent with your medical (or common-sense)
            knowledge, you generally trust responses from
            FDAHandler.

            The answer should express the strength of evidence for
            the answer and the magnitude of the effect. If the
            user states that FDAAgent does not have this information,
            you should accept it.

            If the answer given contains {NO_ANSWER}, accept it as 
            long as the answer clearly expresses why it was not
            possible to answer the question. If it states that
            this is because FDAHandler responded with {NO_ANSWER},
            you should accept it as sufficient justification.
            Otherwise, ask the user to express whether FDAHandler
            responded with {NO_ANSWER}, and, if not, to state
            why it was not possible to answer the question. If
            it does so, the answer is acceptable and the other
            requirements need not be enforced.
            """,
            llm=llm,
            use_functions_api=use_fn_api,
            use_tools=not use_fn_api,
        )
    )
    critic_task = Task(critic_agent, name="Critic", interactive=False, recipient="AGENT")

    if rag:
        doc_task = get_fda_handler(
            use_fn_api=use_fn_api,
            use_tools_api=not use_fn_api,
            llm=llm,
            embed=embed,
            recipient="LLM",
        )
    else:
        doc_task = get_mock_fda_handler(llm=llm, recipient="LLM")

    task.add_sub_task([doc_task, critic_task])

    return task


class CategoryEffectTool(FinalAnswerTool):
    request: str = "category_effect_tool"
    purpose: str = "To express the association of a category of drugs with a condition and the reasoning steps used to derive it"

    answer: CatEffectOutputs

def category_effect_task(
    use_fn_api: bool,
    llm: lr.language_models.OpenAIGPTConfig,
    condition: str,
    cat_name: str,
    cat_name_printing: str,
    condition_name_printing: str,
    handler: Callable[[CatEffectOutputs, CatEffectOutputs, int], None],
    suffix: Optional[str] = None,
    computed_final: bool = True,
    max_attempts: int = 5,
) -> Task:
    omop_label_agent = lr.ChatAgent(
        lr.ChatAgentConfig(
            llm=llm,
            system_message=f"""
            You are experienced in general medical reasoning and have
            general medical knowledge.

            You will be provided a list of passages answering, for
            each of a set of drugs X, whether drug X increases or
            decreases the risk of {condition}. They all belong to
            category {cat_name}.

            You must provide your final answer with the `category_effect_tool`
            tool/function; make sure to clearly state my question, the
            reasoning used to derive the answer, including the
            evidence from the passages, as well as your final answer
            in the `answer` field.

            In your `answer`, make sure that `label`, is one of the
            following: "increase," "decrease," or "no-effect," and
            make sure to include your justification.

            Your `justification` must clearly express the magnitude of
            risk indicated and the strength of evidence. Provide a
            `confidence` value between 0 and 1 indicating the
            confidence in your assigned `label` and a `probability`
            value indicating the probability that the drug will
            cause the condition (or prevent the condition) in a
            given patient.

            Express the frequency that the drug has an effect as
            either "none," "rare," or "common" with the `frequency`
            field and express the strength of `evidence` as either
            "strong" (for example, evidence is strong when shown in a
            clinical trial) or "weak" (for example, this applies to
            purely correlational evidence) or "none" if no evidence
            exists.
            """,
        ),
    )
    omop_label_agent.enable_message(CategoryEffectTool)
    handle_with_pass_to(omop_label_agent, CategoryEffectTool, "Critic")
    omop_label_agent.enable_message(FeedbackTool, handle=True, use=False)

    first_attempt = None
    attempts = 0

    def handle(msg: FeedbackTool) -> str:
        nonlocal first_attempt, attempts
        attempts += 1
        answer = getattr(msg, "answer")

        if first_attempt is None:
            first_attempt = answer

        if msg.valid or computed_final or attempts >= max_attempts:
            handler(first_attempt, answer, attempts)
            return DONE

        return msg.handle()

    handle_with(omop_label_agent, FeedbackTool, handle)

    critic_agent = Critic(
        CriticConfig(
            user_message="""
            You are also experienced in medical reasoning, and have
            general medical knowledge. Unless the responses are
            inconsistent with your medical (or common-sense)
            knowledge, you generally trust responses from
            FDAHandler. Similarly, you trust that the user's
            statements about passages are correct without the need to
            review them directly.

            The answer provided should indicate an increase, decrease,
            or no effect on the risk, and must be no effect if no
            evidence linking the drug category to the risk of the
            condition exists.

            The answer should be drawn from the specified passages,
            hence, the absence of information related to a condition
            in the FDA data for all drugs in a category should be
            enough to conclude that there is no effect for that drug.

            The answer should express the degree of certainty and the
            magnitude of change in risk, ensure that the provided
            answer is consistent with the evidence.
            """,
            criticized=CategoryEffectTool,
            llm=llm,
            use_functions_api=use_fn_api,
            use_tools=not use_fn_api,
        )
    )

    name = f"CORA-{snake_case(condition_name_printing)}-{snake_case(cat_name_printing)}"
    if suffix is not None:
        name = f"{name}-{snake_case(suffix)}"

    omop_label_task = Task(
        omop_label_agent,
        interactive=False,
        name=name,
    )
    omop_label_task.add_sub_task(Task(
        critic_agent,
        interactive=False,
        name="Critic",
    ))

    return omop_label_task


def get_omop_interactions(
    local: bool = False,
    local_model: str = "ollama/llama3:70b",
    local_embed: str = "BAAI/bge-large-en-v1.5",
    strong_llm: Literal["azure", "openai"] = "azure",
    interactions_path: str = "outputs/interactions_ablation.json",
    recompute_interactions: bool = False,
    recompute_labels: bool = False,
) -> None:
    if local:
        llm = lr.language_models.OpenAIGPTConfig(
            chat_model=local_model,
            chat_context_length=128000,
        )
        embed = local_embed
    else:
        embed = "openai"
        if strong_llm == "azure":
            llm=lm.AzureConfig()
        else:
            llm = lm.OpenAIGPTConfig()

    use_fn_api = not local

    load_interactions = load(InteractionsAblation, interactions_path)
    save_interactions = save(InteractionsAblation, interactions_path)

    if _resolve_path(interactions_path).exists():
        interactions: InteractionsAblation = load_interactions()
    else:
        interactions = InteractionsAblation()

    categories = get_drugs_by_category(categories=omop_drug_values, llm=llm)
    for cat in omop_drugs:
        cat_name_printing = printing_name(cat)
        cat = cat.value
        cat_name = str(cat)
        representative_drugs_in_cat = categories.summaries[cat_name].representative_drugs

        if cat_name not in interactions.categories:
            interactions.categories[cat_name] = DrugCategoryInteractionsAblation(
                representative_drugs=representative_drugs_in_cat,
            )

        cat_interactions = interactions.categories[cat_name]

        def gen_interaction_inputs(msg: str) -> list[str]:
            return [f"Does {drug} {msg}?" for drug in representative_drugs_in_cat]

        def merge_outputs(outputs: list[str]) -> str:
            return format_list(
                [
                    f"Drug {drug}: {answer}"
                    for drug, answer in zip(representative_drugs_in_cat, outputs)
                ],
                numbered=True,
            )

        for cond in omop_conditions:
            condition_printing_name = printing_name(cond)
            condition = cond.value
            init_outputs = {}
            final_outputs = {} 
            no_rag_outputs = {} 
            init_no_rag_outputs = {} 

            def gen_drug_condition_task(i: int, rag=True) -> Task:
                if rag:
                    computed_final = condition in cat_interactions.conditions and not recompute_interactions
                else:
                    computed_final = condition in cat_interactions.conditions_no_rag and not recompute_interactions
                    
                def handler(first_answer, final_answer, _):
                    if rag:
                        init_outputs[i] = first_answer
                        if computed_final:
                            final_outputs[i] = cat_interactions.conditions[condition].raw_outputs
                        else:
                            final_outputs[i] = final_answer
                    else:
                        init_no_rag_outputs[i] = first_answer
                        if computed_final:
                            no_rag_outputs[i] = cat_interactions.conditions_no_rag[condition].raw_outputs
                        else:
                            no_rag_outputs[i] = final_answer

                return gen_drug_condition_interaction_task(
                    llm=llm,
                    embed=embed,
                    handler=handler,
                    name=f"{condition_printing_name}-{representative_drugs_in_cat[i]}",
                    use_fn_api=use_fn_api,
                    computed_final=computed_final,
                    rag=rag,
                )

            if condition not in cat_interactions.conditions or condition not in cat_interactions.initial_conditions or recompute_interactions:
                logger.info(f"Retrieving FDA label information related to {cat_name} and {condition}.")

                for i, inp in enumerate(
                    gen_interaction_inputs(f"increase or decrease the risk of {condition}")
                ):
                    gen_drug_condition_task(i).run(inp)

                outputs = merge_outputs([
                    final_outputs[i]
                    for i in range(len(representative_drugs_in_cat))
                ])
                initial_outputs = merge_outputs([
                    init_outputs[i] for i in range(len(representative_drugs_in_cat))
                ])

                if condition in cat_interactions.conditions:
                    cat_interactions.conditions[condition].raw_outputs = outputs
                else:
                    cat_interactions.conditions[condition] = DrugCategoryConditionInteractionsAblation(
                        raw_outputs=outputs,
                    )

                if condition in cat_interactions.initial_conditions:
                    cat_interactions.initial_conditions[condition].raw_outputs = initial_outputs
                else:
                    cat_interactions.initial_conditions[condition] = DrugCategoryConditionInteractionsAblation(
                        raw_outputs=initial_outputs,
                    )
                save_interactions(interactions)

            if condition not in cat_interactions.conditions_no_rag or condition not in cat_interactions.initial_conditions_no_rag or recompute_interactions:
                logger.info(f"Substituting FDA label retrieval with LLM: {cat_name} and {condition}.")

                for i, inp in enumerate(
                    gen_interaction_inputs(f"increase or decrease the risk of {condition}")
                ):
                    gen_drug_condition_task(i, rag=False).run(inp)
                
                outputs = merge_outputs([
                    no_rag_outputs[i]
                    for i in range(len(representative_drugs_in_cat))
                ])
                initial_outputs = merge_outputs([
                    init_no_rag_outputs[i] for i in range(len(representative_drugs_in_cat))
                ])

                if condition in cat_interactions.conditions_no_rag:
                    cat_interactions.conditions_no_rag[condition].raw_outputs = outputs
                else:
                    cat_interactions.conditions_no_rag[condition] = DrugCategoryConditionInteractionsAblation(
                        raw_outputs=outputs,
                    )

                if condition in cat_interactions.initial_conditions_no_rag:
                    cat_interactions.initial_conditions_no_rag[condition].raw_outputs = initial_outputs
                else:
                    cat_interactions.initial_conditions_no_rag[condition] = DrugCategoryConditionInteractionsAblation(
                        raw_outputs=initial_outputs,
                    )

                save_interactions(interactions)

        for interactions_critic, rag, conditions in zip(
                [True, False, True, False],
                [True, True, False, False],
                [
                    cat_interactions.conditions,
                    cat_interactions.initial_conditions,
                    cat_interactions.conditions_no_rag,
                    cat_interactions.initial_conditions_no_rag,
                ],
        ):
            to_update = omop_conditions if recompute_labels else [
                c
                for c in omop_conditions
                if not conditions[c.value].effect_computed
                or not conditions[c.value].initial_effect_computed
            ]

            def gen_output_task(i: int):
                cond = to_update[i]
                condition_printing_name = printing_name(cond)
                condition = cond.value
                computed_final = condition in conditions and conditions[condition].effect_computed and not recompute_labels
                def handler(first_attempt: CatEffectOutputs, answer: CatEffectOutputs, _) -> None:
                    conditions[condition].initial_effect = first_attempt
                    conditions[condition].initial_effect_computed = True

                    if not computed_final:
                        conditions[condition].effect = answer
                        conditions[condition].effect_computed = True
                    save_interactions(interactions)

                suffixes = []
                if not interactions_critic:
                    suffixes.append("nc")
                if not rag:
                    suffixes.append("nrag")
                
                suffix = "-".join(suffixes) if len(suffixes) > 0 else None

                return category_effect_task(
                    use_fn_api=use_fn_api,
                    llm=llm,
                    cat_name=cat_name,
                    cat_name_printing=cat_name_printing,
                    condition=condition,
                    condition_name_printing=condition_printing_name,
                    handler=handler,
                    computed_final=computed_final,
                    suffix=suffix,
                )

            def gen_inputs(i: int):
                condition = to_update[i].value
                outputs = conditions[condition].raw_outputs
                return f"""
                    Passages:
                    {outputs}
                    ---------
                    Does the {cat_name} category of drugs increase the risk of
                    {condition}, decrease it, or is there no clear effect?
                """

            for i in range(len(to_update)):
                gen_output_task(i).run(gen_inputs(i))


if __name__ == "__main__":
    Fire(get_omop_interactions)

    
    
