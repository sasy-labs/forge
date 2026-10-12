"""
Updated from `omop_variance.py` to support observability and the reference monitor.
Key changes: importing `malade.utils.configure_sasy` to instrument and
recording the IDs of the messages which produce the output of each agent in the
pipeline; these are then used to record the dependencies outside of the framework
 code via `sasy.instrumentation.langroid.record_message()`.
"""
# Models are selected explicitly below. Langroid otherwise queries OpenAI's
# model-list endpoint during import whenever a key is present, even for Azure
# runs; a stalled discovery call prevents both the parent and spawned workers
# from starting. Suppress that lookup only during import, then restore the key.
import os

_openai_discovery_key = os.environ.pop("OPENAI_API_KEY", None)
try:
    import langroid as lr
finally:
    if _openai_discovery_key is not None:
        os.environ["OPENAI_API_KEY"] = _openai_discovery_key
    del _openai_discovery_key

import logging
import time
from pathlib import Path

from dotenv import load_dotenv
from sasy.instrumentation.langroid import record_message, recorded_version_id

from malade.drug_categories import *
from malade.omop import *
from malade.omop_ablation import *
from malade.omop_variance import get_representative_trials
from malade.utils.pydantic import _resolve_path

_instrumented = False

def _setup_instrumentation():
    """Set up observability and runtime policy enforcement for MALADE."""
    global _instrumented
    if not _instrumented:
        _instrumented = True
        import malade.utils.configure_sasy  # noqa: F401

lr.utils.configuration.settings.cache = int(os.environ.get("MALADE_DEMO_PAUSE", "0")) > 0
logger = logging.getLogger(__name__)

class DrugCategoryOutput(BaseModel):
    representative_drugs: list[str]
    justification: str

    def __str__(self):
        return f"""
        The representative drugs are {format_list(self.representative_drugs)}

        Justification: {self.justification}
        """


def _make_constrained_submit_answer(
    candidates: list[str],
) -> type[FinalAnswerTool]:
    """Build a per-task SubmitAnswer whose ``representative_drugs`` accepts
    only names drawn from *candidates* (exact, case-preserved).

    Pydantic's Literal validation is the authoritative gate: invalid
    names fail parsing before reaching any handler.  The ``candidates``
    class attribute (a frozenset) is exposed for tests.

    Raises ``ValueError`` if *candidates* is empty — the caller must
    fail closed rather than fall back to the unconstrained SubmitAnswer.
    """
    if not candidates:
        raise ValueError("candidates must be non-empty for constrained submission")

    from typing import Literal as L

    literal_type = L[tuple(candidates)]  # type: ignore[valid-type]

    class _ConstrainedDrugCategoryOutput(BaseModel):
        representative_drugs: list[literal_type]  # type: ignore[valid-type]
        justification: str

        def __str__(self):
            return f"""
            The representative drugs are {format_list(self.representative_drugs)}

            Justification: {self.justification}
            """

    _frozen = frozenset(candidates)

    class _ConstrainedSubmitAnswer(FinalAnswerTool):
        request: str = "submit_answer"
        purpose: str = (
            "To express your answer, the representative drugs with justification, "
            "and the reasoning steps to derive it."
        )
        answer: _ConstrainedDrugCategoryOutput

    _ConstrainedSubmitAnswer.candidates = _frozen  # type: ignore[attr-defined]

    return _ConstrainedSubmitAnswer


class SubmitAnswer(FinalAnswerTool):
    request: str = "submit_answer"
    purpose: str = "To express your answer, the representative drugs with justification, and the reasoning steps to derive it."
    answer: DrugCategoryOutput


AZURE_STRUCTURED_OUTPUT_MIN = "2024-08-01-preview"
AZURE_STRUCTURED_OUTPUT_DEFAULT = "2024-10-21"

# Azure models known to support structured outputs.  Restricted to
# the GPT-4.1 family actually used by MALADE; opaque gpt-4o aliases
# may resolve to older snapshots that lack support.
_AZURE_STRUCTURED_OUTPUT_MODELS = frozenset({
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
})


def _patch_agent_strict_support(agent: lr.ChatAgent) -> None:
    """Enable strict JSON / strict tools on *agent* when the provider
    supports structured outputs but the installed Langroid version
    mis-detects it.

    The installed ``AzureGPT.__init__`` recalculates
    ``supports_json_schema`` from a hardcoded allowlist that omits the
    ``2024-10-21`` GA version, ignoring the config flag.  This function
    patches the **model instance** after construction so that
    ``_json_schema_available()`` and ``_strict_tools_available()``
    return True.  The override is scoped to models in
    ``_AZURE_STRUCTURED_OUTPUT_MODELS`` on ``api_version >=
    2024-08-01-preview``; legacy models at a modern API version are
    not patched.
    """
    from langroid.language_models.azure_openai import AzureGPT

    llm = agent.llm
    if llm is None:
        return
    if not isinstance(llm, AzureGPT):
        return
    if llm.supports_json_schema:
        return
    if llm.config.api_version < AZURE_STRUCTURED_OUTPUT_MIN:
        return
    model = llm.config.chat_model
    if model not in _AZURE_STRUCTURED_OUTPUT_MODELS:
        return
    llm.supports_json_schema = True
    llm.supports_strict_tools = True


def get_representative_drug_extraction_task(
    category,
    # Updated for observability: add one additional argument (the output node ID) to the callback
    handler: Callable[[str, str, int, str], None],
    use_fn_api: bool,
    llm: lr.language_models.OpenAIGPTConfig,
    candidates: Optional[list[str]] = None,
    max_attempts: int = 5,
    suffix: Optional[str] = None,
    computed_final: bool = False,
):
    cat_str = str(category.value)

    if candidates is not None:
        submit_tool = _make_constrained_submit_answer(candidates)
        candidate_names = format_list(candidates)
        candidate_constraint = (
            f"\n        The ONLY permitted drug names are: {candidate_names}. "
            "Your answer must use names from this exact list."
        )
    else:
        submit_tool = SubmitAnswer
        candidate_constraint = ""

    representative_drug_extraction_prompt = f"""
        You are a helpful assistant with general medical and
        pharmacological knowledge.  I will provide you with a list
        of drugs, and the result of a query on a medical database
        with their usage rates; your goal is to find N
        representative drugs in category {cat_str} out of the provided
        drugs.

        Prefer generic names if possible, and do not include both
        a brand and generic name for the same drug in your list.

        If possible, prefer drugs with different active
        ingredients (i.e. avoid derivatives of a drug already in
        the list), keeping your choices to the most basic variant
        of a given drug from the list (use the total prescription
        rate of variants of the same base drug to select the top
        drugs); disregard this if you cannot find N with this
        restriction. If fewer than N meet the conditions, you may
        include fewer than N (but never more).

        The names of the selected representatives must EXACTLY
        match one of the provided drugs; choose the names from
        the original list, not the database query.{candidate_constraint}

        You must provide your final answer with the `submit_answer`
        tool/function; make sure to clearly state my question, as
        well as the reasoning used to derive the answer. Include
        the requirements on your answer in the `question`
        field.
        """

    representative_drug_agent = lr.ChatAgent(
        lr.ChatAgentConfig(
            llm=llm,
            use_functions_api=use_fn_api,
            use_tools=not use_fn_api,
            system_message=representative_drug_extraction_prompt,
        )
    )
    _patch_agent_strict_support(representative_drug_agent)
    representative_drug_agent.enable_message(submit_tool)
    representative_drug_agent.enable_message(FeedbackTool, use=False)

    first_attempt = None
    attempts = 0

    def handle(msg: FeedbackTool, chat_doc: lr.ChatDocument) -> str:
        nonlocal first_attempt, attempts
        attempts += 1
        # The critic keeps the answer it reviewed. The SASY adapter hands each
        # responder a detached copy of its input, so a value the critic sets on
        # the tool object does not reach this handler.
        answer = critic.answer

        if first_attempt is None:
            first_attempt = answer

        if msg.valid or computed_final or attempts >= max_attempts:
            # Record ID of final message with answer for dependencies
            handler(first_attempt, answer, attempts, recorded_version_id(chat_doc))
            return DONE

        return msg.handle()

    handle_with(representative_drug_agent, FeedbackTool, handle)

    critic = Critic(
        CriticConfig(
            llm=llm,
            use_functions_api=use_fn_api,
            use_tools=not use_fn_api,
            criticized=submit_tool,
            user_message=f"""
            You are also an expert in medical and pharmacological
            reasoning.

            Your goal is to ensure that the selected drugs are
            distinct members of the category {cat_str} of drugs.
            You will consider information provided directly
            to the user to be reliable (for example, this
            might include prescription rates and a complete
            list of drugs in category {cat_str}). Unless this
            contradicts your pharmacological knowledge,
            the user's choices of representatives for a
            category are acceptable unless they do not
            represent the basic form of a given drug.
            """
        )
    )
    name = f"RD-{snake_case(printing_name(category))}"
    if suffix is not None:
        name = f"{name}-{suffix}"
    name = name[:64] # OpenAI restriction

    representative_drug_task = Task(
        representative_drug_agent,
        name=name,
        interactive=False,
        erase_substeps=True,
    )
    representative_drug_task.add_sub_task(Task(
        critic,
        interactive=False,
        name="Critic",
    ))
    handle_with_pass_to(representative_drug_agent, submit_tool, "Critic")

    # When strict JSON is available, set the constrained submit_tool as
    # the agent's output_format from the outset.  This makes every API
    # call include a response_format with strict=True and the
    # enumerated candidate names, so the model outputs valid JSON
    # directly rather than relying on Langroid's AnyTool recovery
    # wrapper (which nests the tool in a ``tool_or_function`` envelope
    # that doesn't route through pass_to/critic).
    #
    # Unsupported providers hit a BadRequestError on the first call;
    # Langroid catches this, clears output_format, disables strict
    # mode, and retries — falling back to Pydantic Literal validation.
    if representative_drug_agent._json_schema_available():
        representative_drug_agent.set_output_format(
            submit_tool,
            force_tools=True,
            use=True,
            handle=True,
            instructions=True,
        )

    return representative_drug_task


def gen_drug_condition_interaction_task(
        name: str,
        use_fn_api: bool,
        llm: lr.language_models.OpenAIGPTConfig,
        # Updated for observability: add one additional argument (the output node ID) to the callback
        handler: Callable[[str, str, int, str], None],
        embed: str,
        rag: bool=True,
        computed_final: bool=True,
        max_attempts: int = 5,
        nl_policy: bool = False,
        collection_name: str = "malade",
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

    def handle(msg: FeedbackTool, chat_doc: lr.ChatDocument) -> str:
        nonlocal first_attempt, attempts
        attempts += 1
        # The critic keeps the answer it reviewed. The SASY adapter hands each
        # responder a detached copy of its input, so a value the critic sets on
        # the tool object does not reach this handler.
        answer = critic_agent.answer

        if first_attempt is None:
            first_attempt = answer

        if msg.valid or computed_final or attempts >= max_attempts:
            handler(first_attempt, answer, attempts, recorded_version_id(chat_doc))
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
            nl_policy=nl_policy,
            collection_name=collection_name,
        )
    else:
        doc_task = get_mock_fda_handler(llm=llm, recipient="LLM")

    task.add_sub_task([doc_task, critic_task])

    return task


def category_effect_task(
    use_fn_api: bool,
    llm: lr.language_models.OpenAIGPTConfig,
    condition: str,
    cat_name: str,
    cat_name_printing: str,
    condition_name_printing: str,
    # Not changed, as this is the final output of the system, and nothing
    # depends on it
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
        # The critic keeps the answer it reviewed. The SASY adapter hands each
        # responder a detached copy of its input, so a value the critic sets on
        # the tool object does not reach this handler.
        answer = critic_agent.answer

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


def _sum_task_tokens(task: lr.Task) -> tuple[int, float]:
    """Recursively sum token usage and cost across a task and all sub-tasks."""
    tokens = task.agent.total_llm_token_usage
    cost = task.agent.total_llm_token_cost
    for sub in task.sub_tasks:
        st, sc = _sum_task_tokens(sub)
        tokens += st
        cost += sc
    return tokens, cost


class TrialResults(BaseModel):
    representative_drugs_in_cat: list[str]
    representative_drugs_attempts: int
    interactions_summary: str
    drug_condition_interaction_agent_attempt_counts: list[int]
    # A task may stop without producing a label; retain it as an unsuccessful trial.
    prediction: Optional[CatEffectOutputs] = None
    category_outcome_risk_agent_attempt_count: int
    total_tokens: int = 0
    total_cost: float = 0.0
    runtime_seconds: float = 0.0
    fda_violations: int = 0
    fda_search_calls: int = 0
    fda_blocked_calls: int = 0
    # Count of FDA search-tool invocations made before the agent
    # registered. Distinct from ``fda_violations`` (which only counts
    # cases where the API actually returned non-empty data) — this
    # covers every attempt regardless of whether the API responded
    # with content. The right metric for "how often did the
    # non-instrumented agent ignore the prompt's register-first
    # instruction".
    fda_pre_reg_attempts: int = 0
    correct: bool = False


def _collect_fda_stats(task: lr.Task) -> tuple[int, int, int, int]:
    """Sum (violations, search_calls, blocked_calls,
    pre_reg_attempts) across a task tree.
    """
    from malade.doc.fda_handler import SearchDocChatAgent
    violations = 0
    search_calls = 0
    blocked_calls = 0
    pre_reg_attempts = 0
    if isinstance(task.agent, SearchDocChatAgent):
        violations += task.agent.fda_violations
        search_calls += task.agent.fda_search_calls
        blocked_calls += task.agent.fda_blocked_calls
        pre_reg_attempts += getattr(task.agent, "fda_pre_reg_attempts", 0)
    for sub in task.sub_tasks:
        sv, ss, sb, sp = _collect_fda_stats(sub)
        violations += sv
        search_calls += ss
        blocked_calls += sb
        pre_reg_attempts += sp
    return violations, search_calls, blocked_calls, pre_reg_attempts
    
def run_trial(
    category: OMOPDrugs,
    condition: OMOPConditions,
    use_fn_api: bool,
    db: str,
    llm: lr.language_models.OpenAIGPTConfig,
    embed: str,
    trial: int=0,
    rag: bool=True,
    fix_outputs: Literal["none", "representatives", "interactions"] = "none",
    critic_representatives: bool=True,
    critic_interactions: bool=True,
    critic_labels: bool=True,
    max_attempts: int=5,
    instrumented: bool=True,
    nl_policy: bool=False,
) -> TrialResults:
    # Scope every recorded event and every policy query for this
    # trial to its own session id. Without this all trials share
    # the global partition on the SASY server, which slows queries
    # (the rebuild pulls in unrelated state) and makes per-trial
    # debugging harder.
    #
    # Default to a fresh UUID per call so reruns of the same trial
    # number in the same process don't share a session bucket
    # (which would mix old and new state on the server). Opt into
    # the readable {cat}-{cond}-trial-{n} pattern by exporting
    # SASY_SESSION_READABLE=1 — useful when you want to grep logs
    # for a specific trial across runs and accept the collision
    # risk.
    import os
    from pathlib import Path

    import sasy
    session_id = (
        f"malade-{category.value}-{condition.value}-trial-{trial}"
        if os.environ.get("SASY_SESSION_READABLE") == "1"
        else None
    )
    # Pin each trial to the MALADE policy via the session-scoped
    # API. The server dedupes by content hash so all trials in this
    # run share one registry entry. Override with
    # SASY_POLICY_PATH=<file>.
    policy = (
        Path(os.environ["SASY_POLICY_PATH"])
        if os.environ.get("SASY_POLICY_PATH")
        else Path(__file__).resolve().parent.parent / "malade_policy.dl"
    )
    if not policy.exists():
        policy = None  # fall through to engine default
    with sasy.session(session_id, policy=policy):
        return _run_trial_inner(
            category=category, condition=condition, use_fn_api=use_fn_api,
            db=db, llm=llm, embed=embed, trial=trial, rag=rag,
            fix_outputs=fix_outputs,
            critic_representatives=critic_representatives,
            critic_interactions=critic_interactions,
            critic_labels=critic_labels,
            max_attempts=max_attempts,
            instrumented=instrumented, nl_policy=nl_policy,
        )


def _run_trial_inner(
    category: OMOPDrugs,
    condition: OMOPConditions,
    use_fn_api: bool,
    db: str,
    llm: lr.language_models.OpenAIGPTConfig,
    embed: str,
    trial: int=0,
    rag: bool=True,
    fix_outputs: Literal["none", "representatives", "interactions"] = "none",
    critic_representatives: bool=True,
    critic_interactions: bool=True,
    critic_labels: bool=True,
    max_attempts: int=5,
    instrumented: bool=True,
    nl_policy: bool=False,
) -> TrialResults:
    t0 = time.monotonic()
    representative_drugs_in_cat = []
    representative_drugs_attempts = 0
    # Track the ID of the message which produced the representative drugs for observability
    representatives_out_id = None
    # Collect all tasks for token tracking
    _all_tasks: list[lr.Task] = []


    if fix_outputs == "none":
        def representative_drug_handler(_, final_answer, attempts, output_node_id):
            nonlocal representative_drugs_attempts, representative_drugs_in_cat, representatives_out_id
            representative_drugs_in_cat = final_answer.representative_drugs
            representative_drugs_attempts = attempts
            representatives_out_id = output_node_id

        candidates = sorted(set(get_drugs_of_class(str(category.value))))
        if not candidates:
            return TrialResults(
                representative_drugs_in_cat=[],
                representative_drugs_attempts=0,
                interactions_summary="",
                drug_condition_interaction_agent_attempt_counts=[],
                prediction=None,
                category_outcome_risk_agent_attempt_count=0,
                runtime_seconds=time.monotonic() - t0,
            )

        outputs = get_drug_query_outputs(db=db).categories[str(category.value)]
        nc_notice = "nc-" if not critic_representatives else ""
        rd_task = get_representative_drug_extraction_task(
            category,
            llm=llm,
            use_fn_api=use_fn_api,
            handler=representative_drug_handler,
            candidates=candidates,
            max_attempts=max_attempts,
            computed_final=not critic_representatives,
            suffix=f"{nc_notice}trial-{trial}"
        )
        rd_task.run(
            f"""
            Out of {format_list(candidates)}, which three are the most common?

            The result of a query on a medical database is below:
            {outputs}
            """
        )
        _all_tasks.append(rd_task)

        if int(os.environ.get("MALADE_DEMO_PAUSE", "0")) > 0:
            print("[DEMO] Representative Drug Extraction complete — press Enter to continue")
            input()
    else:
        representative_drugs_in_cat = get_drugs_by_category(
           categories=omop_drug_values,
           llm=llm,
        ).summaries[str(category.value)].representative_drugs
    
    # Track the ID of the messages with the extracted interaction information for observability
    dcia_final_outputs = {}
    dcia_attempts = {}
    output_node_ids = {}

    interaction_tasks_complete = True
    if fix_outputs in ["none", "representatives"]:
        def gen_dcia_task(i: int) -> Task:
            def handler(_, final_answer, attempts, output_node_id):
                dcia_final_outputs[i] = final_answer
                dcia_attempts[i] = attempts
                output_node_ids[i] = output_node_id
            nc_notice = "-nc" if not critic_interactions else ""
            nrag_notice = "-nrag" if not rag else ""
            return gen_drug_condition_interaction_task(
                llm=llm,
                embed=embed,
                use_fn_api=use_fn_api,
                handler=handler,
                name=f"{printing_name(condition)}-{representative_drugs_in_cat[i]}{nc_notice}{nrag_notice}-trial-{trial}",
                max_attempts=max_attempts,
                rag=rag,
                computed_final=not critic_interactions,
                nl_policy=nl_policy,
                collection_name=f"malade-{'instr' if instrumented else 'noinstr'}-trial-{trial}-{i}",
            )

        def merge_outputs(outputs: list[str]) -> str:
            return format_list(
                [
                    f"Drug {drug}: {answer}"
                    for drug, answer in zip(representative_drugs_in_cat, outputs)
                ],
                numbered=True,
            )

        for i, inp in enumerate([
            f"Does {drug} increase or decrease the risk of {condition.value}?"
            for drug in representative_drugs_in_cat
        ]):
            # Represent input as a chat document and record the dependencies on the drug extraction
            inp_chat_doc = lr.ChatDocument(
              content=inp,
              metadata=lr.ChatDocMetaData(
                 sender=lr.Entity.USER,
              )
            )
            # The question is composed from the representatives answer, so
            # it is recorded as computed from that answer's recorded version.
            if _instrumented:
                record_message(inp_chat_doc, input_ids=[representatives_out_id])

            dcia_task_i = gen_dcia_task(i)
            dcia_task_i.run(inp_chat_doc)
            _all_tasks.append(dcia_task_i)

        # A stopped subtask is an unsuccessful trial, not a missing-key
        # exception that causes the whole trial to be resampled.
        interaction_tasks_complete = bool(representative_drugs_in_cat) and all(
            i in dcia_final_outputs for i in range(len(representative_drugs_in_cat))
        )
        drug_interaction_data = merge_outputs([
            dcia_final_outputs.get(i, NO_ANSWER)
            for i in range(len(representative_drugs_in_cat))
        ])

        if int(os.environ.get("MALADE_DEMO_PAUSE", "0")) > 0:
            print("[DEMO] Drug-Condition Interaction complete — press Enter to continue")
            input()
    else:
       drug_interaction_data = load(Interactions, "outputs/interactions.json")().categories[
          str(category.value)
      ].conditions[
         str(condition.value)
      ].raw_outputs

    attempts = 0
    output = None
    
    def handler(_, final_answer, attempt_count):
        nonlocal attempts, output
        output = final_answer
        attempts = attempt_count

    nc_notice = "nc-" if not critic_labels else ""

    # Use a ChatDocument rather than a string to attach a unique ID
    chat_document = lr.ChatDocument(
        content=
        f"""
            Passages:
            {drug_interaction_data}
            ---------
            Does the {category.value} category of drugs increase the risk of
            {condition.value}, decrease it, or is there no clear effect?
        """,
        metadata=lr.ChatDocMetaData(
            sender=lr.Entity.USER,
        )
    )

    # The passages are the interaction answers, so the document is recorded
    # as computed from their recorded versions.
    if _instrumented:
        record_message(chat_document, input_ids=list(output_node_ids.values()))

    ce_task = category_effect_task(
        llm=llm,
        use_fn_api=use_fn_api,
        handler=handler,
        condition=condition.value,
        cat_name=str(category.value),
        cat_name_printing=printing_name(category),
        condition_name_printing=printing_name(condition),
        computed_final=not critic_labels,
        suffix=f"{nc_notice}trial-{trial}"
    )
    if interaction_tasks_complete:
        ce_task.run(chat_document)
        _all_tasks.append(ce_task)
    
    runtime_seconds = time.monotonic() - t0

    total_tokens = 0
    total_cost = 0.0
    fda_violations = 0
    fda_search_calls = 0
    fda_blocked_calls = 0
    fda_pre_reg_attempts = 0
    for t in _all_tasks:
        tt, tc = _sum_task_tokens(t)
        total_tokens += tt
        total_cost += tc
        tv, ts, tb, tp = _collect_fda_stats(t)
        fda_violations += tv
        fda_search_calls += ts
        fda_blocked_calls += tb
        fda_pre_reg_attempts += tp

    # Compute accuracy against ground truth
    correct = False
    if output is not None:
        ground_truth = omop_table.get(category, {}).get(condition, OMOPLabels.NO_EFFECT)
        gt_label = ground_truth.value
        if gt_label == "no-effect-verified":
            gt_label = "no-effect"
        correct = output.label == gt_label

    return TrialResults(
        representative_drugs_in_cat=representative_drugs_in_cat,
        representative_drugs_attempts=representative_drugs_attempts,
        interactions_summary=(
            drug_interaction_data.content
            if isinstance(drug_interaction_data, lr.ChatDocument)
            else drug_interaction_data
        ),
        drug_condition_interaction_agent_attempt_counts=list(dcia_attempts.values()),
        prediction=output, # type: ignore
        category_outcome_risk_agent_attempt_count=attempts,
        total_tokens=total_tokens,
        total_cost=total_cost,
        runtime_seconds=runtime_seconds,
        fda_violations=fda_violations,
        fda_search_calls=fda_search_calls,
        fda_blocked_calls=fda_blocked_calls,
        fda_pre_reg_attempts=fda_pre_reg_attempts,
        correct=correct,
    )

class VarianceTrialResults(BaseModel):
    representatives_by_ground_truth: Optional[dict[str, tuple[str, str]]]
    representatives_by_confidence: Optional[dict[str, tuple[str, str]]]
    results: dict[str, dict[str, list[TrialResults]]]


def _malade_trial_worker(
    cat: str,
    cond: str,
    trial_idx: int,
    config: dict,
) -> Optional[TrialResults]:
    """Module-level worker for one (cat, cond, trial) cell.

    Module-level (not a closure) so ``map_concurrent(mode="processes")``
    can pickle it via plain pickle by qualified name. Each process
    has its own Python heap, which means Langroid's ``Task._cache``
    classvar and litellm/OpenAI response caches don't leak across
    concurrent trials. The worker reconstructs its LLM config from
    the plain-dict ``config`` payload.

    Returns the trial's ``TrialResults`` or ``None`` after retries.
    The caller aggregates returned values back into the shared
    results dict in the parent process.
    """
    if config["env_credentials"]:
        repo_root = Path(__file__).resolve().parents[3]
        load_dotenv(repo_root / ".env-credentials", override=True)
    if config["instrumented"]:
        _setup_instrumentation()

    if config["local"]:
        llm = lr.language_models.OpenAIGPTConfig(
            chat_model=config["local_model"],
            chat_context_length=128000,
        )
    else:
        if config["strong_llm"] == "azure":
            llm = lm.AzureConfig(
                chat_model="gpt-4.1",
                api_version=AZURE_STRUCTURED_OUTPUT_DEFAULT,
            )
        else:
            llm = lm.OpenAIGPTConfig(
                chat_model=lr.language_models.OpenAIChatModel.GPT4_1,
            )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            category = OMOPDrugs(cat)
            condition = OMOPConditions(cond)
            trial_result = run_trial(
                category=category,
                condition=condition,
                use_fn_api=config["use_fn_api"],
                llm=llm,
                embed=config["embed"],
                db=config["db"],
                trial=trial_idx,
                rag=config["rag"],
                fix_outputs=config["fix_outputs"],
                critic_representatives=config["critic_representatives"],
                critic_interactions=config["critic_interactions"],
                critic_labels=config["critic_labels"],
                max_attempts=config["max_attempts"],
                instrumented=config["instrumented"],
                nl_policy=config["nl_policy"],
            )
            print(
                f"  Trial {trial_idx} | {cat} x {cond}"
                f" | Time: {trial_result.runtime_seconds:.1f}s"
                f" | Tokens: {trial_result.total_tokens:,}"
                f" | Cost: ${trial_result.total_cost:.4f}"
                f" | FDA searches: {trial_result.fda_search_calls}"
                f" | Blocked: {trial_result.fda_blocked_calls}"
                f" | Violations: {trial_result.fda_violations}"
                f" | PreReg: {trial_result.fda_pre_reg_attempts}"
                f" | Correct: {trial_result.correct}"
            )
            return trial_result
        except Exception as e:
            print(
                f"  Trial {trial_idx} | {cat} x {cond} | "
                f"FAILED (attempt {attempt+1}/{max_retries}): {e}"
            )
            if attempt == max_retries - 1:
                print(
                    f"  Trial {trial_idx} | {cat} x {cond} | "
                    f"SKIPPED after {max_retries} attempts"
                )
    return None


def get_variance_trial_results(
    local: bool = False,
    local_model: str = "ollama/llama3:70b",
    local_embed: str = "BAAI/bge-large-en-v1.5",
    strong_llm: Literal["azure", "openai"] = "azure",
    cache_path: str = "outputs/trials/variance_trial_results.json",
    db: str = "mimiciv",
    trials: int = 1,
    rag: bool=True,
    critic: bool=True,
    critic_representatives: Optional[bool]=None,
    critic_interactions: Optional[bool]=None,
    critic_labels: Optional[bool]=None,
    fix_outputs: Literal["none", "representatives", "interactions"] = "none",
    max_critic_attempts: int = 5,
    representatives: bool = True,
    representative_cache_path: str = "outputs/variance_representatives.json",
    ground_truth_only: bool = True,
    recompute: bool = True,
    instrumented: bool = True,
    nl_policy: Optional[bool] = None,
    env_credentials: bool = False,
    max_concurrency: int = 1,
    concurrency_mode: str = "processes",
) -> VarianceTrialResults:
    if env_credentials:
        # .env-credentials lives at the workspace root
        repo_root = Path(__file__).resolve().parents[3]
        load_dotenv(repo_root / ".env-credentials", override=True)

    if nl_policy is None:
        nl_policy = not instrumented
    if instrumented:
        _setup_instrumentation()

    if critic_representatives is None:
       critic_representatives = critic
    if critic_interactions is None:
       critic_interactions = critic
    if critic_labels is None:
       critic_labels = critic

    if local:
        llm = lr.language_models.OpenAIGPTConfig(
            chat_model=local_model,
            chat_context_length=128000,
        )
        embed = local_embed
    else:
        if strong_llm == "azure":
            embed = "azure-openai"
            llm = lm.AzureConfig(
                chat_model="gpt-4.1",
                api_version=AZURE_STRUCTURED_OUTPUT_DEFAULT,
            )
        else:
            embed = "openai"
            llm = lm.OpenAIGPTConfig(
               chat_model=lr.language_models.OpenAIChatModel.GPT4_1,
            )

    use_fn_api = not local

    load_results = load(VarianceTrialResults, cache_path)
    save_results = save(VarianceTrialResults, cache_path)

    if _resolve_path(cache_path).exists():
        results: VarianceTrialResults = load_results()
    else:
        variance_representatives = get_representative_trials(representative_cache_path)
        results = VarianceTrialResults(
            representatives_by_confidence=variance_representatives.representatives_by_confidence if representatives else None,  
            representatives_by_ground_truth=variance_representatives.representatives_by_ground_truth if representatives else None, 
            results={},
        )
        save_results(results)

    # Build list of (cat, cond) cells to evaluate
    if representatives:
        selected = [tuple(v) for v in (results.representatives_by_ground_truth or {}).values()]
    else:
        selected = [
            (cat, cond)
            for cat in omop_drug_values
            for cond in condition_values
        ]

    # Pre-populate results dict keys so we can check existing trials
    for cat, cond in selected:
        results.results.setdefault(cat, {}).setdefault(cond, [])

    # Bundle worker config into a plain dict so ``map_concurrent``
    # can ship it across to process workers (LLM configs aren't
    # cloudpickle-friendly, but their construction args are).
    worker_config: dict = {
        "use_fn_api": use_fn_api,
        "embed": embed,
        "db": db,
        "rag": rag,
        "fix_outputs": fix_outputs,
        "critic_representatives": critic_representatives,
        "critic_interactions": critic_interactions,
        "critic_labels": critic_labels,
        "max_attempts": max_critic_attempts,
        "instrumented": instrumented,
        "nl_policy": nl_policy,
        "strong_llm": strong_llm,
        "local": local,
        "local_model": local_model,
        "local_embed": local_embed,
        "env_credentials": env_credentials,
    }

    args: list[tuple[tuple, dict]] = []
    for cat, cond in selected:
        existing = 0 if recompute else len(results.results[cat][cond])
        for i in range(existing, trials):
            args.append(((cat, cond, i, worker_config), {}))

    # ``concurrency_mode="processes"`` (default) gives each trial a
    # fresh Python heap. Without it, Langroid's ``Task._cache``
    # classvar and litellm/OpenAI in-process response caches can
    # leak content from one trial into another's LLM context, which
    # we observed in the info-flow example as concurrent benign
    # trials picking up xyz_disclosures phrases from peer attack
    # trials. Threads are still available via ``concurrency_mode``
    # for cases where the workload is provably isolated.
    from malade.utils.parallel import map_concurrent
    if args:
        flat_results = map_concurrent(
            _malade_trial_worker,
            args,
            max_in_flight=max(1, max_concurrency),
            mode=concurrency_mode,
        )
        for (cat, cond, _i, _cfg), trial_result in zip(
            [a[0] for a in args], flat_results
        ):
            if trial_result is not None:
                results.results.setdefault(cat, {}).setdefault(cond, []).append(trial_result)
        save_results(results)

    # Aggregate summary
    all_trials = [
        tr
        for by_cond in results.results.values()
        for trials_list in by_cond.values()
        for tr in trials_list
    ]
    if all_trials:
        n = len(all_trials)
        total_time = sum(tr.runtime_seconds for tr in all_trials)
        total_searches = sum(tr.fda_search_calls for tr in all_trials)
        total_blocked = sum(tr.fda_blocked_calls for tr in all_trials)
        total_violations = sum(tr.fda_violations for tr in all_trials)
        total_correct = sum(1 for tr in all_trials if tr.correct)
        total_cost = sum(tr.total_cost for tr in all_trials)
        print(
            f"\n=== Summary ({n} trials) ==="
            f"\n  Total time: {total_time:.1f}s  Avg: {total_time/n:.1f}s"
            f"\n  Total cost: ${total_cost:.4f}  Avg: ${total_cost/n:.4f}"
            f"\n  FDA searches: {total_searches}  Blocked: {total_blocked}  Violations: {total_violations}"
            f"\n  Accuracy: {total_correct}/{n} ({100*total_correct/n:.1f}%)"
            f"\n  Instrumented: {instrumented}  NL policy: {nl_policy}"
        )

    return results

if __name__ == "__main__":
    Fire(get_variance_trial_results)
