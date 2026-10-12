from malade.omop import *
from malade.omop_ablation import *
from malade.drug_categories import *
from malade.utils.pydantic import _resolve_path
import numpy as np
from collections import defaultdict

lr.utils.configuration.settings.cache = False

class VarianceRepresentatives(BaseModel):
   representatives_by_confidence: dict[str, tuple[str, str]]
   representatives_by_ground_truth: dict[str, tuple[str, str]]

def get_representative_trials(cache_path: str="outputs/variance_representatives.json") -> VarianceRepresentatives:
    if _resolve_path(cache_path).exists():
        return load(VarianceRepresentatives, cache_path)()

    # Sample cells
    rng = np.random.default_rng()
    
    # Considering only categories without subcategories, sample a no effect, increased risk, and decreased risk cell
    omop_cells_by_ground_truth = defaultdict(list)
    
    for category in omop_drugs:
        if category not in omop_drugs_for_evaluation:
            continue

        for condition in conditions:
            omop_cells_by_ground_truth[omop_table[category][condition]].append((str(category.value), str(condition.value)))
    
    representatives_by_ground_truth = {}
    for k in [OMOPLabels.DECREASE, OMOPLabels.INCREASE, OMOPLabels.NO_EFFECT_VERIFIED]:
        representatives_by_ground_truth[str(k.value)] = tuple(rng.choice(omop_cells_by_ground_truth[k]))
    
    all_representatives = list(representatives_by_ground_truth.values())
    
    # Sample a high, med, and low confidence (in the selected class) cell out of the remaining cells
    def confidence_level(score):
        if score < 0.4:
            return "low"
        elif score < 0.8:
            return "moderate"
        else:
            return "high"
            
    interactions = load(Interactions, "outputs/interactions.json")()
    remaining_omop_cells_by_confidence = defaultdict(list)
    confidences = []
    
    for category in omop_drugs:
        category_interactions = interactions.categories[str(category.value)]
        for condition in conditions:
            prediction = category_interactions.conditions[str(condition.value)]
            confidences.append(prediction.confidence)
            cat_cond_tuple = (str(category.value), str(condition.value))
            if cat_cond_tuple not in all_representatives:
                remaining_omop_cells_by_confidence[confidence_level(prediction.confidence)].append(cat_cond_tuple)
    
    representatives_by_confidence = {}
    for k, v in remaining_omop_cells_by_confidence.items():
        representatives_by_confidence[k] = tuple(rng.choice(v))

    variance_representatives = VarianceRepresentatives(
        representatives_by_confidence=representatives_by_confidence,
        representatives_by_ground_truth=representatives_by_ground_truth,
    )
    save(VarianceRepresentatives, cache_path)(variance_representatives)

    return variance_representatives

class DrugCategoryOutput(BaseModel):
    representative_drugs: list[str]
    justification: str
    
    def __str__(self):
        return f"""
        The representative drugs are {format_list(self.representative_drugs)}

        Justification: {self.justification}
        """
        
class SubmitAnswer(FinalAnswerTool):
    request: str = "submit_answer"
    purpose: str = "To express your answer, the representative drugs with justification, and the reasoning steps to derive it."
    answer: DrugCategoryOutput 


# Updated version of representative drug extraction task that allows control over critic behavior, counting attempts, etc.
def get_representative_drug_extraction_task(
    category,
    handler: Callable[[str, str, int], None],
    use_fn_api: bool,
    llm: lr.language_models.OpenAIGPTConfig,
    max_attempts: int = 5,
    suffix: Optional[str] = None,
    computed_final: bool = False,
):
    cat_str = str(category.value)
    
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
        the original list, not the database query.

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
    representative_drug_agent.enable_message(SubmitAnswer)
    representative_drug_agent.enable_message(FeedbackTool, use=False)

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

    handle_with(representative_drug_agent, FeedbackTool, handle)

    critic = Critic(
        CriticConfig(
            llm=llm,
            use_functions_api=use_fn_api,
            use_tools=not use_fn_api,
            criticized=SubmitAnswer,
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
    handle_with_pass_to(representative_drug_agent, SubmitAnswer, "Critic")
        
    return representative_drug_task

class TrialResults(BaseModel):
    representative_drugs_in_cat: list[str]
    representative_drugs_attempts: int
    interactions_summary: str
    drug_condition_interaction_agent_attempt_counts: list[int]
    prediction: CatEffectOutputs
    category_outcome_risk_agent_attempt_count: int
    
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
) -> TrialResults:
    # Per-trial session scoping. UUID by default; opt into the
    # readable {cat}-{cond}-trial-{n} pattern with
    # SASY_SESSION_READABLE=1 (see omop_observability.run_trial).
    import os
    from pathlib import Path
    import sasy
    session_id = (
        f"malade-variance-{category.value}-{condition.value}-trial-{trial}"
        if os.environ.get("SASY_SESSION_READABLE") == "1"
        else None
    )
    # Pin each trial to the MALADE policy (server-side content
    # dedup means re-uploads across trials are free). Override
    # with SASY_POLICY_PATH=<file>.
    policy = (
        Path(os.environ["SASY_POLICY_PATH"])
        if os.environ.get("SASY_POLICY_PATH")
        else Path(__file__).resolve().parent.parent / "malade_policy.dl"
    )
    if not policy.exists():
        policy = None
    with sasy.session(session_id, policy=policy):
        return _run_trial_inner(
            category=category, condition=condition, use_fn_api=use_fn_api,
            db=db, llm=llm, embed=embed, trial=trial, rag=rag,
            fix_outputs=fix_outputs,
            critic_representatives=critic_representatives,
            critic_interactions=critic_interactions,
            critic_labels=critic_labels,
            max_attempts=max_attempts,
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
) -> TrialResults:
    representative_drugs_in_cat = []
    representative_drugs_attempts = 0

    if fix_outputs == "none":
        def representative_drug_handler(_, final_answer, attempts):
            nonlocal representative_drugs_attempts, representative_drugs_in_cat
            representative_drugs_in_cat = final_answer.representative_drugs
            representative_drugs_attempts = attempts

        outputs = get_drug_query_outputs(db=db).categories[str(category.value)]
        nc_notice = "nc-" if not critic_representatives else ""
        get_representative_drug_extraction_task(
            category,
            llm=llm,
            use_fn_api=use_fn_api,
            handler=representative_drug_handler,
            max_attempts=max_attempts,
            computed_final=not critic_representatives,
            suffix=f"{nc_notice}trial-{trial}"
        ).run(
            f"""
            Out of {format_list(list(get_drugs_of_class(str(category.value))))}, which three are the most common?

            The result of a query on a medical database is below:
            {outputs}
            """
        )
    else:
        representative_drugs_in_cat = get_drugs_by_category(
           categories=omop_drug_values,
           llm=llm,
        ).summaries[str(category.value)].representative_drugs
    
    dcia_final_outputs = {}
    dcia_attempts = {}

    if fix_outputs in ["none", "representatives"]:
        def gen_dcia_task(i: int) -> Task:
            def handler(_, final_answer, attempts):
                dcia_final_outputs[i] = final_answer
                dcia_attempts[i] = attempts
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
            gen_dcia_task(i).run(inp)

        drug_interaction_data = merge_outputs([
            dcia_final_outputs[i]
            for i in range(len(representative_drugs_in_cat))
        ])
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
    category_effect_task(
        llm=llm,
        use_fn_api=use_fn_api,
        handler=handler,
        condition=condition.value,
        cat_name=str(category.value),
        cat_name_printing=printing_name(category),
        condition_name_printing=printing_name(condition),
        computed_final=not critic_labels,
        suffix=f"{nc_notice}trial-{trial}"
    ).run(
        f"""
            Passages:
            {drug_interaction_data}
            ---------
            Does the {category.value} category of drugs increase the risk of
            {condition.value}, decrease it, or is there no clear effect?
        """
    )
    
    return TrialResults(
        representative_drugs_in_cat=representative_drugs_in_cat,
        representative_drugs_attempts=representative_drugs_attempts,
        interactions_summary=drug_interaction_data,
        drug_condition_interaction_agent_attempt_counts=list(dcia_attempts.values()),
        prediction=output, # type: ignore
        category_outcome_risk_agent_attempt_count=attempts,
    )

class VarianceTrialResults(BaseModel):
    representatives_by_ground_truth: Optional[dict[str, tuple[str, str]]]
    representatives_by_confidence: Optional[dict[str, tuple[str, str]]]
    results: dict[str, dict[str, list[TrialResults]]]

def get_variance_trial_results(
    local: bool = False,
    local_model: str = "ollama/llama3:70b",
    local_embed: str = "BAAI/bge-large-en-v1.5",
    strong_llm: Literal["azure", "openai"] = "azure",
    cache_path: str = "outputs/variance_trial_results.json",
    db: str = "mimiciv",
    trials: int = 10,
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
) -> VarianceTrialResults:
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
        embed = "openai"
        if strong_llm == "azure":
            llm=lm.AzureConfig(chat_model="GPT-4.1")
        else:
            llm = lm.OpenAIGPTConfig()

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

    def run_trials(cat: str, cond: str) -> None:
        category = OMOPDrugs(cat)
        condition = OMOPConditions(cond)

        if cat not in results.results:
            results.results[cat] = {}
        if cond not in results.results[cat]:
            results.results[cat][cond] = []
            
        existing_trials = len(results.results[cat][cond])

        for i in range(existing_trials, trials):
            results.results[cat][cond].append(run_trial(
                category=category,
                condition=condition,
                use_fn_api=use_fn_api,
                llm=llm,
                embed=embed,
                db=db,
                trial=i,
                rag=rag,
                fix_outputs=fix_outputs,
                critic_representatives=critic_representatives,
                critic_interactions=critic_interactions,
                critic_labels=critic_labels,
                max_attempts=max_critic_attempts,
            ))
            save_results(results)
    if representatives:
        # selected_representatives = list(
        #     results.representatives_by_ground_truth.values() # type: ignore
        # ) + [] if ground_truth_only else list(
        #     results.representatives_by_confidence.values() # type: ignore
        # )

        selected_representatives = [(OMOPDrugs.AMPHOTERICIN_B.value, OMOPConditions.RENAL_FAILURE.value)]

        for cat_cond in selected_representatives:
            cat, cond = cat_cond
            run_trials(cat, cond)

    else:
        for cat in omop_drug_values:
            for cond in condition_values:
                run_trials(cat, cond)
            
    return results

if __name__ == "__main__":
    Fire(get_variance_trial_results)
