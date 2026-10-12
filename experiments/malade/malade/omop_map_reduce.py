import langroid as lr
import langroid.language_models as lm
from itertools import chain
from collections import defaultdict
from typing import TypeVar, Callable, Iterator,  Iterable, Literal
from fire import Fire

from malade.utils.formatting import format_list
from malade.utils.openfda import get_drugs_of_class
from omop import OMOPConditions, OMOPDrugs, omop_drugs, conditions, printing_name
from malade.omop_ablation import CatEffectOutputs, gen_drug_condition_interaction_task, category_effect_task
from malade.omop_variance import get_drug_query_outputs, get_representative_drug_extraction_task

lr.utils.configuration.settings.cache = True

A, B, C, D, E = [TypeVar(n) for n in ["A", "B", "C", "D", "E"]]

def map_reduce(
        map_fn: Callable[[A, B], Iterable[tuple[C, D]]],
        reduce_fn: Callable[[C, Iterable[D]], Iterable[E]],
) -> Callable[[Iterable[tuple[A, B]]], Iterable[E]]:
    def f(input_tuples: Iterable[tuple[A, B]]) -> Iterable[E]:
        mapped: Iterator[tuple[C, D]] = chain.from_iterable(
            map(lambda tup: map_fn(tup[0], tup[1]), input_tuples)
        )

        groups: defaultdict[C, list[D]] = defaultdict(list)

        for (c, d) in mapped:
            groups[c].append(d)

        return chain.from_iterable(
            map(lambda tup: reduce_fn(tup[0], tup[1]), groups.items())
        )

    return f

def compose(*fns):
    reversed_fns = list(reversed(list(fns)))

    def f(x):
        for fn in reversed_fns:
            x = fn(x)

        return x

    return f

def run_malade_map_reduce(
        local: bool = False,
        local_model: str = "ollama/llama3:70b",
        local_embed: str = "BAAI/bge-large-en-v1.5",
        strong_llm: Literal["azure", "openai"] = "azure",
        db: str = "mimiciv",
        max_critic_attempts: int=5,
        run_all_cells: bool = False, # by default, run just one cell
) -> list[tuple[tuple[OMOPDrugs, OMOPConditions], CatEffectOutputs]]:
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

    # Drug representative function
    def drug_finder(
            category: OMOPDrugs,
            conditions: list[OMOPConditions]
    ) -> list[tuple[OMOPDrugs, tuple[list[OMOPConditions], str]]]:
        drugs = []
        def handler(_, result, *args):
            nonlocal drugs
            drugs = result.representative_drugs

        query_outputs = get_drug_query_outputs(db=db).categories[str(category.value)]
        get_representative_drug_extraction_task(
            category,
            llm=llm,
            use_fn_api=use_fn_api,
            handler=handler,
            max_attempts=max_critic_attempts,
            computed_final=False,
            suffix="mr"
        ).run(
            f"""
            Out of {format_list(list(get_drugs_of_class(str(category.value))))}, which three are the most common?

            The result of a query on a medical database is below:
            {query_outputs}
            """
        )

        return [(category, (conditions, drug)) for drug in drugs]

    # This reduction and the passing of the list of conditions as a value
    # is somewhat convoluted and may be a place where operating on a table
    # would make things cleaner
    def drug_finder_reduce(
            category: OMOPDrugs,
            conditions_drug: Iterable[tuple[list[OMOPConditions], str]],
    ) -> list[tuple[tuple[str, OMOPDrugs], OMOPConditions]]:
        return [
            ((drug, category), cond)
            for (conditions, drug) in conditions_drug
            for cond in conditions
        ]

    # Core category, condition interaction functions
    def malade_map(
            drug_details: tuple[str, OMOPDrugs],
            condition: OMOPConditions
    ) -> list[tuple[tuple[OMOPDrugs, OMOPConditions], tuple[str, str]]]:
        drug, category = drug_details

        output: str = ""
        def handler(_, result, *args):
            nonlocal output
            output = result
        
        gen_drug_condition_interaction_task(
            llm=llm,
            embed=embed,
            use_fn_api=use_fn_api,
            handler=handler,
            name=f"{printing_name(condition)}-{drug}-mr",
            max_attempts=max_critic_attempts,
            rag=True,
            computed_final=False,
        ).run(
            f"Drug {drug} increase or decrease the risk of {condition.value}?"
        )

        return [((category, condition), (drug, output))]

    def malade_reduce(
            cell: tuple[OMOPDrugs, OMOPConditions],
            drug_interactions: Iterable[tuple[str, str]],
    ) -> list[tuple[tuple[OMOPDrugs, OMOPConditions], CatEffectOutputs]]:
        category, condition = cell
        drug_interactions_summary = format_list(
            [
                f"Drug {drug}: {output}"
                for drug, output in drug_interactions
            ],
            numbered=True,
        )

        output = None
        def handler(_, result, *args):
            nonlocal output
            output = result

        category_effect_task(
            llm=llm,
            use_fn_api=use_fn_api,
            handler=handler,
            condition=condition.value,
            cat_name=str(category.value),
            cat_name_printing=printing_name(category),
            condition_name_printing=printing_name(condition),
            computed_final=False,
            suffix="mr"
        ).run(
            f"""
                Passages:
                {drug_interactions_summary}
                ---------
                Does the {category.value} category of drugs increase the risk of
                {condition.value}, decrease it, or is there no clear effect?
            """
        )

        return [(cell, output)]

    # Full system 
    malade = compose(
        # Eager evaluation
        list,
        # Core of system
        map_reduce(malade_map, malade_reduce),
        # Identify representative drugs
        map_reduce(drug_finder, drug_finder_reduce), 
    )

    table = [
        (cat, conditions)
        for cat in omop_drugs
    ] if run_all_cells else [(omop_drugs[0], conditions[:1])]

    # Further postprocessing for merging subcategories is dependent on evaluation method, see omop_evaluation.py
    return malade(table)

if __name__ == "__main__":
    Fire(run_malade_map_reduce)
