import logging

from tqdm.auto import tqdm
from langchain_core.runnables import Runnable
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq


_logger = logging.getLogger(__name__)


def get_prompt_generator(model_name: str, *, temperature: float = 0.0) -> Runnable:
    """Get the prompt generator chain"""
    _logger.debug("Creating prompt generator for model %s", model_name)
    output_parser = StrOutputParser()
    model = ChatGroq(name=model_name, temperature=temperature)  # type: ignore
    prompt_gen_prompt = PromptTemplate(
        template="{query}", input_variables=["query"])
    prompt_generator_chain = prompt_gen_prompt | model | output_parser
    return prompt_generator_chain


def get_prompts(chain_query_generator: Runnable, queries: list[str], categories: list[str]) -> list[list[str]]:
    """Get the prompts for the queries and categories."""
    _logger.debug("Generating prompts for categories")
    prompts = []
    for category in tqdm(categories):
        prompts.append(
            [chain_query_generator.invoke({"query": query.format(category=category)})
             for query in tqdm(queries, leave=False)])
    return prompts
