"""Module with the functionality of the general prompter."""
import logging

from langchain_core.runnables import Runnable
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


from agents.parsers import BasicQueryParser


_logger = logging.getLogger(__name__)


SYSTEM_PROMPT_GENERATOR = """You are provided with prompt template examples for a dataset, which
are provided to the LLM to generate descriptions for the categories in these datasets.
Your task is to generate {amount} diverse prompts for another dataset for which you are 
also provided the dataset name and the description. Format it correctly for use in a Python script,
and do not repeat the prompts.

Make sure to give only the prompts and no additional comment in your response.

the variable to be inyected into each prompt is {{category}}.

Context:
Dataset Name: {dataset_name}

Description: {dataset_description}

Example (delemited by ---):
---
Context:
Dataset Name: Oxford Flowers Dataset
Description: Oxford Flowers consists of 102 flower 
categories. The flowers chosen to be flowers commonly 
occur in the United Kingdom.

Prompts:
    "Describe how does the flower type {{category}} looks like."
    "How can you recognize the flower type {{category}}?"
    "What does the flower type {{category}} look like?"
    "Describe an image from the internet of the flower type {{category}}."
    "How can you identify the flower type of {{category}}?"
---
"""

HUMAN_PROMPT_GENERATOR = """Prompts:
"""

SYSTEM_PROMPT_CHECKER = """You are a system expert on checking that the given text only contains
lines that match the following schema where each sentence is delemited by " and has the variable
{{category}}:

"<text> {{category}} <text><punctuation>"

Where <text> can be any text, even blank text, and <punctuation> should be any punctuation mark.

If any line of text doesn't match the schema, correct it.
Don't give any additional comment.

Good examples:
    "How can you recognize the flower type {{category}}?"
    "What does the flower type {{category}} look like?"
    "Describe how does the flower type {{category}} looks like."
Bad examples
    "{{category}}",
    "How can you recognize the flower type {{category}}"
    "What does the flower type {{category}} look like {{category}}?"
"""

HUMAN_PROMPT_CHECKER = """
Keep only lines that match the schema.
Return only the correct text. Don't give any additional comment in your response

The input text is the following: 
{input}
"""


def get_query_generator(model_name: str, *, temperature: float = 0.0) -> Runnable:
    """Creates the query generator chain of the model and temperature"""
    _logger.debug("Creating query generator for model %s", model_name)
    output_parser = BasicQueryParser()
    model = ChatGroq(model=model_name, temperature=temperature)  # type: ignore
    query_gen_prompt = ChatPromptTemplate.from_messages([  # type: ignore
        ("system", SYSTEM_PROMPT_GENERATOR),
        ("human", HUMAN_PROMPT_GENERATOR)
    ])
    # query_gen_checker_prompt = ChatPromptTemplate.from_messages([  # type: ignore
    #     ("system", SYSTEM_PROMPT_CHECKER),
    #     ("human", HUMAN_PROMPT_CHECKER)
    # ])
    unchecked_query_gen_chain = query_gen_prompt | model | output_parser
    # query_chain_gen = unchecked_query_gen_chain | query_gen_checker_prompt | model | output_parser
    return unchecked_query_gen_chain


def get_queries(query_gen_chain, *, dataset_name: str, dataset_description: str, amount=10):
    """Obtain the queries for a dataset using a chain."""
    _logger.debug("Generating %i prompts for dataset %s", amount, dataset_name)
    return query_gen_chain.invoke({"amount": amount,
                                   "dataset_name": dataset_name,
                                   "dataset_description": dataset_description})
