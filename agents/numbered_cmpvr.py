"""Modified version of MPVR with check of queries"""
from typing import TypedDict, Any, Optional, override

import logging

from tqdm import tqdm
from langchain_core.tracers.context import tracing_v2_enabled
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.example_selectors import BaseExampleSelector
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from agents.rate_limiters import get_rate_limiter
from agents.parsers import BasicQueryParser
from agents.callbacks import BatchBarCallback


_logger = logging.getLogger(__name__)
# _logger.setLevel(logging.DEBUG)

type Example = dict[str, str]


GEN_SYSTEM_PROMPT = (
    "You are provided with prompt template examples for a dataset, which are "
    "provided to the LLM to generate descriptions for the categories in these "
    "datasets. Your task is to generate diverse prompts for another dataset "
    "for which you are also provided the dataset name and the description. "
    "Format it correctly following the examples, and do not repeat the "
    "prompts. The variable to include is {{category}}. Do not make any "
    "reference to the dataset name."
    "\n"
    "Make sure to give only the prompts and no additional comment in your "
    "response.\n"
)

GEN_HUMAN_PROMPT = (
    "Dataset Name: {dataset_name}\n"
    "Description: {dataset_description}\n"
    "Make sure that you only give me {amount} prompts.\n"
    "Prompts:"
)

GEN_AI_OUTPUT = "{prompts}"

GEN_EXAMPLES: list[Example] = [
    {
        "dataset_name": "Describable Textures Dataset",
        "dataset_description": (
            "The Describable Textures Dataset is an evolving collection "
            "of textural images in the wild, annotated with a series of "
            "human-centric attributes, inspired by the perceptual properties"
            "of textures."
        ),
        "amount": "10",
        # For some reason this are raw literals by default so no need to escape.
        "prompts": (
            "1. \"Describe the visual characteristics of the texture labeled as {category}.\"\n"
            "2. \"How would you recognize the texture labeled as {category}?\"\n"
            "3. \"What are the key features of the texture labeled as {category}?\"\n"
            "4. \"Provide a detailed description of the appearance of the texture labeled as {category}.\"\n"
            "5. \"If you see an image with the texture labeled as {category}, what would stand out to you?\"\n"
            "6. \"Imagine you encounter a surface with the texture labeled as {category}. How would you describe it?\"\n"
            "7. \"What visual attributes define the texture category {category}?\"\n"
            "8. \"Describe an image featuring the texture labeled as {category}.\"\n"
            "9. \"Create a caption for an image showcasing the texture labeled as {category}:\"\n"
            "10. \"Detail the unique aspects that distinguish the texture labeled as {category} from others.\"\n"
        )
    },
    {
        "dataset_name": "EuroSAT",
        "dataset_description": (
            "This dataset addresses the challenge of land use and land cover "
            "classification openly and freely accessible provided in the Earth "
            "observation program tion using Sentinel-2 satellite images. The "
            "Sentinel-2 satellite images areCopernicus. We present a novel "
            "dataset based on Sentinel-2 satellite images covering 13 spectral "
            "bands and consisting out of 10 classes with in total 27,000 "
            "labeled and geo-referenced images."
        ),
        "amount": "10",
        # For some reason this are raw literals by default so no need to escape.
        "prompts": (
            "1. \"Describe how does the {category} looks like from a satellite.\"\n"
            "2. \"How can you recognize the {category} from a satellite?\"\n"
            "3. \"What does the satellite photo of {category} look like?\"\n"
            "4. \"Describe the satellite photo from the internet of {category}.\"\n"
            "5. \"How can you identify the {category} from a satellite?\"\n"
            "6. \"Explain the geographical features visible in the satellite image of {category}.\"\n"
            "7. \"Highlight the distinguishing characteristics of {category} in satellite imagery.\"\n"
            "8. \"What land cover details are discernible in the satellite snapshot of {category}?\"\n"
            "9. \"Discuss the unique patterns captured in the satellite photo of {category}.\"\n"
            "10. \"Elaborate on the topographical variations showcased in the satellite view of {category}.\"\n"
        )
    }
]

CHECKER_SYSTEM_PROMPT = (
    "As of now, you are a system expert on checking that the given query are"
    "correct in shape and content. There should always be {amount} queries, if "
    "not, remove queries or include new using the context of the other "
    "messages. Every query should be delemited by quotes and include the "
    "variable `{{category}}`. If there is any error you should fix it.\n"
    "\n"
    "You will receive a series of prompts and your answer should be the "
    "corrected version of them. Do not make any additional comments."
)

CHECKER_HUMAN_PROMPT = "Prompts:\n{queries}\nPrompts:"

PROMPT_GEN_SYSTEM_PROMPT = (
    "You receive a question that you must answer in a concise way, without "
    "making comments that are not relevant to the question. The answer you "
    "provide is going to be used as a prompt for a visual recognition system, "
    "so it must be descriptive and condensed."
)


class _QueryVariables(TypedDict):
    dataset_name: str
    dataset_description: str
    amount: int


class _PromptVariables(TypedDict):
    query: str


class NoDTDExampleSelector(BaseExampleSelector):
    """Select the example based if it's the DTD dataset or not."""
    examples: list[Example]

    def __init__(self, examples: list[Example]):
        self.examples = examples

    @override
    def add_example(self, example: Example) -> Any:
        self.examples.append(example)

    @override
    def select_examples(self, input_variables: dict[str, str]) -> list[Example]:
        if input_variables["dataset_name"] == "Describable Textures Dataset":
            return [self.examples[1]]
        return [self.examples[0]]


class NumberedCMPVRAgent:
    """Agent that recreates the original MPVR"""

    def __init__(self, query_model_name: str, prompt_model_name: str,
                 query_model_args: Optional[dict[str, Any]] = None,
                 prompt_model_args: Optional[dict[str, Any]] = None) -> None:
        self.query_model_name = query_model_name
        self.prompt_model_name = prompt_model_name
        self.query_model_args = query_model_args or {}
        self.prompt_model_args = prompt_model_args or {}
        # Set default parameters
        self.query_model_args["temperature"] = self.query_model_args.get(
            "temperature", 0.0)
        self.prompt_model_args["temperature"] = self.prompt_model_args.get(
            "temperature", 0.99)
        self.prompt_model_args["max_tokens"] = self.prompt_model_args.get(
            "max_tokens", 50)
        self.prompt_model_args["max_retries"] = self.prompt_model_args.get(
            "max_retries", 2)
        assert (len(self.prompt_model_args) == 3 and
                len(self.query_model_args) == 1)

    def _get_query_generator(self) -> Runnable[_QueryVariables, list[str]]:
        """Creates the query generator chain of the model and temperature"""
        _logger.debug("Creating query generator for model %s",
                      self.query_model_name)
        rate_limiter = get_rate_limiter(self.query_model_name)
        query_parser = BasicQueryParser()
        output_parser = StrOutputParser()
        examples_template = ChatPromptTemplate.from_messages([
            ("human", GEN_HUMAN_PROMPT),
            ("ai", GEN_AI_OUTPUT)
        ])
        example_selector = NoDTDExampleSelector(examples=GEN_EXAMPLES)
        few_show_template = FewShotChatMessagePromptTemplate(
            input_variables=["dataset_name"],
            example_prompt=examples_template,
            example_selector=example_selector
        )
        query_gen_template = ChatPromptTemplate.from_messages([
            ("system", GEN_SYSTEM_PROMPT),
            few_show_template,
            ("human", GEN_HUMAN_PROMPT)
        ])
        model = ChatGroq(model=self.query_model_name, rate_limiter=rate_limiter,
                         **self.query_model_args)
        unchecked_query_gen_chain = query_gen_template | model | output_parser
        query_gen_checker_template = ChatPromptTemplate.from_messages([
            ("system", CHECKER_SYSTEM_PROMPT),
            ("human", CHECKER_HUMAN_PROMPT)
        ])
        query_gen_chain = (
            {"amount": lambda input_: input_["amount"],
             "queries": unchecked_query_gen_chain}
            | query_gen_checker_template
            | model
            | query_parser
        )
        return query_gen_chain.with_config({"run_name": "Generate queries"})

    def _get_prompt_generator(self) -> Runnable[_PromptVariables, str]:
        """Get the prompt generator chain"""
        _logger.debug("Creating prompt generator for model %s",
                      self.prompt_model_name)
        output_parser = StrOutputParser()
        rate_limiter = get_rate_limiter(self.prompt_model_name)
        model = ChatGroq(name=self.prompt_model_name, rate_limiter=rate_limiter,
                         **self.prompt_model_args)
        prompt_gen_template = ChatPromptTemplate.from_messages([
            ("system", PROMPT_GEN_SYSTEM_PROMPT),
            ("human", "{query}")
        ])
        prompt_gen_chain = prompt_gen_template | model | output_parser
        return prompt_gen_chain.with_retry(  # type: ignore
            stop_after_attempt=self.prompt_model_args["max_retries"])

    def get_queries(self, *, dataset_name: str, dataset_description: str,
                    amount: int) -> list[str]:
        """Obtain the queries for a dataset using a chain."""
        _logger.debug("Generating %i prompts for dataset %s",
                      amount, dataset_name)
        chain = self._get_query_generator()
        with tracing_v2_enabled(project_name="MPVR TFM"):
            queries = chain.invoke({"amount": amount,
                                    "dataset_name": dataset_name,
                                    "dataset_description": dataset_description})
        return queries

    def get_prompts(self, queries: list[str],
                    categories: list[str]) -> list[list[str]]:
        """Get the prompts for the queries and categories."""
        _logger.debug("Generating prompts for categories")
        chain = self._get_prompt_generator()
        prompts: list[list[str]] = []
        for category in tqdm(categories, desc="Categories"):
            with BatchBarCallback(len(queries), desc=f"Queries for {category}",
                                  leave=False) as bar_cb:
                config: RunnableConfig = {"callbacks": [bar_cb],
                                          "run_name": f"Prompt for {category}"}
                prompts.append(
                    chain.batch([{"query": query.format(category=category)}
                                 for query in queries], config=config))
        return prompts
