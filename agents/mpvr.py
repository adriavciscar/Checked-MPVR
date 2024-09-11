"""MPVR Agent module"""
from typing import Any, TypedDict, Optional

import logging
import time

from langchain_core.tracers.context import tracing_v2_enabled
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.output_parsers import StrOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_groq import ChatGroq
from tqdm import trange
from tqdm.auto import tqdm

from utils import flatten_list_1d
from agents.rate_limiters import get_rate_limiter
from agents.parsers import PythonQueryParser
from agents.callbacks import BatchBarCallback

_logger = logging.getLogger(__name__)

GENERATOR_PROMPT = (
    "You are provided with prompt template examples for a dataset, which are "
    "provided to the LLM to generate descriptions for the categories in these "
    "datasets. Your task is to generate {amount} diverse prompts for another "
    "dataset for which you are also provided the dataset name and the "
    "description. Format it correctly for use in a Python script, and do not "
    "repeat the prompts.\n"
    "\n"
    "Example\n"
    "Dataset Name: Describable Textures Dataset\n"
    "Description: The Describable Textures Dataset is an evolving "
    "collection of textural images in the wild, annotated with a series of "
    "human-centric attributes, inspired by the perceptual properties of "
    "textures.\n"
    "Prompts:\n"
    "prompts.append(\"How would you recognize the texture labeled as \" + category + \"?\")\n"
    "prompts.append(\"What are the key features of the texture labeled as \" + category + \"?\")\n"
    "prompts.append(\"Provide a detailed description of the appearance of the texture labeled as \" + category + \".\")\n"
    "prompts.append(\"If you see an image with the texture labeled as \" + category + \", what would stand out to you?\")\n"
    "prompts.append(\"Imagine you encounter a surface with the texture labeled as \" + category + \". How would you describe it?\")\n"
    "prompts.append(\"What visual attributes define the texture category \" + category + \"?\")\n"
    "prompts.append(\"Describe an image featuring the texture labeled as \" + category + \".\")\n"
    "prompts.append(\"Create a caption for an image showcasing the texture labeled as \" + category + \":\")\n"
    "prompts.append(\"Detail the unique aspects that distinguish the texture labeled as \" + category + \" from others.\")\n"
    "prompts.append(\"Envision a scenario where you encounter the texture labeled as \" + category + \". How would you articulate its appearance?\")\n"
    "Dataset Name: {dataset_name}\n"
    "Description: {dataset_description}\n"
    "Prompts: \n"
)

# If we put too many examples it starts generating until 2000 queries.

GENERATOR_PROMPT_DTD = (
    "You are provided with prompt template examples for a dataset, which are "
    "provided to the LLM to generate descriptions for the categories in these "
    "datasets. Your task is to generate {amount} diverse prompts for another "
    "dataset for which you are also provided the dataset name and the "
    "description. Format it correctly for use in a Python script, and do not "
    "repeat the prompts.\n"
    "\n"
    "Example\n"
    "Dataset Name: EuroSAT\n"
    "Description: This dataset addresses the challenge of land use and land "
    "cover classification openly and freely accessible provided in the Earth "
    "observation program using Sentinel-2 satellite images. We present a novel "
    "dataset based on Sentinel-2 satellite images covering 13 spectral bands "
    "and consisting out of 10 classes with in total 27,000 labeled and "
    "geo-referenced images.\n"
    "Prompts:\n"
    "prompts.append(\"Describe how does the \" + category + \" looks like from a satellite.\")\n"
    "prompts.append(\"How can you recognize the \" + category + \" from a satellite?\")\n"
    "prompts.append(\"What does the satellite photo of \" + category + \" look like?\")\n"
    "prompts.append(\"Describe the satellite photo from the internet of \" + category + \".\")\n"
    "prompts.append(\"How can you identify the \" + category + \" from a satellite?\")\n"
    "prompts.append(\"Explain the geographical features visible in the satellite image of \" + category + \".\")\n"
    "prompts.append(\"Highlight the distinguishing characteristics of \" + category + \" in satellite imagery.\")\n"
    "prompts.append(\"What land cover details are discernible in the satellite snapshot of \" + category + \"?\")\n"
    "prompts.append(\"Discuss the unique patterns captured in the satellite photo of \" + category + \".\")\n"
    "prompts.append(\"Elaborate on the topographical variations showcased in the satellite view of \" + category + \".\")\n"
    "prompts.append(\"Examine the satellite representation of \" + category + \" and identify notable landmarks.\")\n"
    "\n"
    "Dataset Name: {dataset_name}\n"
    "Description: {dataset_description}\n"
    "Prompts: \n"
)


class _QueryVariables(TypedDict):
    dataset_name: str
    dataset_description: str
    amount: int


class _PromptVariables(TypedDict):
    query: str


class MPVRAgent:
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

    def _get_query_generator(
        self,
        is_dtd: bool = False
    ) -> Runnable[_QueryVariables, list[str]]:
        """Creates the query generator chain of the model and temperature"""
        _logger.debug("Creating query generator for model %s",
                      self.query_model_name)
        query_gen_template = PromptTemplate.from_template(
            GENERATOR_PROMPT if not is_dtd else GENERATOR_PROMPT_DTD)
        rate_limiter = get_rate_limiter(self.query_model_name)
        output_parser = PythonQueryParser()
        query_gen_model = ChatGroq(model=self.query_model_name,
                                   rate_limiter=rate_limiter,
                                   **self.query_model_args)
        query_gen_chain = query_gen_template | query_gen_model | output_parser
        return query_gen_chain  # type: ignore

    def _get_prompt_generator(self) -> Runnable[_PromptVariables, str]:
        """Get the prompt generator chain"""
        _logger.debug("Creating prompt generator for model %s",
                      self.prompt_model_name)
        output_parser = StrOutputParser()
        rate_limiter = get_rate_limiter(self.prompt_model_name)
        model = ChatGroq(name=self.prompt_model_name, rate_limiter=rate_limiter,
                         **self.prompt_model_args)
        prompt_gen_template = PromptTemplate.from_template("{query}")
        prompt_gen_chain = prompt_gen_template | model | output_parser
        return prompt_gen_chain.with_retry(  # type: ignore
            stop_after_attempt=self.prompt_model_args["max_retries"])

    def get_queries(self, *, dataset_name: str, dataset_description: str,
                    amount: int) -> list[str]:
        """Obtain the queries for a dataset using a chain."""
        _logger.debug("Generating %i prompts for dataset %s",
                      amount, dataset_name)
        is_dtd = dataset_name == "Describable Textures Dataset"
        chain = self._get_query_generator(is_dtd=is_dtd)
        with tracing_v2_enabled(project_name="MPVR TFM"):
            queries = chain.invoke({"amount": amount,
                                    "dataset_name": dataset_name,
                                    "dataset_description": dataset_description})
        return queries

    def get_prompts(self, queries: list[str], categories: list[str],
                    n: int = 10) -> list[list[str]]:
        """Get the prompts for the queries and categories."""
        _logger.debug("Generating prompts for categories")
        chain = self._get_prompt_generator()
        prompts: list[list[str]] = []
        for category in tqdm(categories, desc="Categories", position=0):
            category_prompts: list[list[str]] = []
            with BatchBarCallback(len(queries) * n,
                                  desc=f"Queries for {category}",
                                  position=1, leave=False) as bar_cb:
                config: RunnableConfig = {
                    "callbacks": [bar_cb],
                    "run_name": f"Prompt for {category}"
                }
                completed = False
                for _ in trange(n, leave=False, position=1, desc="Prompt "):
                    while not completed:
                        try:
                            category_prompts.append(
                                chain.batch(
                                    [{"query": query.format(category=category)}
                                     for query in queries], config=config
                                )
                            )
                            completed = True
                        except OutputParserException as e:
                            raise e
                        except Exception:
                            _logger.exception(
                                "Exception raised during batch. Retrying in 30 minutes.")
                            for _ in trange(0, 30 * 60,
                                            desc="Waiting time", leave=False,
                                            position=1, unit="s"):
                                time.sleep(1)
            prompts.append(flatten_list_1d(category_prompts))
        return prompts
