"""Main module for the zero-shot inference.."""
from typing import Optional, TypedDict

import os
import logging
import pathlib
import json
import importlib
import datetime

import torch
import dotenv

from tqdm import tqdm
from PIL import Image

from agents import get_agent
from clip import (get_text_embeds, load_clip, zeroshot_inference,
                  get_image_embeds, CLIPModel, CLIPProcessor, DEVICE)
from datasets import get_dataset_files
from utils import parse_logging_level, chunks

_logger = logging.getLogger(__name__)


# Shorthand types
type DatasetFiles = tuple[list[str], list[str]]


class _EmbeddingsRecord(TypedDict):
    features: torch.Tensor
    targets: list[str]


def load_dataset(nick: str) -> tuple[str, str, Optional[_EmbeddingsRecord],
                                     Optional[DatasetFiles]]:
    """Load the dataset info and image embeddings"""
    ds = importlib.import_module(f"datasets.{nick}")
    embeds_path = pathlib.Path("embeds", f"{nick}.pth")
    image_embs: _EmbeddingsRecord | None = None
    dataset_files = None
    if embeds_path.exists():
        image_embs = torch.load(embeds_path)
        assert image_embs is not None
        image_embs["features"] = image_embs["features"].to(DEVICE)
    else:
        dataset_files = get_dataset_files(nick)

    return ds.NAME, ds.DESCRIPTION, image_embs, dataset_files


def get_ds_img_embeds(
    model: CLIPModel, processor: CLIPProcessor, ds_nick: str, files: list[str],
    labels: list[str], *, chunk_size: int = 150, save_embeds: bool = True
) -> _EmbeddingsRecord:
    """Get the image embeddings of a dataset."""
    img_emb_list = []
    dataset_path = pathlib.Path("datasets", ds_nick, "images")
    with (torch.no_grad(),
          tqdm(total=len(files), desc=f"Calculating embeddings {ds_nick}",
               leave=False) as progress_bar):
        for images_chunk in chunks(files, chunk_size):
            opened_images = [Image.open(dataset_path/image)
                             for image in images_chunk]
            image_embeds = get_image_embeds(opened_images, model, processor)
            img_emb_list.append(image_embeds)
            for image in opened_images:
                image.close()
            del image_embeds, opened_images
            progress_bar.update(chunk_size)
    img_emb_list = torch.cat(img_emb_list)
    image_emb: _EmbeddingsRecord = {
        "features": img_emb_list,
        "targets": labels
    }
    if save_embeds:
        embeds_path = pathlib.Path("embeds", f"{ds_nick}.pth")
        torch.save(image_emb, embeds_path)
    del img_emb_list, labels
    return image_emb


def run_experiment(
    ds_nick: str, agent_name: str, query_model_name: str,
    prompt_model_name: str, *, save_embeds: bool = True,
    save_prompts: bool = True, save_queries: bool = True,
    save_results: bool = True,
    temps: tuple[float | None, float | None] = (None, None)
) -> float:
    """Run the experiment."""
    init_time = datetime.datetime.now().strftime("%d-%m-%YT%H-%M-%S")
    _logger.debug("Starting experiment: %s", init_time)
    # Import dataset
    _logger.debug("Loading dataset")
    ds_name, ds_desc, image_emb, ds_files = load_dataset(ds_nick)
    _logger.info("Dataset %s loaded", ds_nick)
    # Load CLIP
    _logger.debug("Loading CLIP")
    clip_model, clip_processor = load_clip()
    _logger.info("CLIP loaded")
    # Image embeddings generation
    if image_emb is None:
        _logger.debug("Embeddings not found, generating embeddings")
        assert ds_files is not None
        image_emb = get_ds_img_embeds(clip_model, clip_processor, ds_nick,
                                      ds_files[0], ds_files[1],
                                      save_embeds=save_embeds)
        _logger.info("Embeddings generated")
    # Chain creation
    agent_class = get_agent(agent_name)
    query_model_args = None
    prompt_model_args = None
    if temps[0] is not None:
        query_model_args = {"temperature": temps[0]}
    if temps[1] is not None:
        prompt_model_args = {"temperature": temps[1]}
    agent = agent_class(query_model_name, prompt_model_name,
                        query_model_args, prompt_model_args)
    # Prompt and queries generation.
    text_path = pathlib.Path("llm_results", ds_nick)
    text_path.mkdir(exist_ok=True)
    categories: list[str] = list(set(image_emb["targets"]))
    _logger.debug("Getting queries")
    queries = agent.get_queries(dataset_name=ds_name,
                                dataset_description=ds_desc, amount=30)
    _logger.info("Queries generated")
    if save_queries:
        with open(text_path/f"queries-{init_time}.json", mode="w",
                  encoding="utf-8") as file:
            json.dump(queries, file)
        _logger.debug("Queries saved on %s", text_path)
    _logger.debug("Getting prompts")
    prompts = agent.get_prompts(queries, categories)
    _logger.info("Prompts generated")
    if save_prompts:
        with open(text_path/f"prompts-{init_time}.json", mode="w",
                  encoding="utf-8") as file:
            json.dump(prompts, file)
        _logger.debug("Prompts saved on %s", text_path)
    # Text embeddings generation
    _logger.debug("Generating text embeddings")
    with torch.no_grad():
        text_embeds = get_text_embeds(prompts, clip_model, clip_processor)
    _logger.info("Text embeddings generated")
    # Zero-shot inference
    _logger.debug("Making zero-shot clasification")
    with torch.no_grad():
        features = image_emb["features"]
        logits_per_image = zeroshot_inference(
            clip_model, features, text_embeds)
    probs = logits_per_image.softmax(dim=1)
    _logger.info("Zero-shot clasification done")
    true_positives = 0
    for image_probs, true_label in zip(probs, image_emb["targets"]):
        max_class_idx = image_probs.argmax()
        if true_label == categories[max_class_idx]:
            true_positives += 1
    accuracy = true_positives / len(probs)
    _logger.info("Top-1%% accuracy of %s: %.3f", ds_nick, accuracy)
    results_path = pathlib.Path("results", f"{ds_nick}.txt")
    if save_results:
        with open(results_path, mode="a", encoding="utf-8") as file:
            file.write(
                f"[{init_time}] - {agent_name}/{prompt_model_name} - Top-1% accuracy: {accuracy}\n")
    return accuracy


def main() -> None:
    """Main function."""
    dataset_nick = os.getenv("DATASET")
    assert dataset_nick is not None
    agent_name = os.getenv("AGENT")
    assert agent_name is not None
    query_model_name = os.getenv("CHAT_MODEL")
    assert query_model_name is not None
    prompt_model_name = os.getenv("CHAT_MODEL")
    assert prompt_model_name is not None
    try:
        run_experiment(dataset_nick, agent_name,
                       query_model_name, prompt_model_name)
    except Exception:
        _logger.exception("Exception occured")


if __name__ == "__main__":
    logging.basicConfig()
    dotenv.load_dotenv()
    LOGGING_LEVEL = parse_logging_level(os.getenv("LOGGING_LEVEL", "INFO"))
    _logger.setLevel(LOGGING_LEVEL)
    _logger.debug("Logging level on DEBUG")
    main()
