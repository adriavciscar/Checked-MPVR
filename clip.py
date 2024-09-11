"""Module for handling CLIP"""
import logging

import torch
from PIL.ImageFile import ImageFile
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel


__all__ = ["load_clip", "get_text_embeds",
           "get_image_embeds", "zeroshot_inference", "CLIPProcessor",
           "CLIPModel", "DEVICE"]


_logger = logging.getLogger(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_clip(*, model_path="openai/clip-vit-base-patch32",
              processor_path="openai/clip-vit-base-patch32",
              eval_=True) -> tuple[CLIPModel, CLIPProcessor]:
    """Load the CLIP model specified."""
    _logger.info("Using device: %s", DEVICE)
    _logger.debug("Loading processor from %s", processor_path)
    processor = CLIPProcessor.from_pretrained(processor_path)
    _logger.info("Processor loaded")
    _logger.debug("Loading model from %s", model_path)
    model = CLIPModel.from_pretrained(model_path).to(DEVICE)  # type: ignore
    _logger.info("Model loaded")
    if eval_:
        _logger.info("Putting model in evaluation mode")
        model.eval()
    else:
        _logger.info("Keeping model in training mode")
    return model, processor  # type: ignore


def get_text_embeds(prompts: list[list[str]],
                    model: CLIPModel,
                    processor: CLIPProcessor) -> torch.Tensor:
    """Get the text embeddings using the model and processor."""
    _logger.debug("Getting text embeddings")
    embeds = []
    for label_prompts in tqdm(prompts):
        inputs = processor(text=label_prompts,
                           return_tensors="pt",
                           padding=True,
                           truncation=True).to(DEVICE)
        outputs = model.get_text_features(**inputs)  # type: ignore
        prompt_mean = torch.mean(outputs, dim=0)
        prompt_norm = prompt_mean / prompt_mean.norm(p=2, dim=-1, keepdim=True)
        embeds.append(prompt_norm)
        del inputs, outputs, prompt_mean, prompt_norm
    return torch.stack(embeds)


def get_image_embeds(images: list[ImageFile],
                     model: CLIPModel,
                     processor: CLIPProcessor) -> torch.Tensor:
    """Get the image embeddings using the model and processor."""
    _logger.debug("Getting image embeddings")
    inputs = processor(images=images, return_tensors="pt").to(DEVICE)
    image_embeds = model.get_image_features(**inputs)  # type: ignore
    image_norm = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
    del image_embeds, inputs
    return image_norm


def zeroshot_inference(model: CLIPModel,
                       img_embeds: torch.Tensor,
                       text_embeds: torch.Tensor) -> torch.Tensor:
    """Make the zero-shot inference using the image and text embeds."""
    _logger.info("Running zero-shot inference")
    _logger.debug("Calculating logits")
    logit_scale = model.logit_scale.exp()
    logits_per_text = torch.matmul(text_embeds, img_embeds.t()) * logit_scale
    logits_per_image = logits_per_text.t()
    del logits_per_text, logit_scale, img_embeds, text_embeds
    return logits_per_image
