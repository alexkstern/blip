import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple, List, Optional
from pathlib import Path

def load_model(model_name: str = "gpt2-medium") -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def get_layer_weights(model: AutoModelForCausalLM, layer_name: str) -> torch.Tensor:
    """Extract weights from a specific layer."""
    return dict(model.named_parameters())[layer_name].detach()

def set_layer_weights(model: AutoModelForCausalLM, layer_name: str, weights: torch.Tensor) -> None:
    """Set weights for a specific layer."""
    dict(model.named_parameters())[layer_name].data = weights

