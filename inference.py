import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List

class ModelInference:
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, layer_name: str):
        self.model = model
        self.tokenizer = tokenizer
        self.layer_name = layer_name
        self.original_weights = dict(model.named_parameters())[layer_name].data.clone()
    
    def generate_with_weights(self, prompt: str, weights: torch.Tensor, 
                            max_length: int = 50) -> str:
        """Generate text using provided weights."""
        # Set new weights
        dict(self.model.named_parameters())[self.layer_name].data = weights
        
        # Tokenize and generate
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=max_length,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Restore original weights
        dict(self.model.named_parameters())[self.layer_name].data = self.original_weights
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
