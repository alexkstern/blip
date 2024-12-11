from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass

@dataclass
class GenerationConfig:
    max_length: int = 10  # Changed from 10 to 100 for more reasonable output length
    temperature: float = 0.7
    num_return_sequences: int = 1
    do_sample: bool = True
    
class ModelWrapper:
    def __init__(self, 
                 model_name: str = "HuggingFaceTB/SmolLM-135M",
                 device: Optional[str] = None):
        """
        Initialize the model wrapper.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set up padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.default_config = GenerationConfig()
    
    def generate(self, 
                text: str, 
                generation_config: Optional[GenerationConfig] = None,
                return_time: bool = False) -> str | tuple[str, float]:
        """
        Generate text based on input prompt.
        """
        config = generation_config or self.default_config
        
        start = time.time()
        
        inputs = self.tokenizer(text, 
                              return_tensors="pt", 
                              add_special_tokens=True)
        inputs = inputs.to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                pad_token_id=self.tokenizer.pad_token_id,
                max_length=config.max_length,
                num_return_sequences=config.num_return_sequences,
                temperature=config.temperature,
                do_sample=config.do_sample
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        end = time.time()
        generation_time = end - start
        
        if return_time:
            return generated_text, generation_time
        return generated_text
    
    def generate_batch(self, 
                      texts: list[str],
                      generation_config: Optional[GenerationConfig] = None,
                      return_time: bool = False) -> list[str] | tuple[list[str], float]:
        """
        Generate text for multiple prompts.
        """
        config = generation_config or self.default_config
        
        start = time.time()
        
        # Tokenize all inputs with padding
        inputs = self.tokenizer(texts, 
                              return_tensors="pt", 
                              add_special_tokens=True,
                              padding=True,
                              truncation=True)
        inputs = inputs.to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                pad_token_id=self.tokenizer.pad_token_id,
                max_length=config.max_length,
                num_return_sequences=config.num_return_sequences,
                temperature=config.temperature,
                do_sample=config.do_sample
            )
        
        # Reshape outputs if necessary (for multiple return sequences)
        if config.num_return_sequences > 1:
            outputs = outputs.reshape(len(texts), config.num_return_sequences, -1)
        
        # Decode outputs
        generated_texts = [
            self.tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]
        
        end = time.time()
        generation_time = end - start
        
        if return_time:
            return generated_texts, generation_time
        return generated_texts
    
    def update_default_config(self, **kwargs) -> None:
        """Update default generation configuration with new values."""
        for key, value in kwargs.items():
            if hasattr(self.default_config, key):
                setattr(self.default_config, key, value)
            else:
                raise ValueError(f"Invalid config parameter: {key}")

# Example usage
if __name__ == "__main__":
    # Initialize model wrapper
    model_names=["HuggingFaceTB/SmolLM-135M", "HuggingFaceTB/SmolLM-360M",
                  "HuggingFaceTB/SmolLM-1.7B", "HuggingFaceTB/SmolLM-360M-Instruct",
                    "HuggingFaceTB/SmolLM-1.7B-Instruct"]

    model_name=model_names[3]
    print(f"Using model: {model_name}")
    wrapper = ModelWrapper(model_name=model_name)
    
    # Single generation
    print("Single generation:")
    text, time_taken = wrapper.generate("Hey What's your name?", return_time=True)
    print(f"Generated: {text}")
    print(f"Time taken: {time_taken:.2f} seconds\n")
    
    # Batch generation
    print("Batch generation:")
    texts = ["Hey What's your name?", "How are you doing?"]
    generations, time_taken = wrapper.generate_batch(texts, return_time=True)
    for i, gen in enumerate(generations):
        print(f"Prompt {i+1}: {texts[i]}")
        print(f"Generated: {gen}")
    print(f"Total time taken: {time_taken:.2f} seconds")