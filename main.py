from pathlib import Path
import argparse
from model_utils import load_model, get_layer_weights
from low_rank import LowRankApproximation
from sampler import LaplaceSampler
from inference import ModelInference

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='gpt2-medium')
    parser.add_argument('--layer_name', type=str, default='lm_head.weight')
    parser.add_argument('--rank', type=int, default=100)
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default='outputs')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load model
    model, tokenizer = load_model(args.model_name)
    
    # Get layer weights
    W = get_layer_weights(model, args.layer_name)
    
    # Compute low-rank approximation
    approximator = LowRankApproximation(args.rank)
    U, S, V = approximator.decompose(W)
    
    # Save low-rank approximation
    approximator.save(U, S, V, W.shape, output_dir / 'low_rank_approx.pt')
    
    # Generate samples
    sampler = LaplaceSampler()
    samples = sampler.sample(U, S, V, args.num_samples)
    
    # Save samples
    sampler.save_samples(samples, output_dir / 'samples.pt')
    
    # Example inference
    inferencer = ModelInference(model, tokenizer, args.layer_name)
    prompt = "The future of AI is"
    
    print("Original output:")
    print(inferencer.generate_with_weights(prompt, W))
    
    print("\nSample outputs:")
    for i, sample in enumerate(samples[:3]):
        print(f"\nSample {i+1}:")
        print(inferencer.generate_with_weights(prompt, sample))

if __name__ == '__main__':
    main()