import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import time
import torch
from torch.profiler import profile, record_function, ProfilerActivity

import mmfreelm
from transformers import AutoModelForCausalLM, AutoTokenizer

def calculate_model_flops(model, num_tokens):
    """
    Estimate FLOPS for model generation.
    For transformer models: FLOPS ≈ 2 * num_params * num_tokens
    """
    num_params = sum(p.numel() for p in model.parameters())
    # Each parameter is used twice per token (forward multiply-add)
    flops = 2 * num_params * num_tokens
    return flops, num_params

def benchmark_generation(model, tokenizer, prompts, max_length=32, num_iterations=1):
    """Run benchmark with multiple prompts and iterations."""
    results = {
        'tps': [],
        'generation_time': [],
        'tokens_generated': [],
        'gflops_per_sec': [],
        'prompt_lengths': []
    }

    print(f"\n{'='*80}")
    print(f"RUNNING BENCHMARK")
    print(f"{'='*80}")
    print(f"Number of prompts: {len(prompts)}")
    print(f"Iterations per prompt: {num_iterations}")
    print(f"Max generation length: {max_length}")
    print(f"{'='*80}\n")

    for prompt_idx, prompt in enumerate(prompts):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        prompt_length = input_ids.shape[1]

        print(f"Prompt {prompt_idx + 1}/{len(prompts)} (length={prompt_length}): \"{prompt[:50]}...\"")

        # Warm-up
        _ = model.generate(input_ids, pad_token_id=tokenizer.eos_token_id, max_length=max_length, do_sample=True, top_p=0.4, temperature=0.6)

        # Benchmark iterations
        for iteration in range(num_iterations):
            start_time = time.time()
            outputs = model.generate(input_ids, pad_token_id=tokenizer.eos_token_id, max_length=max_length, do_sample=True, top_p=0.4, temperature=0.6)
            end_time = time.time()

            # Calculate metrics
            generation_time = end_time - start_time
            output_length = outputs.shape[1]
            tokens_generated = output_length - prompt_length
            tokens_per_second = tokens_generated / generation_time

            # Calculate FLOPS
            total_flops, _ = calculate_model_flops(model, tokens_generated)
            flops_per_second = total_flops / generation_time
            gflops_per_sec = flops_per_second / 1e9

            results['tps'].append(tokens_per_second)
            results['generation_time'].append(generation_time)
            results['tokens_generated'].append(tokens_generated)
            results['gflops_per_sec'].append(gflops_per_sec)
            results['prompt_lengths'].append(prompt_length)

            print(f"  Iteration {iteration + 1}: {tokens_per_second:.2f} tok/s, {gflops_per_sec:.2f} GFLOPS/s, {generation_time:.4f}s")

        print()

    return results

def print_benchmark_results(results, model, implementation_type):
    """Print comprehensive benchmark statistics."""
    import statistics

    # Convert to tensors for easier calculation
    tps_values = results['tps']
    gflops_values = results['gflops_per_sec']
    time_values = results['generation_time']
    tokens_values = results['tokens_generated']

    num_params = sum(p.numel() for p in model.parameters())

    print(f"\n{'='*80}")
    print(f"BENCHMARK RESULTS - {implementation_type}")
    print(f"{'='*80}")
    print(f"\nModel Configuration:")
    print(f"  Total Parameters: {num_params:,}")
    print(f"  Total Runs: {len(tps_values)}")

    print(f"\nTokens Per Second (TPS):")
    print(f"  Mean:   {statistics.mean(tps_values):>10.2f} tok/s")
    print(f"  Median: {statistics.median(tps_values):>10.2f} tok/s")
    print(f"  Std:    {statistics.stdev(tps_values) if len(tps_values) > 1 else 0:>10.2f} tok/s")
    print(f"  Min:    {min(tps_values):>10.2f} tok/s")
    print(f"  Max:    {max(tps_values):>10.2f} tok/s")

    print(f"\nCompute Performance (GFLOPS/s):")
    print(f"  Mean:   {statistics.mean(gflops_values):>10.2f} GFLOPS/s")
    print(f"  Median: {statistics.median(gflops_values):>10.2f} GFLOPS/s")
    print(f"  Std:    {statistics.stdev(gflops_values) if len(gflops_values) > 1 else 0:>10.2f} GFLOPS/s")
    print(f"  Min:    {min(gflops_values):>10.2f} GFLOPS/s")
    print(f"  Max:    {max(gflops_values):>10.2f} GFLOPS/s")

    print(f"\nGeneration Time:")
    print(f"  Mean:   {statistics.mean(time_values):>10.4f} s")
    print(f"  Median: {statistics.median(time_values):>10.4f} s")
    print(f"  Std:    {statistics.stdev(time_values) if len(time_values) > 1 else 0:>10.4f} s")
    print(f"  Min:    {min(time_values):>10.4f} s")
    print(f"  Max:    {max(time_values):>10.4f} s")

    print(f"\nTokens Generated:")
    print(f"  Mean:   {statistics.mean(tokens_values):>10.1f}")
    print(f"  Total:  {sum(tokens_values):>10}")

    # Breakdown by prompt length
    unique_lengths = sorted(set(results['prompt_lengths']))
    if len(unique_lengths) > 1:
        print(f"\nPerformance by Prompt Length:")
        print(f"  {'Length':<10} {'Avg TPS':<15} {'Avg GFLOPS/s':<15}")
        print(f"  {'-'*10} {'-'*15} {'-'*15}")
        for length in unique_lengths:
            indices = [i for i, l in enumerate(results['prompt_lengths']) if l == length]
            avg_tps = statistics.mean([tps_values[i] for i in indices])
            avg_gflops = statistics.mean([gflops_values[i] for i in indices])
            print(f"  {length:<10} {avg_tps:<15.2f} {avg_gflops:<15.2f}")

    print(f"\n{'='*80}\n")

def main():
    # Check if user wants to use fixed-point implementation
    use_fixed_point = "--fixed-point" in sys.argv

    # Benchmark prompts with varying lengths
    prompts = [
        "The quick brown fox",
        "In a shocking finding, scientist discovered a herd of unicorns living in a remote, ",
        "Once upon a time in a faraway kingdom, there lived a wise old wizard who possessed magical powers beyond imagination",
        "Machine learning is",
        "The future of artificial intelligence will bring transformative changes to society, economy, and daily life in ways we are only beginning",
    ]

    if use_fixed_point:
        print("Using fixed-point HGRN implementation with ternary_matmul operations...")
        # Import and use the integrated fixed-point version
        from generate_integrated import replace_with_fixed_point_hgrn

        # Load model
        name = 'ridger/MMfreeLM-370M'
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name, device_map='cpu').half()

        # Replace HGRN layers with fixed-point implementation
        model = replace_with_fixed_point_hgrn(model)
        print("✓ HGRN layers replaced with fixed-point implementation using ternary_matmul")

        implementation_type = "Fixed-Point HGRN"
    else:
        print("Using standard floating-point implementation...")
        # Original implementation
        name = 'ridger/MMfreeLM-370M'
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name, device_map='cpu').half()

        implementation_type = "Floating-Point"

    # Run benchmark
    results = benchmark_generation(model, tokenizer, prompts, max_length=32, num_iterations=1)

    # Print results
    print_benchmark_results(results, model, implementation_type)

    # Show a sample generation
    print(f"{'='*80}")
    print(f"SAMPLE GENERATION")
    print(f"{'='*80}")
    sample_prompt = prompts[1]
    input_ids = tokenizer(sample_prompt, return_tensors="pt").input_ids
    outputs = model.generate(input_ids, pad_token_id=tokenizer.eos_token_id, max_length=32, do_sample=True, top_p=0.4, temperature=0.6)
    print(f"\nInput:  {sample_prompt}")
    print(f"Output: {tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]}")

    if use_fixed_point:
        print("\n✓ Benchmark completed using fixed-point HGRN with ternary_matmul operations!")
        print("  Each HGRN cell performed 4 ternary_matmul operations:")
        print("  1. Input gate projection (w_i)")
        print("  2. Forget gate projection (w_f)")
        print("  3. Gate projection (w_g)")
        print("  4. Output projection (w_o)")

if __name__ == "__main__":
    main()