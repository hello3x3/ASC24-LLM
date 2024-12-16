"""Benchmark offline inference throughput."""
import argparse
import json
import random
import time
from typing import List, Optional, Tuple

import torch
from tqdm import tqdm

from llama import Llama
from torch.utils.data import DataLoader


def format_requests(requests: List[Tuple[str, int, int]]):
    prompts, inp_lens, gen_lens = zip(*requests)
    prompts = list(prompts)
    return prompts, (inp_lens, gen_lens)


def run_transformer(
    requests: List[Tuple[str, int, int]],
    model: str,
    trust_remote_code: bool,
    batch_size: int = 4,  # Batch size for inference
) -> float:
    
    llm = Llama.build(
        ckpt_dir=model,
        max_seq_len=1024,
        max_batch_size=batch_size,
    )

    data_loader = DataLoader(requests, batch_size=batch_size, collate_fn=format_requests, shuffle=False)
    
    input_num_tokens = []
    output_num_tokens = []
    start = time.perf_counter()

    for prompts, (inp_lens, gen_lens) in tqdm(data_loader):
        
        out_lens = llm.text_completion(
            prompts,
            max_gen_len=max(gen_lens),
            temperature=1.0,
            top_p=1.0,
        )

        for i in range(len(inp_lens)):
            input_num_tokens.append(inp_lens[i])
            output_num_tokens.append(len(out_lens[i]))
    
    end = time.perf_counter()
    return end - start, input_num_tokens, output_num_tokens



def main(args: argparse.Namespace):
    global tokenizer
    print(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.dataset is None:
        # Synthesize a prompt with the given input length.
        prompt = "hi" * (args.input_len - 1)
        requests = [(prompt, args.input_len, args.output_len) for _ in range(args.num_samples)]

    else:
        with open(args.dataset) as f:
            requests = json.load(f)

    if args.num_samples is not None:
        requests = requests[0: args.num_samples]

    elapsed_time, input_num_tokens, output_num_tokens = run_transformer(requests, args.model, args.trust_remote_code, batch_size=args.batch_size)
    prompt_num_tokens = sum(prompt_len for prompt_len in input_num_tokens)
    total_num_tokens = sum(output_len for output_len in output_num_tokens)
    print(f"Throughput: {len(requests) / elapsed_time:.2f} requests/s \n"
          f"Tokens/s: {total_num_tokens / elapsed_time:.2f} tokens/s \n"
          f"Prompt_num_tokens:{prompt_num_tokens:.2f} tokens \n"
          f"Total_num_tokens:{total_num_tokens:.2f} tokens \n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the throughput.")
    parser.add_argument("--dataset", type=str, default=None, help="Path to the dataset.")
    parser.add_argument("--model", type=str, default="meta/llama2-70b")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--input-len", type=int, default=None, help="Input prompt length for each request")
    parser.add_argument("--output-len", type=int, default=None, help="Output length for each request")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of first few samples used for inference test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument('--trust-remote-code',
                        action='store_true',
                        help='trust remote code from huggingface')
    parser.add_argument(
        '--dtype',
        type=str,
        default='auto',
        choices=['auto', 'half', 'float16', 'bfloat16', 'float', 'float32'],
        help='data type for model weights and activations. '
        'The "auto" option will use FP16 precision '
        'for FP32 and FP16 models, and BF16 precision '
        'for BF16 models.')
    args = parser.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.model
    if args.dataset is None:
        assert args.input_len is not None
        assert args.output_len is not None
    else:
        assert args.input_len is None
        assert args.output_len is None

    main(args)
