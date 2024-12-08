"""Benchmark offline inference throughput."""
import argparse
import json
import random
import time
from typing import List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase
from tqdm import tqdm

from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import gather_object


accelerator = Accelerator()


def collate_fn(batch):
    """ Custom collate_fn for DataLoader to handle padding for different sequence lengths """
    prompts, prompt_lens, output_lens = zip(*batch)
    tokenizer.padding_side = "left"
    inputs = tokenizer(
        list(prompts),
        padding="longest",
        truncation=False,
        pad_to_multiple_of=8,
        return_tensors="pt",
        add_special_tokens=False
    )
    tokenizer.padding_side = "right"
    return inputs, output_lens


def run_hf(
    requests: List[Tuple[str, int, int]],
    model: str,
    tokenizer: PreTrainedTokenizerBase,
    trust_remote_code: bool,
    batch_size: int = 16,  # Batch size for inference
    max_input_length: int = 512  # Max length for padding
) -> float:

    cuda_devices = torch.cuda.device_count()
    max_memory = {cuda_device: "32GiB" for cuda_device in range(cuda_devices)}

    llm = AutoModelForCausalLM.from_pretrained(
        model,
        # device_map={"": accelerator.process_index},
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=trust_remote_code,
        max_memory=max_memory,
    )

    if llm.config.model_type == "llama":
        # To enable padding in the HF backend.
        tokenizer.pad_token = tokenizer.eos_token
    # llm = llm.cuda()

    input_num_tokens = []
    output_num_tokens = []
    start = time.perf_counter()

    accelerator.wait_for_everyone()
    with accelerator.split_between_processes(requests) as request:
        results = dict(outputs=[], input_num_tokens=0, output_num_tokens=0)
        # 使用 DataLoader 加载数据并进行批量处理
        data_loader = DataLoader(request, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
        for batch in tqdm(data_loader):
            inputs, output_lens_batch = batch
            input_ids_batch = inputs["input_ids"].cuda()

            llm_outputs = llm.generate(
                input_ids=input_ids_batch,
                do_sample=False,
                num_return_sequences=1,
                num_beams=1,
                temperature=1.0,
                top_p=1.0,
                use_cache=True,
                max_new_tokens=max(output_lens_batch),
            )

            for idx in range(len(output_lens_batch)):
                tokenizer.decode(llm_outputs[idx], skip_special_tokens=True)
                input_num_tokens.append(len(input_ids_batch[idx]))
                output_num_tokens.append(len(llm_outputs[idx]))
                results["input_num_tokens"] += len(input_ids_batch[idx])
                results["output_num_tokens"] += len(llm_outputs[idx])

        results = [results]

    results_gathered = gather_object(results)

    end = time.perf_counter()
    # print(">>> result:")
    # print(results)
    if accelerator.is_main_process:
        end = time.perf_counter()
        # print(">>> this is the main thread! <<<")
        # print(">>> results_gathered:")
        # print(results_gathered)
        input_num_tokens = [sum([r["input_num_tokens"] for r in results_gathered])]
        output_num_tokens = [sum([r["output_num_tokens"] for r in results_gathered])]
        # print(">>> input_num_tokens:")
        # print(input_num_tokens)
        # print(">>> output_num_tokens:")
        # print(output_num_tokens)
    return end - start, input_num_tokens, output_num_tokens



def main(args: argparse.Namespace):
    global tokenizer
    print(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Sample the requests.
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=args.trust_remote_code
    )
    if args.dataset is None:
        # Synthesize a prompt with the given input length.
        prompt = "hi" * (args.input_len - 1)
        requests = [(prompt, args.input_len, args.output_len) for _ in range(args.num_samples)]

    else:
        with open(args.dataset) as f:
            requests = json.load(f)

    if args.num_samples is not None:
        requests = requests[0: args.num_samples]

    elapsed_time, input_num_tokens, output_num_tokens = run_hf(requests, args.model, tokenizer,  args.trust_remote_code)
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