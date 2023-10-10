import sys
import time
import random
import argparse
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from autoregressive_sampling import autoregressive_sampling
from speculative_sampling import speculative_sampling

from typing import List, Optional, Tuple
import random
import json

def sample_requests(
    dataset_path: str,
    num_requests: int,
) -> List[Tuple[str, int, int]]:
    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [data["conversations"][0]["value"] for data in dataset]
    # Sample the requests.
    return dataset[:num_requests]
    # sampled_requests = random.sample(dataset, num_requests)
    # return sampled_requests
   


parser = argparse.ArgumentParser(description='Speculative Sampling')
parser.add_argument('--method', default="speculative", help='Sampling Method (autogressive / speculative)')
parser.add_argument('--prompt', required=True, help='Input prompt')
parser.add_argument('--max_new_tokens', type=int, required=True, help='No. of max new tokens')
parser.add_argument('--target_model', default="facebook/opt-13b", help='Target model (HF Causal LM model)')
parser.add_argument('--draft_model', required=False, help='Draft model (HF Causal LM model)')
parser.add_argument('--temperature', default=0, type=float, help='Temperature')
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"



if args.method == "speculative":
    if args.draft_model is None:
        print("Draft model should be specified for Speculative Sampling")
        sys.exit(1)

    print("Using target model:", args.target_model)
    print("Using draft model:", args.draft_model)
    
    dataset_path = "/workspace/datasets/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json"
    sampled_requests = sample_requests(dataset_path,8)
    
    print(f"----------------------------------type(sampled_requests):{type(sampled_requests)}")
     
    # print(f"【INFO】min_length:{min_length}")
        
    print(f"【INFO】sampled_requests:{sampled_requests}")
    # print(f"【INFO】sampled_requests:{len(sampled_requests[0])}")
    # print(f"【INFO】sampled_requests:{len(sampled_requests[1])}")
    # print(f"【INFO】sampled_requests:{len(sampled_requests[2])}")
    # print(f"【INFO】sampled_requests:{len(sampled_requests[3])}") 
    
    target_model = AutoModelForCausalLM.from_pretrained(args.target_model).to(device)
    draft_model = AutoModelForCausalLM.from_pretrained(args.draft_model).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    # sampled_requests = ["Emily found a mysterious letter on her doorstep one sunny morning.",
    #               "Emily found a mysterious letter on her doorstep one sunny morning.",
    #               "Emily found a mysterious letter on her doorstep one sunny morning.",
    #               "Emily found a mysterious letter on her doorstep one sunny morning."]
    # inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    # inputs = tokenizer(sampled_requests, return_tensors="pt").to(device)
    
    input_ids = [tokenizer(i, return_tensors="pt").input_ids for i in sampled_requests]
    print(f"【INFO】input_ids:{input_ids}")
    # min_length = min([len(i) for i in input_ids])
    # input_ids = torch.tensor([i.input_ids[0] for i in inputs]).to(device)
    # input_ids = inputs.input_ids
    input_ids = input_ids[:][:5]
    

    start_time = time.time_ns()
    tokens = speculative_sampling(target_model, draft_model, initial_prompt_seq=input_ids, max_new_tokens=args.max_new_tokens, tokenizer=tokenizer, temperature=args.temperature, debug=False)
    end_time = time.time_ns()

    new_tokens = (len(tokens[0]) - len(inputs.input_ids[0])) * len(sampled_requests)
    time_taken = (end_time - start_time) / 1_000_000_000

    # print(tokenizer.decode(tokens))
    for i in range (len(sampled_requests)):
        print(tokenizer.decode(tokens[i]))
        
    print()
    print(f"Latency (Speculative Sampling): {new_tokens/time_taken:.2f} tok/s")

elif args.method == "autoregressive":
    print("Using target model:", args.target_model)

    target_model = AutoModelForCausalLM.from_pretrained(args.target_model).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    
    dataset_path = "/workspace/datasets/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json"
    sampled_requests = sample_requests(dataset_path,16)
    import numpy as np
    min_length = min([len(i) for i in sampled_requests])
    sampled_requests = [i[:min_length] for i in sampled_requests]
    
    # testTensor = ["Emily found a mysterious letter on her doorstep one sunny morning.",
    #               "Emily found a mysterious letter on her doorstep one sunny morning.",
    #               "Emily found a mysterious letter on her doorstep one sunny morning.",
    #               "Emily found a mysterious letter on her doorstep one sunny morning."]
    inputs = tokenizer(sampled_requests, return_tensors="pt").to(device)
    print(f"【INFO】inputs:{inputs}")
    print(f"【INFO】inputs.input_ids:{inputs.input_ids}")

    # inputs = tokenizer(args.prompt, return_tensors="pt").to(device)

    start_time = time.time_ns()
    tokens = autoregressive_sampling(target_model, initial_prompt_seq=inputs.input_ids, target_len=args.max_new_tokens+len(inputs.input_ids), temperature=args.temperature)
    end_time = time.time_ns()

    new_tokens = (len(tokens[0]) - len(inputs.input_ids[0])) * len(sampled_requests) # batch_size
    time_taken = (end_time - start_time) / 1_000_000_000

    for i in range (len(sampled_requests)):
        print(tokenizer.decode(tokens[i]))
    print()
    print(f"Latency (Naive Autoregressive Sampling): {new_tokens/time_taken:.2f} tok/s")

else:
    print("Method should be either autoregressive / speculative")