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
    dataset = [data["conversations"][0]["value"] for data in dataset if len(data["conversations"][0]["value"])>100 and len(data["conversations"][0]["value"])<150] # 只要字符长度大于1000的prompt
    # Sample the requests.
    # return dataset[:num_requests]  #改为随机取样  
    import random
    # 使用随机抽样获取指定数量的样本
    random_dataset = random.sample(dataset, num_requests)
    # 返回随机抽样后的数据集
    return random_dataset

    # sampled_requests = random.sample(dataset, num_requests)
    # return sampled_requests

parser = argparse.ArgumentParser(description='Speculative Sampling')
parser.add_argument('--method', default="speculative", help='Sampling Method (autogressive / speculative)')
parser.add_argument('--batch_size', type=int, default=1, help='Input prompt batch size')
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
    sampled_requests = sample_requests(dataset_path,args.batch_size)
    for request in sampled_requests:
        print(f"【INPUT】{request}")
    
    print(f"【INFO】lenght sampled_requests:{len(sampled_requests)}")
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
    input_ids = [tokenizer(i, return_tensors="pt").input_ids.to(device) for i in sampled_requests]  # [len(sampled_requests),]  每一个维度都是一个Tensor 采样多少个请求，就是多少个Tensor

    # truncation操作，padding的反义词
    # 获取所有 Tensor 第二个维度的最小值 也就是token数量的最小值
    min_dim = min(tensor.shape[1] for tensor in input_ids)
    print(f"【INFO】min_dim:{min_dim}")
    
    # 截断每个 Tensor 的第二个维度 
    input_ids = [tensor[:, :min_dim] for tensor in input_ids] # [batch_size,Tensor]

    start_time = time.time_ns()
    tokens = speculative_sampling(target_model, draft_model, initial_prompt_seq=input_ids, max_new_tokens=args.max_new_tokens, tokenizer=tokenizer, temperature=args.temperature, debug=False)
    end_time = time.time_ns()
    
    print(f"【INFO】tokens.shape:{tokens.shape}")

    # new_tokens = (len(tokens[0]) - len(input_ids_list[0])) * len(sampled_requests)
    new_tokens = 0
    for token in tokens:
        new_tokens += len(token) - min_dim
    time_taken = (end_time - start_time) / 1_000_000_000

    # print(tokenizer.decode(tokens))
    for i in range (len(tokens)):
        print(f"【OUTPUT】{tokenizer.decode(tokens[i])}")
        
    print()
    print(f"Latency (Speculative Sampling): {new_tokens/time_taken:.2f} tok/s")

elif args.method == "autoregressive":
    print("Using target model:", args.target_model)

    target_model = AutoModelForCausalLM.from_pretrained(args.target_model).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    
    dataset_path = "/workspace/datasets/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json"
    sampled_requests = sample_requests(dataset_path,args.batch_size)
    
    input_ids = [tokenizer(i, return_tensors="pt").input_ids.to(device) for i in sampled_requests]  # [len(sampled_requests),]  每一个维度都是一个Tensor 采样多少个请求，就是多少个Tensor

    # truncation操作，padding的反义词
    # 获取所有 Tensor 第二个维度的最小值 也就是token数量的最小值
    min_dim = min(tensor.shape[1] for tensor in input_ids)
    print(f"【INFO】min_dim:{min_dim}")
    
    # 截断每个 Tensor 的第二个维度 
    input_ids = [tensor[:, :min_dim] for tensor in input_ids] # [batch_size,Tensor]
    
    # testTensor = ["Emily found a mysterious letter on her doorstep one sunny morning.",
    #               "Emily found a mysterious letter on her doorstep one sunny morning.",
    #               "Emily found a mysterious letter on her doorstep one sunny morning.",
    #               "Emily found a mysterious letter on her doorstep one sunny morning."]

    start_time = time.time_ns()
    tokens = autoregressive_sampling(target_model, initial_prompt_seq=input_ids, target_len=args.max_new_tokens+len(input_ids), temperature=args.temperature)
    end_time = time.time_ns()

    new_tokens = 0
    for token in tokens:
        new_tokens += len(token) - min_dim
    time_taken = (end_time - start_time) / 1_000_000_000

    for i in range (len(tokens)):
        print(f"【OUTPUT】{tokenizer.decode(tokens[i])}")
        
    print()
    print(f"Latency (Naive Autoregressive Sampling): {new_tokens/time_taken:.2f} tok/s")

else:
    print("Method should be either autoregressive / speculative")