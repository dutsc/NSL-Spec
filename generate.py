import os
import sys
import time
import random
import argparse
import torch
import contexttimer
from tqdm import tqdm
from colorama import Fore, Style
from transformers import AutoTokenizer, AutoModelForCausalLM
from autoregressive_sampling import autoregressive_sampling
from speculative_sampling import speculative_sampling
from speculative_sampling_kv_cache import speculative_sampling_kv_cache


from typing import List, Optional, Tuple
import random
import json


DEBUG = True
# DEBUG = False

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
    # 使用随机抽样获取指定数量的样本
    random_dataset = random.sample(dataset, num_requests)
    # 返回随机抽样后的数据集
    # print(f'{random_dataset = }')
    # random_dataset: list[num_requests]
    return random_dataset

def block_print(text):
    if DEBUG:
        text = ' ' + text + ' '
        print(f'{text:-^80}')

parser = argparse.ArgumentParser(description='Speculative Sampling')
parser.add_argument('--method', default="speculative", help='Sampling Method (autogressive / speculative)')
parser.add_argument('--batch_size', type=int, default=1, help='Input prompt batch size')
parser.add_argument('--max_new_tokens', type=int, required=True, help='No. of max new tokens')
parser.add_argument('--target_model', default="facebook/opt-13b", help='Target model (HF Causal LM model)')
parser.add_argument('--draft_model', required=False, help='Draft model (HF Causal LM model)')
parser.add_argument('--temperature', default=0, type=float, help='Temperature')
args = parser.parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"

block_print('parameters')
print(f'{args.batch_size = }')
print(f'{args.target_model = }')
if args.method == "speculative":
    print(f'{args.draft_model = }')

device_0 = 'cuda:0'
device_1 = 'cuda:1'

block_print('load model')
target_model = AutoModelForCausalLM.from_pretrained(args.target_model).to(device_0)
if args.method == "speculative":
    if args.draft_model is None:
        print("Draft model should be specified for Speculative Sampling")
        sys.exit(1)
    draft_model = AutoModelForCausalLM.from_pretrained(args.draft_model).to(device_1)

block_print('tokenize')
dataset_path = "/home/zwx/dataset/ShareGPT_V3_unfiltered_cleaned_split.json"
sampled_requests = sample_requests(dataset_path,args.batch_size) # list[batch_size]
# sampled_requests = ['Emily found a mysterious letter on her doorstep one sunny morning.']
tokenizer = AutoTokenizer.from_pretrained(args.target_model)
input_ids = [tokenizer(i, return_tensors="pt").input_ids.to(device_1) for i in sampled_requests]  # [len(sampled_requests),]  每一个维度都是一个Tensor 采样多少个请求，就是多少个Tensor
# input_ids[0].shape = torch.Size([1, str_len])
# truncation操作，padding的反义词
# 获取所有 Tensor 第二个维度的最小值 也就是token数量的最小值
min_dim = min(tensor.shape[1] for tensor in input_ids)
# 截断每个 Tensor 的第二个维度 
input_ids = [tensor[:, :min_dim] for tensor in input_ids] # [batch_size,Tensor]
# input_ids[0].shape = torch.Size([1, min_dim])

def print_text():
    # block_print('generated text')
    for i in range (tokens.shape[0]):
        output = [tokenizer.decode(tokens[i][j]) for j in range(tokens.shape[1])]
        output = output[min_dim:]
        output = ''.join(str(j) for j in output)
        input = [tokenizer.decode(input_ids[i][0][j]) for j in range(min_dim)]
        input = ''.join(str(j) for j in input)
        print('【INFO】' + Fore.GREEN + f'{input}' + Style.RESET_ALL, end='')
        print(Fore.BLUE + f'{output}' + Style.RESET_ALL)

def print_time(time_cost, text = ''):
    # block_print('latency')
    new_tokens = 0
    for token in tokens:
        new_tokens += len(token) - min_dim
    print(f"Latency {text}:  {new_tokens/time_cost:.2f} tok/s")

# warm up
block_print('speculative sampling (warm up)')
# 禁用输出
original_stdout = sys.stdout
sys.stdout = open(os.devnull, 'w')
tokens = speculative_sampling(target_model, draft_model, initial_prompt_seq=input_ids, max_new_tokens=args.max_new_tokens, 
                                tokenizer=tokenizer, temperature=args.temperature, debug=False)
# 恢复输出
sys.stdout.close()
sys.stdout = original_stdout

block_print('speculative sampling with kv cache')
times = 1
with contexttimer.Timer() as t:
    for i in range(times):
        tokens = speculative_sampling_kv_cache(target_model, draft_model, initial_prompt_seq=input_ids, max_new_tokens=args.max_new_tokens, 
                                      tokenizer=tokenizer, temperature=args.temperature, debug=False)
print_text()
print_time(t.elapsed / times, text='speculative_sampling')

block_print('speculative sampling')
times = 1
with contexttimer.Timer() as t:
    for i in range(times):
        tokens = speculative_sampling(target_model, draft_model, initial_prompt_seq=input_ids, max_new_tokens=args.max_new_tokens, 
                                      tokenizer=tokenizer, temperature=args.temperature, debug=False)
print_text()
print_time(t.elapsed / times, text='speculative_sampling')

block_print('autoregressive sampling with kv cache')
times = 1
with contexttimer.Timer() as t:
    for i in range(times):
        tokens = autoregressive_sampling(target_model, initial_prompt_seq=input_ids, target_len=args.max_new_tokens+len(input_ids), 
                                         temperature=args.temperature)
print_text()
print_time(t.elapsed / times, text='autoregressive_sampling')
block_print('autoregressive sampling')
times = 1
with contexttimer.Timer() as t:
    for i in range(times):
        tokens = autoregressive_sampling(target_model, initial_prompt_seq=input_ids, target_len=args.max_new_tokens+len(input_ids), 
                                         temperature=args.temperature, cache=False)
print_text()
print_time(t.elapsed / times, text='autoregressive_sampling')

# block_print('autoregressive sampling (small model)')
# times = 1
# with contexttimer.Timer() as t:
#     for i in range(times):
#         tokens = autoregressive_sampling(draft_model, initial_prompt_seq=input_ids, target_len=args.max_new_tokens+len(input_ids), 
#                                          temperature=args.temperature)
# # print_text()
# print_time(t.elapsed / times, text='autoregressive_sampling_draft_model')