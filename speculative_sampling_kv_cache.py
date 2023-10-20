import torch
import contexttimer
from transformers import AutoTokenizer
from utils import sample_from_draft_model, get_distribution, sample
from kvcache_model import KVCacheModel


def speculative_sampling_kv_cache(target_model, draft_model, initial_prompt_seq, max_new_tokens, tokenizer, lookahead=4, temperature=1.0, debug=True):
    '''
    Implementation of Algorithm 2 of the paper - Accelerating Large Language Model Decoding 
    with Speculative Sampling (https://arxiv.org/abs/2302.01318)
    '''
    # assert initial_prompt_seq.shape[0] == 1, 'Batch size should be 1'

    target_model = KVCacheModel(target_model, temperature)
    draft_model = KVCacheModel(draft_model, temperature)

    n = initial_prompt_seq[0].shape[-1]
    
    # 循环赋值 
    fin_prompt_seq = [prompt.detach().clone() for prompt in initial_prompt_seq]
    # 在第一个batch维度 concat成一个大Tensor
    fin_prompt_seq = torch.cat(fin_prompt_seq, dim=0) # 此时已经是张量 batch形式的
    # fin_prompt_seq : (batch_size, min_token_len(after truncation)))  (8,17)

    small_num = 0
    large_num = 0
    large_time = 0
    small_time = 0

    # print(f'{fin_prompt_seq.shape = }')

    with contexttimer.Timer() as sps_time:
        while n < max_new_tokens:
            n_orig = n
            N = fin_prompt_seq[0].shape[-1]

            fin_prompt_seq = fin_prompt_seq.to('cuda:1').detach()
            with contexttimer.Timer() as t:
                draft_outputs = draft_model._generate_with_kvcache(fin_prompt_seq, lookahead)
            small_num += lookahead
            small_time += t.elapsed

            # print(f'{draft_model._prob_history.shape = }')

            if debug:
                print(f"Possible continuations: {tokenizer.decode(draft_outputs[0,n_orig:], skip_special_tokens=True)}")

            draft_outputs = draft_outputs.to('cuda:0').detach()
            with contexttimer.Timer() as t:
                # 调用一次target_model
                target_model._forward_with_kvcache(draft_outputs)
            large_num += 1
            large_time += t.elapsed

            # print(f'{target_model._prob_history.shape = }')

            target_model_distribution = target_model._prob_history[:, -lookahead-1:, :]
            draft_model_distribution = draft_model._prob_history[:, -lookahead:, :]

            accepted_flag = 1
            accepted_num = 0
            
            for t in range(lookahead):  # K 4
                numerator = target_model_distribution[:, t, draft_outputs[0, N+t]]  # p(x)
                denominator = draft_model_distribution[:, t, draft_outputs[0, N+t]] # q(x)
                numerator = numerator.to('cuda:1')
                ratio = (numerator / denominator)
                uniform_distribution = torch.rand_like(numerator)
                ones_tensor = torch.ones_like(numerator)

                # Rejection Sampling
                ## Acceptance
                if (uniform_distribution < torch.min(ones_tensor, ratio)).any():
                    draft_outputs = draft_outputs.to('cuda:1')
                    fin_prompt_seq = torch.concat([fin_prompt_seq, draft_outputs[:, N+t].unsqueeze(dim=-1)], dim=-1)
                    n += 1
                    accepted_num += 1

                ## Rejection
                else:
                    target_model_distribution = target_model_distribution.to('cuda:1')
                    new_dist = (target_model_distribution[:, t, :] - draft_model_distribution[:, t, :])
                    new_dist = torch.max(torch.zeros_like(new_dist), new_dist)
                    new_dist = new_dist / new_dist.sum(dim=-1, keepdim=True)
                    token_id = torch.multinomial(new_dist, num_samples=1)
                    fin_prompt_seq = torch.concat([fin_prompt_seq, token_id], dim=-1)
                    accepted_flag = 0
                    target_model.rollback(n)
                    draft_model.rollback(n)
                    break

            if accepted_flag == 1:
                sample_token = sample(target_model_distribution[:, -1, :], temperature=temperature)
                sample_token = sample_token.to('cuda:1')
                fin_prompt_seq = torch.concat([fin_prompt_seq, sample_token], dim=-1)
            
            if debug:
                print(f"Accepted continuations: {tokenizer.decode(fin_prompt_seq[0,n_orig:], skip_special_tokens=True)}")
            
            # print(f'{accepted_num = }')

            n += 1

            # print(f'{fin_prompt_seq.shape = }')
            # print(f'{target_model._prob_history.shape = }')
            # print(f'{draft_model._prob_history.shape = }')

    # print(f'{large_time = }')
    # print(f'{small_time = }')
    # sps_time = sps_time.elapsed
    # print(f'{sps_time = }')
    # print(f'{small_num =}')
    # print(f'{large_num =}')

    return fin_prompt_seq