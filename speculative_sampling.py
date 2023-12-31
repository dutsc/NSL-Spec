import torch
from utils import sample_from_draft_model, get_distribution, sample
from transformers import AutoTokenizer



def speculative_sampling(target_model, draft_model, initial_prompt_seq, max_new_tokens, tokenizer, lookahead=4, temperature=1.0, debug=True):
    '''
    Implementation of Algorithm 2 of the paper - Accelerating Large Language Model Decoding 
    with Speculative Sampling (https://arxiv.org/abs/2302.01318)
    '''
    # assert initial_prompt_seq.shape[0] == 1, 'Batch size should be 1'

    n = initial_prompt_seq[0].shape[-1]
    
    # 循环赋值 
    fin_prompt_seq = [prompt.detach().clone() for prompt in initial_prompt_seq]
    # 在第一个batch维度 concat成一个大Tensor
    fin_prompt_seq = torch.cat(fin_prompt_seq, dim=0) # 此时已经是张量 batch形式的
    print(f"【INFO】concatenated_tensor.shape:{fin_prompt_seq.shape}")  # (batch_size, min_token_len(after truncation)))  (8,17)

    while n < max_new_tokens:
        n_orig = n
        N = fin_prompt_seq[0].shape[-1]
        import time
        t1 = time.time()
        draft_outputs, draft_logits = sample_from_draft_model(draft_model, fin_prompt_seq, new_tokens=lookahead, temperature=temperature)
        t2 = time.time()
        print(f"【INFO】sample_from_draft_model time:{(t2-t1)*1000} ms.")
        if debug:
            print(f"Possible continuations: {tokenizer.decode(draft_outputs[0,n_orig:], skip_special_tokens=True)}")

        # 调用一次target_model
        t3 = time.time()
        target_logits = target_model(draft_outputs).logits[:, -lookahead-1:, :]
        t4 = time.time()
        print(f"【INFO】target_model infer time:{(t4-t3)*1000} ms.")

    
        target_model_distribution = get_distribution(target_logits, temperature)
        print(f"【INFO】target_model_distribution.shape:{target_model_distribution.shape}") # [8, 5, 50272]
        draft_model_distribution = get_distribution(draft_logits, temperature)

        accepted_flag = 1
        t5 = time.time()
        for t in range(lookahead):  # K 4
            numerator = target_model_distribution[:, t, draft_outputs[0, N+t]]  # p(x)
            denominator = draft_model_distribution[:, t, draft_outputs[0, N+t]] # q(x)
            ratio = (numerator / denominator)
            uniform_distribution = torch.rand_like(numerator)
            ones_tensor = torch.ones_like(numerator)

            # Rejection Sampling
            ## Acceptance
            if (uniform_distribution < torch.min(ones_tensor, ratio)).any():
                fin_prompt_seq = torch.concat([fin_prompt_seq, draft_outputs[:, N+t].unsqueeze(dim=-1)], dim=-1)
                n += 1

            ## Rejection
            else:
                print(f"【INFO】target_model_distribution[:, t, :]:{target_model_distribution[:, t, :]}")
                print(f"【INFO】draft_model_distribution[:, t, :]:{draft_model_distribution[:, t, :]}")
                new_dist = (target_model_distribution[:, t, :] - draft_model_distribution[:, t, :])
                print(f"【INFO】new_dist:{new_dist}")
                new_dist = torch.max(torch.zeros_like(new_dist), new_dist)
                new_dist = new_dist / new_dist.sum(dim=-1, keepdim=True)
                token_id = torch.multinomial(new_dist, num_samples=1)
                fin_prompt_seq = torch.concat([fin_prompt_seq, token_id], dim=-1)
                accepted_flag = 0
                break
        t6 = time.time()
        print(f"【INFO】Rejection Sampling lookahead times time:{(t6-t5)*1000} ms.")
        
        if accepted_flag == 1:
            sample_token = sample(target_logits[:, -1, :], temperature=temperature)
            fin_prompt_seq = torch.concat([fin_prompt_seq, sample_token], dim=-1)
        
        if debug:
            print(f"Accepted continuations: {tokenizer.decode(fin_prompt_seq[0,n_orig:], skip_special_tokens=True)}")

        n += 1
        t7 = time.time()
        print(f"【INFO】one token generate time:{(t7-t1)*1000} ms.")
        print("-----------------------end--------------------")
    tt = time.time()
    return fin_prompt_seq


# 0.0204  0.022   target   包含迁移时间->  0.057     不包含迁移时间-> 0.043 0.044
# 0.0152  0.018   small    包含迁移时间->  0.0186    不包含迁移时间-> 0.014 0.014

# 单独
# 0.0185  0.019   target
# 0.0077  0.008   small