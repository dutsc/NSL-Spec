import torch
from utils import sample

def autoregressive_sampling(model, initial_prompt_seq, target_len, temperature=1.0):
    # 此时initial_prompt_seq已经是一个batch形式的Tensor
    n = initial_prompt_seq[0].shape[-1]
    
    # 循环赋值 
    fin_prompt_seq = [prompt.detach().clone() for prompt in initial_prompt_seq]
    # 在第一个batch维度 concat成一个大Tensor
    fin_prompt_seq = torch.cat(fin_prompt_seq, dim=0) # 此时已经是张量 batch形式的
    # print(f"【INFO】concatenated_tensor.shape:{fin_prompt_seq.shape}")  # (batch_size, min_token_len(after truncation)))  (8,17)
    
    large_num = 0

    import contexttimer
    with contexttimer.Timer() as as_time:
        while n < target_len:
            sample_token_logits = model(fin_prompt_seq).logits[:, -1, :]
            # print(f"【INFO】sample_token_logits:{sample_token_logits}")
            
            sample_token = sample(sample_token_logits, temperature=temperature)
            # print(f"【INFO】sample_token:{sample_token}")
            
            fin_prompt_seq = torch.concat([fin_prompt_seq, sample_token], dim=-1)
            n += 1
            large_num += 1

    print(f'{large_num = }')
    print(f'{as_time}')
    return fin_prompt_seq