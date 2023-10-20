import torch
from utils import sample

@torch.no_grad() # 不加会显存不足，加上能提高约10%的吞吐率
def autoregressive_sampling(model, initial_prompt_seq, target_len, temperature=1.0, cache=True):
    # 此时initial_prompt_seq已经是一个batch形式的Tensor
    n = initial_prompt_seq[0].shape[-1]
    
    # 循环赋值 
    fin_prompt_seq = [prompt.detach().clone() for prompt in initial_prompt_seq]
    # 在第一个batch维度 concat成一个大Tensor
    fin_prompt_seq = torch.cat(fin_prompt_seq, dim=0) # 此时已经是张量 batch形式的
    fin_prompt_seq = fin_prompt_seq.to('cuda:0').detach()
    # print(f"【INFO】concatenated_tensor.shape:{fin_prompt_seq.shape}")  # (batch_size, min_token_len(after truncation)))  (8,17)
    
    large_num = 0

    import contexttimer
    with contexttimer.Timer() as as_time:

        # 实现kv cache后提速1.5倍
        past_key_values = None
        while n < target_len:
            # print('[loop]')
            if past_key_values:
                outputs = model(fin_prompt_seq[:, -1:], past_key_values = past_key_values, use_cache = True)
            else:
                outputs = model(fin_prompt_seq)

            sample_token_logits = outputs.logits[:, -1, :]
            sample_token = sample(sample_token_logits, temperature=temperature)
            fin_prompt_seq = torch.concat([fin_prompt_seq, sample_token], dim=-1)

            if cache:
                past_key_values = outputs.past_key_values
            n += 1
            large_num += 1

    print(f'{large_num = }')
    as_time = as_time.elapsed
    print(f'{as_time = }')
    return fin_prompt_seq