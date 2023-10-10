import torch

def get_distribution(logits, temperature):
    probs = torch.softmax(logits / (temperature + 1e-10), dim=-1)
    return probs

def sample(logits, temperature):
    probs = get_distribution(logits, temperature)
    
    multinomial = torch.multinomial(probs, num_samples=1)
    print(f"【INFO】torch.multinomial(probs, num_samples=1):{multinomial}")
    print(f"【INFO】torch.multinomial(probs, num_samples=1) shape:{multinomial.shape}")
    return multinomial

    # return torch.multinomial(probs, num_samples=1)[0]

def sample_from_draft_model(model, initial_prompt_seq, new_tokens, temperature=1.0):
    # 此时initial_prompt_seq已经是一个batch形式的Tensor
    fin_prompt_seq = initial_prompt_seq.detach().clone()
    # # 循环赋值
    # fin_prompt_seq = [prompt.detach().clone() for prompt in initial_prompt_seq]
    # # 难道应该concat成一个大Tensor，输入model?
    # fin_prompt_seq = torch.cat(fin_prompt_seq, dim=0)
    # print(f"【INFO】concatenated_tensor.shape:{fin_prompt_seq.shape}")  # (batch_size, min_token_len(after truncation)))  (8,17)
    out_logits = []

    for _ in range(new_tokens):
        # first_output = model(fin_prompt_seq[0])
        # print(f"【INFO】first_output.logits.shape:{first_output.logits.shape}")  # (1,35,50272)
        # print(f"【INFO】first_output:{first_output}")
        sample_token_logits = model(fin_prompt_seq).logits[:, -1, :]  # (batch_size, sequence_length, config.vocab_size)
        sample_token = sample(sample_token_logits, temperature=temperature)
        fin_prompt_seq = torch.concat([fin_prompt_seq, sample_token], dim=-1)
        out_logits.append(sample_token_logits)

    out_logits = torch.stack(out_logits, dim=1)
    return fin_prompt_seq, out_logits
    