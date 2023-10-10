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
    fin_prompt_seq = initial_prompt_seq.detach().clone()
    out_logits = []

    for _ in range(new_tokens):
        sample_token_logits = model(fin_prompt_seq).logits[:, -1, :]
        sample_token = sample(sample_token_logits, temperature=temperature)
        fin_prompt_seq = torch.concat([fin_prompt_seq, sample_token], dim=-1)
        out_logits.append(sample_token_logits)

    out_logits = torch.stack(out_logits, dim=1)
    return fin_prompt_seq, out_logits
    