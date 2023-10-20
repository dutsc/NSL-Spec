import torch
from utils import get_distribution, sample

class KVCacheModel():
    def __init__(self, model : torch.nn.Module, temperature : float = 0) -> None:
        self._model = model
        self._past_key_values = None
        self._prob_history = None
        self._temperature = temperature

    @torch.no_grad()
    def _forward_with_kvcache(self, input_ids : torch.Tensor) -> torch.Tensor:
        if self._past_key_values is None:
            # outputs
            outputs = self._model(input_ids) 
            # logits : tensor(batch_size * seq_len * vocab_size)
            self._prob_history = get_distribution(outputs.logits, self._temperature)
            # _past_key_values : n_layer * 2 * tensor(batch * head * seq_len * hidden_dim)
            self._past_key_values = outputs.past_key_values
            # last_token
            last_token = sample(outputs.logits[:, -1, :], self._temperature)
        else:
            cached_len = self._past_key_values[0][0].shape[2]
            # print(f'{input_ids.shape = }')
            # print(f'{cached_len = }')
            last_input_id = input_ids[:, cached_len:]
            # print(f'{last_input_id = }')
            # outputs
            outputs = self._model(last_input_id, past_key_values=self._past_key_values, use_cache=True)
            # not_cached_q
            not_cached_q = get_distribution(outputs.logits, self._temperature)
            # 然后将其加入到概率存储中
            # _past_key_values
            self._prob_history = torch.cat([self._prob_history, not_cached_q], dim=1)
            self._past_key_values = outputs.past_key_values
            # last_token
            last_token = sample(outputs.logits[:, -1, :], self._temperature)
        return last_token

    @torch.no_grad()
    def _generate_with_kvcache(self, prefix : torch.Tensor, gamma : int) -> torch.Tensor:
        x = prefix
        for _ in range(gamma):
            next_tok = self._forward_with_kvcache(x)
            x = torch.cat((x, next_tok), dim=1)
        return x
    
    @torch.no_grad()
    def rollback(self, end_pos : int):
        past_key_values_trimmed = []
        assert self._past_key_values
        for kv in self._past_key_values:
            k, v = kv
            k = k[:, :, :end_pos, :]
            v = v[:, :, :end_pos, :]
            kv_trimmed = (k, v)
            past_key_values_trimmed.append(kv_trimmed)
        
        self._past_key_values = past_key_values_trimmed
        self._prob_history = self._prob_history[:, :end_pos, :]
    
