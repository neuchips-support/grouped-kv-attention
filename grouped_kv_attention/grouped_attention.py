import math
import torch
import torch.nn as nn
import torch.nn.functional as F



from typing import Optional


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class GroupedKVAttention:
    def __init__(self, config=None, *, num_heads=None, num_kv_heads=None):

        self.hidden_size = config.hidden_size
        

        if config:
            self.num_heads = config.num_attention_heads
            self.num_key_value_heads = config.num_key_value_heads
        elif num_heads and num_kv_heads:
            self.num_heads = num_heads
            self.num_key_value_heads = num_kv_heads
        else:
            raise ValueError("Must provide either config or both num_heads and num_kv_heads.")

        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)

        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.selected_query_idx = self._build_query_index()

    def _build_query_index(self):
        if self.num_key_value_groups == 1:
            return None
        idx = []
        for i in range(self.num_key_value_groups):
            selected = [t * self.num_key_value_groups + i for t in range(self.num_key_value_heads)]
            idx.append(torch.IntTensor(selected))
        return idx

    def grouped_kv_attention_forward(
            self,
        module: nn.Module,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        dropout: float = 0.0,
        **kwargs,
    ):
        

        if self.num_key_value_groups != 1 and query_states.shape[2] == 1:
            # print("kevin attn qk")
            attn_weights = torch.zeros(query_states.shape[0], query_states.shape[1], query_states.shape[2], key_states.shape[2])
            
            for i in range(0, self.num_key_value_heads):
                start_idx = i*self.num_key_value_groups
                attn_weights[:, start_idx:start_idx+self.num_key_value_groups, :, :] = torch.matmul(query_states[:, start_idx:start_idx+self.num_key_value_groups, :, :], key_states[:, i, :, :].transpose(1, 2)) / math.sqrt(self.head_dim)
        else:
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
       

        
        if self.num_key_value_groups != 1 and query_states.shape[2] == 1:
            # print("kevin_attn")
            attn_output = torch.zeros(attn_weights.shape[0], attn_weights.shape[1], attn_weights.shape[2], value_states.shape[3])
        
            for i in range(0, self.num_key_value_heads):
                start_idx = i*self.num_key_value_groups
                attn_output[:, start_idx:start_idx+self.num_key_value_groups, :, :] = torch.matmul(attn_weights[:, start_idx:start_idx+self.num_key_value_groups, :, :], value_states[:, i, :, :]) 
        else:
            attn_output = torch.matmul(attn_weights, value_states)
        
        attn_output = attn_output.transpose(1, 2).contiguous()

        return attn_output, attn_weights