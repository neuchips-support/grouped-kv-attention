from grouped_kv_attention.grouped_attention import GroupedKVAttention

def patch_llama_attention(config=None):
    from transformers.models.llama import modeling_llama

    modeling_llama.eager_attention_forward = GroupedKVAttention(config).grouped_kv_attention_forward

    original_init = modeling_llama.LlamaAttention.__init__
    
    def patched_init(self, config, layer_idx):
        original_init(self, config, layer_idx)
        self.config._attn_implementation = "eager"

    modeling_llama.LlamaAttention.__init__ = patched_init

def patch_mistral_attention(config=None):
    from transformers.models.mistral import modeling_mistral

    modeling_mistral.eager_attention_forward = GroupedKVAttention(config).grouped_kv_attention_forward

    original_init = modeling_mistral.MistralAttention.__init__
    
    def patched_init(self, config, layer_idx):
        original_init(self, config, layer_idx)
        self.config._attn_implementation = "eager"
    modeling_mistral.MistralAttention.__init__ = patched_init


def patch_gemma2_attention(config=None):
    from transformers.models.gemma2 import modeling_gemma2

    modeling_gemma2.eager_attention_forward = GroupedKVAttention(config).grouped_kv_attention_forward

    original_init = modeling_gemma2.Gemma2Attention.__init__
    
    def patched_init(self, config, layer_idx):
        original_init(self, config, layer_idx)
        self.config._attn_implementation = "eager"
    modeling_gemma2.Gemma2Attention.__init__ = patched_init

def patch_gemma3_attention(config=None):
    from transformers.models.gemma3 import modeling_gemma3

    modeling_gemma3.eager_attention_forward = GroupedKVAttention(hidden_size=config.text_config.hidden_size, 
                                                                 num_heads=config.text_config.num_attention_heads, 
                                                                 num_kv_heads=config.text_config.num_key_value_heads).grouped_kv_attention_forward

    original_init = modeling_gemma3.Gemma3Attention.__init__
    
    def patched_init(self, config, layer_idx):
        original_init(self, config, layer_idx)
        self.config._attn_implementation = "eager"
    modeling_gemma3.Gemma3Attention.__init__ = patched_init
