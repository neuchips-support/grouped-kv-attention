# grouped-kv-attention
grouped-kv-attention is a lightweight monkey patch for HuggingFace Transformers that replaces the default `eager_attention_forward` with an optimized grouped attention implementation, especially for CPUs without AVX512
