<div align="center">
  <img src="https://github.com/neuchips-support/grouped-kv-attention/blob/main/repo-logo.png" width="ˋ480">
</div>

**grouped-kv-attention** is a lightweight monkey patch for HuggingFace Transformers that replaces the default `eager_attention_forward` with an optimized grouped attention implementation.

## ✨ Features

- 🚀 Speeds up token-by-token generation on CPU (especially without AVX512)
- 🧩 Compatible with LLaMA, Mistral, and Gemma models
- 🔧 Drop-in replacement: no need to modify model internals
- 📦 Minimal dependencies: only `torch` and `transformers`

## 📦 Installation

```bash
git clone https://github.com/neuchips-support/grouped-kv-attention.git
cd grouped-kv-attention
pip install .
```

## 🔧 Usage

```python
from grouped_kv_attention import patch_llama_attention

from transformers import  AutoConfig, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
config = AutoConfig.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

patch_llama_attention(config)
```



## 🧠 Supported Models

| Model   | Class Name         | Patch Function              |
|---------|--------------------|-----------------------------|
| LLaMA   | `LlamaAttention`   | `patch_llama_attention()`   |
| Mistral | `MistralAttention` | `patch_mistral_attention()` |
| Gemma2  | `Gemma2Attention`  | `patch_gemma2_attention()`  |

---

Feel free to fork, contribute, or integrate this into your own transformer stack.
