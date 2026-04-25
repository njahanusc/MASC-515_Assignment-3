# MASC 515 Assignment 3 - microGPT

## Overview

This assignment extends microGPT — a minimal, dependency-free Python implementation of a GPT-like model created by Andrej Karpathy. The original code (~243 lines of pure Python) trains a character-level language model from scratch with no external libraries. Four advanced algorithms used in modern large language models have been implemented and integrated into the codebase.

---

## Algorithms Implemented

### 1. GELU (Gaussian Error Linear Unit)
**Paper:** https://arxiv.org/abs/1606.08415

**Underlying Idea:**  
Standard ReLU activation hard-clips all negative values to zero. GELU improves on this by instead *probabilistically* gating each input: it multiplies the value by the probability that it is positive under a standard Gaussian (normal) distribution.

Mathematically:  
`GELU(x) = x · Φ(x)`  
where `Φ(x) = 0.5 · (1 + erf(x / √2))` is the Gaussian cumulative distribution function.

A fast approximation (used in GPT-2) is:  
`GELU(x) ≈ 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))`

Because neurons are softly gated rather than hard-clipped, gradients flow more smoothly during backpropagation, which typically leads to better and faster training compared to ReLU. GELU is the default activation in GPT-2, BERT, and most modern transformers.

---

### 2. LoRA (Low-Rank Adaptation)
**Paper:** https://arxiv.org/abs/2106.09685

**Underlying Idea:**  
Fine-tuning all parameters of a large pre-trained model is expensive — it requires storing and updating billions of values. LoRA addresses this by observing that weight updates during fine-tuning tend to have low *intrinsic rank*: the important changes lie in a small subspace.

Instead of modifying the full weight matrix `W`, LoRA freezes it and inserts a trainable low-rank bypass:  
`W' = W + B · A`  
where `A` is a small `(rank × input)` matrix and `B` is a small `(output × rank)` matrix. Since `rank << min(input, output)`, the number of trainable parameters is drastically reduced (e.g., from millions to thousands), while the model retains most of its pre-trained knowledge. The output is scaled by `alpha/rank` to keep activation magnitudes stable.

In this implementation, LoRA is applied to the attention output projection of each transformer layer.

---

### 3. RoPE (Rotary Positional Embedding)
**Paper:** https://arxiv.org/abs/2104.09864

**Underlying Idea:**  
Transformers need to know the position of each token in a sequence. The original approach (used in GPT-2) adds a learned embedding vector to each token. RoPE takes a different approach: it encodes position directly into the *query* and *key* vectors inside the attention mechanism by **rotating** them.

Specifically, consecutive pairs of dimensions `(x_{2i}, x_{2i+1})` are rotated by an angle proportional to the token's position `pos` and the dimension index `i`:  
`θ = pos / (10000^(2i/d))`

The rotation is:  
`[x cos θ − y sin θ,   x sin θ + y cos θ]`

A crucial property: the dot-product between a query at position `m` and a key at position `n` depends only on the *relative* offset `m − n`, not their absolute positions. This means the model naturally learns relative position information, and generalises better to sequence lengths it has not seen during training. RoPE is used in LLaMA, Mistral, and many other modern LLMs.

---

### 4. Mixture of Experts (MoE)
**Reference:** https://huggingface.co/blog/moe

**Underlying Idea:**  
A standard transformer feed-forward block runs every token through the same network. Mixture of Experts replaces this with multiple parallel feed-forward networks ("experts") and a lightweight **gating (router) network**.

For each token, the router computes a score for every expert and selects only the top-k experts (commonly k=1 or k=2). The token is processed only by those selected experts, and their outputs are combined using the gate scores as weights:  
`output = Σ (gate_score_i · expert_i(x))`  for the top-k experts.

Because only a small fraction of parameters are active for any given token, MoE allows the total number of model parameters to be scaled up dramatically — giving the model more capacity and knowledge — without a proportional increase in computation per forward pass. MoE is the architecture behind Mixtral, and is widely believed to be used in GPT-4.

In this implementation, the standard MLP block in each transformer layer is replaced with a 4-expert MoE layer using top-2 routing and GELU activations inside each expert.

---

## Notes

- The original microGPT code structure by Andrej Karpathy was preserved. The four algorithms were added as clearly documented extensions.
- Source: https://karpathy.github.io/2026/02/12/microgpt/
