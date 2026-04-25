"""
The most atomic way to train and run inference for a GPT in pure, dependency-free Python.
This file is the complete algorithm.
Everything else is just efficiency.

@karpathy

Extended for MASC 515 Assignment 3:
Added implementations of GELU, LoRA, RoPE (Rotary Positional Embedding), and Mixture of Experts (MoE).
"""

import os    # os.path.exists
import math  # math.log, math.exp
import random  # random.seed, random.choices, random.gauss, random.shuffle
random.seed(42)  # Let there be order among chaos

# Let there be a Dataset `docs`: list[str] of documents (e.g. a list of names)
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')
docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")

# Let there be a Tokenizer to translate strings to sequences of integers ("tokens") and back
uchars = sorted(set(''.join(docs)))  # unique characters in the dataset become token ids 0..n-1
BOS = len(uchars)   # token id for a special Beginning of Sequence (BOS) token
vocab_size = len(uchars) + 1  # total number of unique tokens, +1 is for BOS
print(f"vocab size: {vocab_size}")

# Let there be Autograd to recursively apply the chain rule through a computation graph
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')  # Python optimization for memory usage

    def __init__(self, data, children=(), local_grads=()):
        self.data = data          # scalar value of this node calculated during forward pass
        self.grad = 0             # derivative of the loss w.r.t. this node, calculated in backward pass
        self._children = children  # children of this node in the computation graph
        self._local_grads = local_grads  # local derivative of this node w.r.t. its children

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))
    def log(self): return Value(math.log(self.data), (self,), (1/self.data,))
    def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad

# ---------------------------------------------------------------------------
# ALGORITHM 1: GELU (Gaussian Error Linear Unit)
# Paper: https://arxiv.org/abs/1606.08415
#
# Underlying idea:
#   Standard ReLU activation simply zeroes out negative values. GELU improves on
#   this by multiplying each input by the probability that it is positive under a
#   Gaussian (normal) distribution. Concretely:
#       GELU(x) = x * Phi(x)
#   where Phi(x) = 0.5 * (1 + erf(x / sqrt(2))) is the cumulative distribution
#   function of the standard normal.  A fast approximation (used in GPT-2) is:
#       GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
#   Because neurons are stochastically gated rather than hard-clipped, gradients
#   flow more smoothly during training, which typically leads to faster convergence.
# ---------------------------------------------------------------------------
def gelu(x):
    """GELU activation: smooth, probabilistic alternative to ReLU."""
    # tanh approximation of GELU
    inner = (Value(2.0 / math.pi) ** 0.5) * (x + Value(0.044715) * x ** 3)
    # tanh(z) = (exp(2z) - 1) / (exp(2z) + 1)
    e2 = (inner * 2).exp()
    tanh_val = (e2 - 1) * (e2 + 1) ** -1
    return x * (Value(1.0) + tanh_val) * Value(0.5)

# ---------------------------------------------------------------------------
# ALGORITHM 2: LoRA (Low-Rank Adaptation)
# Paper: https://arxiv.org/abs/2106.09685
#
# Underlying idea:
#   Fine-tuning all parameters of a large pre-trained model is expensive.
#   LoRA observes that the weight updates needed for a new task tend to lie in a
#   low-dimensional subspace.  Instead of updating a full weight matrix W (of
#   rank r_full), LoRA freezes W and adds a trainable bypass:
#       W' = W + B * A
#   where A is (rank x in_features) and B is (out_features x rank), with rank
#   much smaller than either dimension.  Only A and B are trained, reducing
#   trainable parameters dramatically while preserving most of the model's
#   expressiveness.  The output is scaled by alpha/rank to keep magnitudes stable.
# ---------------------------------------------------------------------------
class LoRALinear:
    """
    A linear layer with a LoRA adapter.
    The base weight W is frozen; only the low-rank matrices A and B are trained.
    """
    def __init__(self, w_base, rank=2, alpha=1.0):
        self.w_base = w_base           # frozen base weight (list of list of Value)
        out_features = len(w_base)
        in_features  = len(w_base[0])
        self.rank  = rank
        self.scale = alpha / rank      # scaling factor

        # A: (rank x in_features), B: (out_features x rank) — both initialised near zero
        self.A = [[Value(random.gauss(0, 0.01)) for _ in range(in_features)]  for _ in range(rank)]
        self.B = [[Value(0.0)                   for _ in range(rank)]          for _ in range(out_features)]

    def __call__(self, x):
        # Base linear pass (frozen)
        base_out = [sum(wij * xi for wij, xi in zip(row, x)) for row in self.w_base]
        # LoRA bypass: B * (A * x) * scale
        ax = [sum(aij * xi for aij, xi in zip(row, x)) for row in self.A]
        bax = [sum(bij * axj for bij, axj in zip(row, ax)) for row in self.B]
        return [b + Value(self.scale) * d for b, d in zip(base_out, bax)]

# ---------------------------------------------------------------------------
# ALGORITHM 3: RoPE (Rotary Positional Embedding)
# Paper: https://arxiv.org/abs/2104.09864
#
# Underlying idea:
#   Transformers need to know the position of each token in a sequence. The
#   original GPT adds a learned position embedding to the token embedding.
#   RoPE encodes position directly into the query and key vectors of attention
#   by *rotating* consecutive pairs of dimensions by an angle proportional to
#   the token position:
#       [x_{2i}, x_{2i+1}]  →  [x_{2i} cos(θ) − x_{2i+1} sin(θ),
#                                x_{2i} sin(θ) + x_{2i+1} cos(θ)]
#   where θ = pos / (10000^(2i/d)).
#   Because a dot-product between a query at position m and a key at position n
#   then depends only on (m − n), the model learns relative positions naturally
#   without needing extra embedding parameters.  This generalises better to
#   sequences longer than those seen during training.
# ---------------------------------------------------------------------------
def rope_rotate(x, pos, base=10000):
    """
    Apply Rotary Positional Embedding to vector x at sequence position pos.
    x must have even length.
    """
    d = len(x)
    out = list(x)  # copy
    for i in range(0, d, 2):
        theta = pos / (base ** (i / d))
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        x0, x1 = x[i], x[i + 1]
        out[i]     = x0 * Value(cos_t) - x1 * Value(sin_t)
        out[i + 1] = x0 * Value(sin_t) + x1 * Value(cos_t)
    return out

# ---------------------------------------------------------------------------
# ALGORITHM 4: Mixture of Experts (MoE)
# Reference: https://huggingface.co/blog/moe
#
# Underlying idea:
#   Instead of running every token through the same feed-forward network (FFN),
#   MoE maintains multiple FFN "experts" and a lightweight gating network.  For
#   each token the gate selects the top-k experts (commonly k=1 or k=2), routes
#   the token only to those experts, and combines their outputs weighted by the
#   gate scores.  Because only a fraction of parameters is active per token, MoE
#   allows the total parameter count to grow (more capacity / knowledge) without
#   a proportional increase in FLOPs per forward pass.  This is the architecture
#   behind models such as Mixtral and GPT-4 (reportedly).
# ---------------------------------------------------------------------------
class MoELayer:
    """
    Sparse Mixture-of-Experts feed-forward layer.
    Each expert is a simple one-hidden-layer MLP using GELU activation.
    The router picks the top-k experts for each token.
    """
    def __init__(self, n_embd, num_experts=4, top_k=2):
        self.top_k = top_k
        self.num_experts = num_experts
        # Each expert: two weight matrices (expand then contract)
        hidden = 4 * n_embd
        matrix = lambda r, c: [[Value(random.gauss(0, 0.08)) for _ in range(c)] for _ in range(r)]
        self.experts_fc1 = [matrix(hidden,  n_embd) for _ in range(num_experts)]
        self.experts_fc2 = [matrix(n_embd,  hidden) for _ in range(num_experts)]
        # Router / gate: maps n_embd → num_experts scores
        self.gate_w = matrix(num_experts, n_embd)

    def __call__(self, x):
        # Compute gate logits and softmax probabilities
        gate_logits = [sum(wij * xi for wij, xi in zip(row, x)) for row in self.gate_w]
        gate_probs  = softmax(gate_logits)

        # Select top-k experts by gate probability
        scores = [(gate_probs[i].data, i) for i in range(self.num_experts)]
        scores.sort(reverse=True)
        top_k_idx = [idx for _, idx in scores[:self.top_k]]

        # Weighted sum of selected experts' outputs
        out = [Value(0.0)] * len(x)
        weight_sum = sum(gate_probs[i] for i in top_k_idx)
        for i in top_k_idx:
            # Expert forward pass with GELU activation
            h = [sum(wij * xi for wij, xi in zip(row, x))  for row in self.experts_fc1[i]]
            h = [gelu(hi) for hi in h]
            h = [sum(wij * hi for wij, hi in zip(row, h))  for row in self.experts_fc2[i]]
            w = gate_probs[i] / weight_sum   # normalised gate weight
            out = [o + w * hi for o, hi in zip(out, h)]
        return out

# Initialize the parameters, to store the knowledge of the model
n_layer = 1   # depth of the transformer neural network (number of layers)
n_embd  = 16  # width of the network (embedding dimension)
block_size = 16  # maximum context length of the attention window
n_head  = 4   # number of attention heads
head_dim = n_embd // n_head  # derived dimension of each head
matrix  = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
state_dict = {'wte': matrix(vocab_size, n_embd), 'wpe': matrix(block_size, n_embd), 'lm_head': matrix(vocab_size, n_embd)}
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)

# Instantiate add-on modules (LoRA on the output projection, MoE as alternate FFN demo)
lora_layers = {f'layer{i}.attn_wo': LoRALinear(state_dict[f'layer{i}.attn_wo'], rank=2) for i in range(n_layer)}
moe_layers  = {f'layer{i}.moe': MoELayer(n_embd, num_experts=4, top_k=2) for i in range(n_layer)}

params = [p for mat in state_dict.values() for row in mat for p in row]
# Also include LoRA trainable parameters
for ll in lora_layers.values():
    for row in ll.A: params.extend(row)
    for row in ll.B: params.extend(row)
print(f"num params: {len(params)}")

# Define the model architecture
def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

def gpt(token_id, pos_id, keys, values):
    tok_emb = state_dict['wte'][token_id]
    pos_emb = state_dict['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)

    for li in range(n_layer):
        # 1) Multi-head Attention block (with RoPE on Q and K, LoRA on output projection)
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])

        # Apply RoPE to Q and K so each head dimension encodes relative position
        q = rope_rotate(q, pos_id)
        k = rope_rotate(k, pos_id)

        keys[li].append(k)
        values[li].append(v)
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]
            attn_weights = softmax(attn_logits)
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]
            x_attn.extend(head_out)

        # Use LoRA-adapted output projection instead of plain linear
        x = lora_layers[f'layer{li}.attn_wo'](x_attn)
        x = [a + b for a, b in zip(x, x_residual)]

        # 2) MoE Feed-Forward block (replaces standard MLP)
        x_residual = x
        x = rmsnorm(x)
        x = moe_layers[f'layer{li}.moe'](x)   # sparse mixture-of-experts FFN
        x = [a + b for a, b in zip(x, x_residual)]

    logits = linear(x, state_dict['lm_head'])
    return logits

# Let there be Adam, the blessed optimizer and its buffers
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m = [0.0] * len(params)
v_buf = [0.0] * len(params)

num_steps = 1000
for step in range(num_steps):
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)

    keys_cache, values_cache = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = gpt(token_id, pos_id, keys_cache, values_cache)
        probs = softmax(logits)
        loss_t = -probs[target_id].log()
        losses.append(loss_t)
    loss = (1 / n) * sum(losses)

    loss.backward()

    lr_t = learning_rate * (1 - step / num_steps)
    for i, p in enumerate(params):
        m[i]    = beta1 * m[i]    + (1 - beta1) * p.grad
        v_buf[i] = beta2 * v_buf[i] + (1 - beta2) * p.grad ** 2
        m_hat   = m[i]    / (1 - beta1 ** (step + 1))
        v_hat   = v_buf[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        p.grad  = 0

    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}", end='\r')

# Inference
temperature = 0.5
print("\n--- inference (new, hallucinated names) ---")
for sample_idx in range(20):
    keys_cache, values_cache = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    token_id = BOS
    sample = []
    for pos_id in range(block_size):
        logits = gpt(token_id, pos_id, keys_cache, values_cache)
        probs = softmax([l / temperature for l in logits])
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == BOS:
            break
        sample.append(uchars[token_id])
    print(f"sample {sample_idx+1:2d}: {''.join(sample)}")
