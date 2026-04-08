# Lesson 3: The KV Cache - Where the computer will live

## Concepts to learn

### The problem with naive generation

During training, we process the whole sequence in parallel - one forward pass handles all 256 tokens at once. But during generation, we produce tokens one at a time:

```
Step 1: "The"           → predict next → "cat"
Step 2: "The cat"       → predict next → "sat"
Step 3: "The cat sat"   → predict next → "on"
```

The naive approach recomputes *everything* at each step. At step 3, we recompute Q, K, V for "The", "cat", and "sat" - even though we already computed K and V for "The" and "cat" in steps 1 and 2. For a sequence of length n, we do O(n²) work total. That's wasteful.

### The key insight: K and V don't change

Look at the attention computation for token 3 ("sat"):

```
Q3 attends to K1, K2, K3
```

When we later process token 4 ("on"):

```
Q4 attends to K1, K2, K3, K4
```

K1, K2, K3 are exactly the same both times. The causal mask means future tokens can't affect past computations. So we can compute K and V once and store them.

### The KV cache

We pre-allocate two big tensors, one for keys and one for values:

```
K cache: shape (n_layers, batch_size, max_seq_len, n_heads, head_dim)
V cache: shape (n_layers, batch_size, max_seq_len, n_heads, head_dim)
```

Generation now works like this:

```
Step 1: Process "The"
  - Compute Q1, K1, V1
  - Store K1, V1 in cache at position 0
  - Attention: Q1 @ K_cache[0:1] → weights → weights @ V_cache[0:1]
  - Output: logits → sample → "cat"

Step 2: Process "cat" (only this one token)
  - Compute Q2, K2, V2
  - Store K2, V2 in cache at position 1
  - Attention: Q2 @ K_cache[0:2] → weights → weights @ V_cache[0:2]
  - Output: logits → sample → "sat"

Step 3: Process "sat" (only this one token)
  - Compute Q3, K3, V3
  - Store K3, V3 in cache at position 2
  - Attention: Q3 @ K_cache[0:3] → weights → weights @ V_cache[0:3]
  ...
```

Each step processes exactly 1 token through the network but attends to all previous tokens via the cache. Total work is O(n) per step instead of O(n) per step with O(n) recomputation = O(n²) total.

### Why this matters for the stack computer

Here's where it gets interesting. The KV cache is just a bank of (Key, Value) pairs. Right now, each pair comes from processing a real text token through the network. But **there's nothing preventing us from writing arbitrary values into the cache**.

What if we reserved the first 16 positions of the cache for a stack machine?

```
KV Cache positions:
[0]  [1]  [2]  ... [15]  [16]   [17]   [18]   ...
|--- stack slots ---|     |--- actual text tokens ---|
```

The model's attention queries would attend to positions 0-15 just like any other positions. But instead of containing information derived from text tokens, they contain the current state of a stack computer.

- The **Key** at position 3 encodes "I am stack slot 3" (learned embedding)
- The **Value** at position 3 encodes "the number currently stored in slot 3" (learned projection from a scalar value)

When a stack operation executes (say, PUSH 42), we rewrite the cache entries at the affected positions. The next forward pass sees the updated stack state through attention - no special wiring needed.

The model can't tell the difference between "real" KV entries from text and "synthetic" KV entries from the stack. Attention is attention. This is the elegance of Jefferson's approach.

### RoPE (Rotary Position Embeddings)

The model needs to know token order - "the cat" and "cat the" should produce different attention patterns. There are several ways to encode position:

**Learned position embeddings** (what GPT-2 uses): Learn a separate embedding vector for each position (0, 1, 2, ..., max_len). Add it to the token embedding. Simple, but can't generalize beyond max_len.

**RoPE** (what modern models use): Encode position by *rotating* the Q and K vectors. Each position gets a unique rotation angle. The dot product Q·K then naturally depends on the *relative* distance between positions, not their absolute positions.

The rotation is applied to pairs of dimensions:

```
For dimensions (0,1): rotate by θ₀ * position
For dimensions (2,3): rotate by θ₁ * position
For dimensions (4,5): rotate by θ₂ * position
...
```

Where θᵢ are fixed frequencies (typically θᵢ = 10000^(-2i/d)).

**Why we need RoPE for the stack computer**: With learned positional embeddings, the model has seen specific positions only from text tokens during training. RoPE is more flexible - we can assign any position index to any KV entry. Stack slots get positions 0-15, text starts at 16, and the relative-position-aware attention handles everything correctly.

---

## What we'll build

### Files

| File | Purpose |
|------|---------|
| `nanochat/engine.py` | `KVCache` class: pre-allocated tensors, methods to read/write/advance position. `Engine` class: wraps model + tokenizer for efficient cached generation. |
| `nanochat/model.py` | Updated: RoPE implementation (precompute cos/sin buffers, apply rotation to Q/K). `CausalSelfAttention` gains optional `kv_cache` parameter. |

### KVCache interface

```python
class KVCache:
    def __init__(self, n_layers, max_seq_len, n_heads, head_dim, batch_size=1):
        self.k = torch.zeros(n_layers, batch_size, max_seq_len, n_heads, head_dim)
        self.v = torch.zeros(n_layers, batch_size, max_seq_len, n_heads, head_dim)
        self.pos = 0  # current write position

    def write(self, layer, k, v):
        """Store new K, V at current position for this layer."""
        self.k[layer, :, self.pos] = k
        self.v[layer, :, self.pos] = v

    def read(self, layer):
        """Read all K, V up to current position for this layer."""
        return self.k[layer, :, :self.pos+1], self.v[layer, :, :self.pos+1]

    def advance(self):
        """Move to next position."""
        self.pos += 1
```

### How attention changes with the cache

Without cache (training):
```python
def forward(self, x):
    Q, K, V = self.qkv_proj(x)  # all tokens at once
    # Q, K, V shape: (batch, seq_len, n_heads, head_dim)
    output = scaled_dot_product_attention(Q, K, V, is_causal=True)
    return output
```

With cache (generation):
```python
def forward(self, x, kv_cache=None):
    Q, K, V = self.qkv_proj(x)  # just 1 token
    if kv_cache is not None:
        kv_cache.write(self.layer_idx, K, V)
        K_full, V_full = kv_cache.read(self.layer_idx)
        output = scaled_dot_product_attention(Q, K_full, V_full)
        # no causal mask needed - we only have past tokens in cache
    else:
        output = scaled_dot_product_attention(Q, K, V, is_causal=True)
    return output
```

### How you'll know it works

Generate the same text prompt with and without the KV cache. The output should be **identical** token-for-token (with the same random seed). If even one token differs, there's a bug.

Speed: cached generation should be noticeably faster for longer sequences. For a 200-token generation, expect roughly 3-5x speedup.

---

## Key questions to understand before moving on

1. Why can't we cache Q as well? (Q is specific to the *current* token - it's asking a question about what this particular token needs. Past tokens' queries are irrelevant to the current computation.)

2. Why pre-allocate the cache rather than dynamically growing it? (Pre-allocation avoids repeated memory allocation and copying. GPU memory operations are expensive. With pre-allocation, writing to the cache is just an index operation.)

3. What happens if we modify a cached K or V entry? (The next forward pass will attend to the modified values. The model won't know they were changed - it just sees KV pairs. This is *exactly* how the stack computer will work.)

4. Why does RoPE use fixed frequencies instead of learned ones? (Fixed frequencies give smooth, predictable behavior and generalize better to unseen positions. Learned frequencies can overfit to training sequence lengths.)

5. Could we put the stack anywhere in the cache, not just at the beginning? (Yes, but the beginning is simplest. Stack slots always have the same position indices (0-15), making them stable across different text lengths. The model always knows where to find the stack.)
