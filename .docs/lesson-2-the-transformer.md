# Lesson 2: The Transformer - Attention is the whole game

## Concepts to learn

### The problem before transformers

Before transformers (2017), the dominant approach was Recurrent Neural Networks (RNNs/LSTMs). They read text one token at a time, maintaining a hidden state - like reading a book while trying to remember everything in a single mental note. By token 100, the information from token 1 has been compressed through 99 transformations. Details get lost. This is the "vanishing gradient" problem.

### Attention: letting every token look at every other token

Attention solves this by letting every token directly access every other token, no matter how far apart. Token 100 can look straight at token 1 without information passing through 99 intermediate steps.

Here's the mechanical process. Each token's embedding (a vector of 256 numbers) gets linearly projected into three separate vectors:

- **Query (Q)**: "I'm looking for information about X"
- **Key (K)**: "I contain information about Y"
- **Value (V)**: "Here's the actual information I carry"

Think of it like a library:
- Q is your search query
- K is the label on each book
- V is the content of each book

To process token 50:
1. Take token 50's Q vector
2. Compute the dot product of Q with every previous token's K vector (tokens 1-49). High dot product = "this token is relevant to what I'm looking for"
3. Divide by √(head_dim) to keep numbers stable (scaled dot-product attention)
4. Apply softmax to get attention weights that sum to 1
5. Multiply each token's V vector by its attention weight and sum them up
6. The result is token 50's output: a weighted blend of information from all relevant previous tokens

In code-like pseudocode:
```
scores = Q[50] @ K[1:49].T          # how relevant is each previous token?
scores = scores / sqrt(head_dim)      # scale to prevent huge numbers
weights = softmax(scores)             # normalize to probabilities
output = weights @ V[1:49]            # weighted sum of values
```

### Multi-head attention

One attention head might learn to track grammatical structure ("this adjective modifies that noun"). Another might track semantic relationships ("this pronoun refers to that entity"). Another might track local patterns ("after 'q', 'u' usually follows").

With 4 heads and embedding dim 256, each head operates on 64 dimensions (256/4). Each head has its own Q, K, V projections. Their outputs are concatenated and projected back to 256 dimensions:

```
head_1 = attention(Q1, K1, V1)    # 64-dim output
head_2 = attention(Q2, K2, V2)    # 64-dim output
head_3 = attention(Q3, K3, V3)    # 64-dim output
head_4 = attention(Q4, K4, V4)    # 64-dim output
output = concat(head_1..4) @ W_out  # back to 256-dim
```

### The causal mask

During training, we process the whole sequence in parallel (much faster than one token at a time). But there's a problem: token 50 shouldn't be able to see tokens 51, 52, 53... because during generation, those tokens don't exist yet.

The **causal mask** solves this by setting attention scores to -infinity for any position ahead of the current token. After softmax, those positions get weight 0. Each token can only attend to itself and everything before it.

```
Token:  1  2  3  4  5
    1: [✓  ✗  ✗  ✗  ✗]   ← token 1 sees only itself
    2: [✓  ✓  ✗  ✗  ✗]   ← token 2 sees 1 and itself
    3: [✓  ✓  ✓  ✗  ✗]   ← token 3 sees 1, 2, and itself
    4: [✓  ✓  ✓  ✓  ✗]
    5: [✓  ✓  ✓  ✓  ✓]   ← token 5 sees everything
```

### Residual connections

Deep networks have a problem: gradients can vanish or explode as they flow through many layers. Residual connections fix this by adding the input of each sub-layer directly to its output:

```
x = x + attention(norm(x))    # attention adds to the original
x = x + mlp(norm(x))          # MLP adds to the original
```

This means each layer only needs to learn the *difference* from its input, not a complete transformation. Gradients can flow directly through the addition, making training much more stable.

### Layer normalization (RMSNorm)

Neural networks train better when their internal values are roughly normalized (mean ~0, variance ~1). RMSNorm (Root Mean Square Normalization) divides each vector by its root mean square:

```
rms = sqrt(mean(x^2))
output = x / rms * scale    # scale is a learned parameter
```

We apply this *before* each sub-layer (pre-norm), not after (post-norm). Pre-norm is more stable for training.

### The feedforward MLP

Attention lets tokens talk to each other. The MLP lets each token "think" independently. It's a simple two-layer network:

```
x → Linear(256 → 1024) → GELU activation → Linear(1024 → 256) → output
```

The 4x expansion (256 → 1024) gives the network more room to compute. GELU is a smooth activation function (similar to ReLU but differentiable everywhere).

### The full transformer block

Putting it all together:

```
input
  ↓
  ├──→ RMSNorm → Multi-Head Attention ──→ + (add residual)
  │                                        ↓
  ├──→ RMSNorm → MLP ─────────────────→ + (add residual)
  │                                        ↓
  output
```

Stack 4 of these blocks, add token embeddings at the bottom and a linear layer at the top, and you have a GPT.

---

## What we'll build

The actual model classes. Every line should be understood.

### Files

| File | Purpose |
|------|---------|
| `nanochat/model.py` | `GPTConfig` (hyperparameters), `CausalSelfAttention`, `MLP`, `Block`, `GPT`. This is the core of the entire project. |

### Architecture details

```python
GPTConfig:
    vocab_size = 65        # Shakespeare characters
    n_layer = 4            # transformer blocks
    n_head = 4             # attention heads per block
    n_embd = 256           # embedding dimension
    sequence_len = 256     # max context length
    dropout = 0.1          # regularization during training
```

This gives ~2.5M parameters. For reference:
- GPT-2 small: 124M parameters
- GPT-3: 175B parameters
- Our model: 2.5M parameters

We're 50x smaller than the smallest GPT-2, but that's enough to learn character patterns in Shakespeare.

### How you'll know it works

This lesson's code is what makes Lesson 1's training loop actually work. The verification is the same: loss decreases, generated text looks like English.

---

## Key questions to understand before moving on

1. Why does attention use Q, K, V instead of just comparing embeddings directly? (Separating "what I'm looking for" from "what I advertise" from "what I contain" gives the model more flexibility. A token might be relevant to token 50 for a different reason than why it's relevant to token 30.)

2. Why divide by √(head_dim)? (Without scaling, dot products of high-dimensional vectors can be very large, making softmax produce near-one-hot distributions. Dividing keeps the variance at ~1 regardless of dimension.)

3. Why pre-norm instead of post-norm? (Pre-norm means the residual stream carries unnormalized values with direct gradient flow. Post-norm can cause training instability in deeper models.)

4. What does each transformer layer "do"? (No one layer has a clean single role. But research suggests early layers learn local patterns, middle layers learn syntactic structure, and later layers learn semantic/task-specific features. This is approximate.)

5. Why is the MLP 4x wider than the embedding dim? (More parameters = more capacity for per-token computation. The 4x ratio is empirical - it works well. Some modern models use different ratios.)
