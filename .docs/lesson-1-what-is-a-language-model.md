# Lesson 1: What is a language model, really?

## Concepts to learn

### Next-token prediction

A language model does one thing: given a sequence of tokens, predict what comes next. "The cat sat on the" → "mat" (probably). That's it. Everything else - conversations, code generation, reasoning - emerges from this one capability.

Think of it like autocomplete on your phone, but scaled up massively. Your phone predicts the next word from the last few words. A language model predicts the next token from *thousands* of previous tokens, using billions of learned parameters.

### Tokens

Text is split into discrete units called tokens. We'll start with the simplest possible tokenization: each character is a token.

For Shakespeare text, the vocabulary is roughly 65 characters: lowercase letters, uppercase letters, digits, punctuation, spaces, newlines. Each character gets an integer ID:

```
'a' → 0, 'b' → 1, ... 'z' → 25, 'A' → 26, ... ' ' → 52, '\n' → 53, ...
```

Later systems (GPT-2, GPT-4) use subword pieces (BPE) where common words like "the" are single tokens and rare words get split into pieces. But characters keep everything simple and visible - you can see exactly what the model sees.

### The training loop

Take a chunk of text, say 256 characters long. The setup:

```
Text:    T h e   c a t   s a t
Input:   T h e   c a t   s a
Target:    h e   c a t   s a t
```

The input is every character except the last. The target is every character except the first (shifted by one position). At every position, the model tries to predict the target character from all the input characters before and including that position.

The model outputs a probability distribution over all 65 characters. If the target is 'h' and the model assigned probability 0.7 to 'h', that's pretty good. If it assigned probability 0.01, that's bad.

**Cross-entropy loss** measures how wrong the model was. It's `-log(probability assigned to the correct answer)`. If the model was confident and right (p=0.9), loss is low (0.1). If it was wrong (p=0.01), loss is high (4.6). We average this across all positions in the sequence.

We compute the gradient of this loss with respect to every parameter in the model (backpropagation), then nudge each parameter a tiny bit in the direction that would reduce the loss (gradient descent). Repeat millions of times.

### Temperature and sampling

The model outputs logits (raw scores) for each possible next token. To turn these into probabilities, we apply softmax. But first, we can divide the logits by a **temperature** value:

- **Temperature = 1.0**: Standard probabilities. The model's actual confidence.
- **Temperature = 0.5**: Sharper distribution. The most likely token becomes even more likely. More predictable output.
- **Temperature = 1.5**: Flatter distribution. Less likely tokens get a boost. More creative/chaotic output.
- **Temperature → 0**: Always pick the most likely token (greedy decoding). Deterministic.

**Top-k sampling**: Instead of sampling from all 65 characters, only consider the top k most likely ones. This prevents the model from occasionally picking absurd tokens.

---

## What we'll build

A character-level model that trains on Shakespeare (~1M characters) and generates plausible-looking text. The model will have ~2.5M parameters - tiny by modern standards, but enough to learn English character patterns.

### Files

| File | Purpose |
|------|---------|
| `nanochat/tokenizer.py` | Character-level tokenizer: `encode("hello") → [7,4,11,11,14]`, `decode([7,4,11,11,14]) → "hello"`. Reserve extra vocabulary slots for the special tokens we'll add later (stack instructions). |
| `nanochat/model.py` | The transformer model. GPTConfig (hyperparameters), CausalSelfAttention, MLP, Block, GPT. ~2.5M params: 4 layers, 4 heads, 256 embedding dimension. |
| `nanochat/train.py` | The training loop. AdamW optimizer, cosine learning rate schedule with warmup, gradient accumulation. Logs training/validation loss. |
| `nanochat/generate.py` | Text generation. Feed a prompt, sample tokens one at a time, print the result. Supports temperature and top-k. |
| `data/prepare.py` | Download the Shakespeare dataset (~1M characters). Split into train (90%) and validation (10%). |

### Model architecture (preview)

```
Input characters → Token Embeddings (65 → 256 dim)
                   + Position Embeddings
                         ↓
                   [Transformer Block] × 4
                   (each block = Attention + MLP)
                         ↓
                   Layer Norm
                         ↓
                   Linear (256 → 65) → logits over characters
```

### How you'll know it works

- Training loss drops from ~4.17 (that's ln(65), the loss when guessing uniformly at random across 65 characters) down to ~1.0-1.5
- Generated text contains real English words, proper spacing, rough grammar
- You can see the model improve: early output is random garbage, later output looks like garbled Shakespeare

### Scale

- ~5 minutes on a modern GPU (RTX 3090, A100, etc.)
- ~30 minutes on an M-series Mac (using MPS backend)
- ~1-2 hours on CPU (works, just slow)

---

## Key questions to understand before moving on

1. Why is next-token prediction sufficient to learn language? (Because the only way to consistently predict the next character is to understand the patterns of language - spelling, grammar, meaning.)

2. Why character-level and not word-level? (Simpler vocabulary, no tokenization edge cases, and we can see exactly what the model processes. The downside is longer sequences - "hello" is 5 tokens instead of 1.)

3. What does the loss number actually mean? (A loss of 1.5 means the model is, on average, as confident as if it had narrowed down the next character to about e^1.5 ≈ 4.5 equally-likely options, out of 65 total. That's quite good.)

4. Why do we need a separate validation set? (To check if the model is actually learning patterns vs. just memorizing the training data. If training loss is low but validation loss is high, the model is overfitting.)
