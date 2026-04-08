# Lesson 6: Fine-tuning a real model with LoRA

## Why this lesson exists

In Lessons 1-5, you trained a tiny model (~25M params) from scratch. It works, but it's limited - small vocabulary, character-level tokenization, trained on a small dataset. Real language models (Gemma 2B, Llama 3B) have been pre-trained on trillions of tokens and already understand language deeply.

What if instead of building a stack computer LM from scratch, you could take a pre-trained model that already speaks fluent English and teach it *just* the stack computer part? That's what fine-tuning is. And LoRA makes it practical on a single laptop.

This lesson draws from [gemma-tuner-multimodal](https://github.com/mattmireles/gemma-tuner-multimodal), a toolkit for fine-tuning Gemma models on Apple Silicon.

---

## Concepts to learn

### The pre-training / fine-tuning paradigm

Training a language model has two phases:

**Pre-training** (expensive): Train on a massive text corpus (trillions of tokens) for weeks on hundreds of GPUs. The model learns language, facts, reasoning patterns. This costs millions of dollars. Someone else does this (Google, Meta, etc.) and releases the weights.

**Fine-tuning** (cheap): Take the pre-trained model and train it further on a small, specific dataset. The model retains everything it learned during pre-training but gains new capabilities. This takes hours on a single GPU.

For us: the pre-trained model already understands "What is 23 + 45?" as a math question. We just need to teach it to respond with stack instructions instead of guessing the answer.

### Why you can't fine-tune all the parameters

Gemma 2B has 2 billion parameters. Each parameter is a 32-bit float = 4 bytes. Just the model weights are 8GB. During training, you also need:
- Gradients: another 8GB
- Optimizer state (AdamW stores two extra values per parameter): 16GB
- Activations for backpropagation: varies, but several GB

Total: ~40GB+ for full fine-tuning of a 2B model. That doesn't fit on most GPUs, let alone a laptop.

### LoRA: Low-Rank Adaptation

LoRA is the key insight that makes fine-tuning practical. Here's the idea:

A weight matrix in the transformer might be 2048 × 2048 = 4 million parameters. During fine-tuning, you want to change this matrix by some amount ΔW. LoRA's observation: **ΔW doesn't need to be full rank**.

Instead of learning a 2048×2048 ΔW (4M params), decompose it into two smaller matrices:

```
ΔW = A × B
where A is 2048 × r and B is r × 2048
```

If rank r = 8, then A has 16,384 params and B has 16,384 params. Total: 32,768 params instead of 4 million. That's a 125x reduction.

During fine-tuning:
- The original weight matrix W is **frozen** (no gradients, no optimizer state)
- Only A and B are trained
- The forward pass computes: `output = (W + A×B) @ input`
- After training, you can merge: `W_new = W + A×B` and discard A and B

Why this works: the changes needed to add stack computer capabilities to a language model are probably low-rank. You're not changing *how* the model processes language - you're adding a narrow new skill on top.

**In practice:**
- Pre-trained model: 2B params, frozen → 8GB memory, no gradients needed
- LoRA adapters: ~10M trainable params → negligible memory
- Total training memory: ~10-12GB, fits on a laptop GPU

### Apple Silicon and MPS

Apple's M-series chips have unified memory - CPU and GPU share the same RAM. PyTorch's MPS (Metal Performance Shaders) backend lets you train on the Mac GPU.

Key differences from NVIDIA/CUDA:
- **No bfloat16**: MPS requires float32 (uses more memory but works)
- **Unified memory**: 16GB of RAM means 16GB for both CPU and GPU, no separate VRAM
- **Attention implementation**: Must use "eager" mode (not Flash Attention)
- **Memory management**: Need to monitor watermarks to prevent swap thrashing

The gemma-tuner-multimodal repo handles all of these quirks. The pattern: detect the device, adjust dtypes and batch sizes accordingly.

### What changes when fine-tuning for stack instructions

The pre-trained model's tokenizer doesn't know about `<|stack_push|>` or `<|stack_add|>`. We need to:

1. **Extend the vocabulary**: Add our ~12 instruction tokens to the tokenizer. This adds 12 new rows to the embedding matrix and the LM head (output) matrix. These new rows are randomly initialized - only they need heavy training.

2. **Apply LoRA to attention layers**: The Q, K, V, and output projection matrices get LoRA adapters. This lets the model learn new attention patterns (like attending to stack state in the KV prefix) without rewriting its entire attention mechanism.

3. **Keep the training data format**: Same as Lesson 5 - math problems followed by stack instructions followed by answers. But now the model starts from a much stronger base, so it learns faster with less data.

### Multimodal extension (optional/advanced)

The gemma-tuner-multimodal repo also shows how to add image and audio inputs to a text model. The principle is the same as our stack computer:

- **Stack computer**: Inject synthetic KV entries (stack state) that the model attends to
- **Vision**: Inject encoded image features as KV entries that the model attends to
- **Audio**: Inject encoded audio features as KV entries that the model attends to

All three are "extra stuff in the attention context that isn't text." The model learns to attend to whatever is useful. This is the same architectural pattern - the KV cache is a general-purpose interface for giving the model non-text information.

---

## What we'll build

### Files

| File | Purpose |
|------|---------|
| `nanochat/lora.py` | LoRA implementation: `LoRALinear` layer that wraps a frozen linear layer with trainable low-rank adapters A and B. Methods to merge/unmerge weights. |
| `nanochat/finetune.py` | Fine-tuning script: load a pre-trained Gemma 2B (or similar small model), extend tokenizer with instruction tokens, apply LoRA to attention layers, train on stack instruction data. |
| `nanochat/device.py` | Device abstraction: detect MPS/CUDA/CPU, set appropriate dtypes and batch sizes, manage memory. |

### Approach

1. **Load pre-trained model**: Download Gemma 2B (or Llama 3.2 1B for even smaller) from Hugging Face
2. **Extend tokenizer**: Add stack instruction tokens, resize model embeddings
3. **Apply LoRA**: Wrap Q, K, V, output projections with LoRALinear (r=8)
4. **Generate training data**: Same procedural generators from Lesson 5, but using the model's BPE tokenizer instead of character-level
5. **Fine-tune**: Train LoRA adapters on mixed text + computation data
6. **Wire up StackEngine**: Same execution logic from Lesson 4, but using the fine-tuned model
7. **Merge and export**: Merge LoRA weights into the base model for efficient inference

### How you'll know it works

- The fine-tuned model solves arithmetic problems it hasn't seen by emitting valid stack instruction sequences
- It still generates coherent text (pre-trained capabilities preserved)
- Compared to the from-scratch model (Lesson 5): faster training, better language quality, potentially better generalization on harder problems
- Training fits in ~16GB RAM (M-series Mac) or ~12GB VRAM (consumer GPU)

### Scale

- Base model: Gemma 2B or Llama 3.2 1B
- LoRA rank: 8 (trainable params: ~10M out of 1-2B total)
- Training data: 5-10M tokens of mixed text + computation
- Training time: ~1-2 hours on consumer GPU, ~3-4 hours on M-series Mac
- Memory: ~10-16GB

---

## Key questions to understand before moving on

1. Why is low rank sufficient? (Language models are overparameterized. The "direction" in weight space needed for a new skill can usually be described by a few dimensions. Rank 8 means 8 independent directions of adaptation - enough for adding stack operations, which is a narrow skill.)

2. Why LoRA specifically and not other parameter-efficient methods? (LoRA is simple, well-understood, and adds zero inference overhead after merging. Other methods like prefix tuning or adapters add latency. LoRA's merged weights are indistinguishable from a fully fine-tuned model.)

3. Could you add the stack computer to GPT-4 / Claude this way? (In principle yes, if you had access to the weights. The technique is model-agnostic. In practice, you'd want to do it with open-weight models like Gemma or Llama.)

4. How does this compare to the from-scratch approach? (From-scratch gives you deep understanding of every component. LoRA fine-tuning is what you'd do in practice - it's faster, cheaper, and the resulting model is much more capable because it inherits billions of tokens of pre-training.)

5. What's the connection between multimodal inputs and the stack computer? (Both use the same mechanism: inject non-text information into the attention context. Images become KV entries via a vision encoder. Stack state becomes KV entries via the stack value encoder. The model attends to both the same way. The KV cache is a universal interface.)
