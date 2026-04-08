# nanochat

A course for building a language model that is also a stack computer, from scratch.

Each forward pass is both a compute tick for a stack machine and a new token for the language model. The stack lives in the KV cache. The model learns to emit stack instructions when it needs to compute. Each instruction executes inside the model's inference loop. The result is immediately available for generating the next token.

Inspired by [Andrew Jefferson](https://x.com/EastlondonDev). Built from scratch, nanochat-style.

## Lessons

Each lesson teaches concepts first, then you implement.

| # | Lesson | What you'll learn | What you'll build |
|---|--------|-------------------|-------------------|
| 1 | [What is a language model?](.docs/lesson-1-what-is-a-language-model.md) | Next-token prediction, tokens, training loop, cross-entropy loss, temperature/sampling | Character-level tokenizer, training loop, text generation |
| 2 | [The Transformer](.docs/lesson-2-the-transformer.md) | Attention (Q/K/V), multi-head attention, causal mask, residual connections, RMSNorm, MLP | GPTConfig, CausalSelfAttention, MLP, Block, GPT classes |
| 3 | [The KV Cache](.docs/lesson-3-the-kv-cache.md) | Why caching works, KV cache structure, RoPE, and how synthetic KV entries enable the stack computer | KVCache class, Engine class, RoPE implementation |
| 4 | [The Stack Computer](.docs/lesson-4-the-stack-computer.md) | Stack machines vs register machines, RPN, how the stack lives in KV cache positions 0-15, the execution cycle | StackComputer, StackEngine, instruction tokens |
| 5 | [Teaching the Model](.docs/lesson-5-teaching-the-model.md) | Procedural data generation, why answer must come after computation, curriculum learning, teacher forcing | Data generators, training loop with stack injection, evaluation |
| 6 | [Fine-tuning a Real Model](.docs/lesson-6-finetuning-a-real-model.md) | LoRA, parameter-efficient fine-tuning, Apple Silicon/MPS training, multimodal as the same pattern | LoRA adapter, fine-tune Gemma/Llama with stack instructions, device abstraction |

Lessons 1-5 build everything from scratch so you understand every component. Lesson 6 shows how to do it in practice: take a pre-trained model that already understands language and teach it just the stack computer skill using LoRA. Based on [gemma-tuner-multimodal](https://github.com/mattmireles/gemma-tuner-multimodal).

## How it works

```
Token: <|stack_push|> 5 <|stack_push|> 3 <|stack_add|> <|stack_read|>
         |                                    |              |
         v                                    v              v
    Stack: [5]         Stack: [5,3]      Stack: [8]     Inject "8"
         |                  |                |           as next token
    Written into KV cache prefix before each forward pass
```

The model doesn't memorize "5+3=8". It emits stack instructions, the stack machine executes them inside the KV cache, and the result is available via attention for the next token.

## Dependencies

```
torch >= 2.1.0
numpy
tqdm
```
