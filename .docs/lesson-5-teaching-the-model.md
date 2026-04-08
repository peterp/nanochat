# Lesson 5: Teaching the model to compute

## Concepts to learn

### The bootstrapping problem

We have all the pieces: a transformer, a KV cache, and a stack machine wired into the cache. But the model is randomly initialized. It has no idea that:
- Stack instruction tokens exist
- They correspond to mathematical operations
- It should emit them when it sees a math problem
- The results will appear in the KV cache

We need training data that teaches all of this simultaneously.

### Why we can't use internet data

Normal LLM training uses text scraped from the internet. But nobody on the internet writes math solutions in stack instructions. We can't find examples of "What is 23+45? <|stack_push|> 23 <|stack_push|> 45 <|stack_add|>" anywhere.

We need **synthetic data** - training examples we generate procedurally with Python code.

### Procedural data generation

The idea is simple: write Python functions that create random math problems and their corresponding stack solutions.

```python
import random

def generate_addition():
    a = random.randint(1, 999)
    b = random.randint(1, 999)
    answer = a + b

    problem = f"What is {a} + {b}?"
    stack_program = (
        f"<|stack_begin|>"
        f"<|stack_push|> {a} "
        f"<|stack_push|> {b} "
        f"<|stack_add|>"
        f"<|stack_read|>"
        f"<|stack_end|>"
    )
    result = f"The answer is {answer}."

    return f"{problem}\n{stack_program}\n{result}"
```

Each call to `generate_addition()` produces a different random example. We can generate millions of these. And because *we* compute the answer in Python, every example is guaranteed correct.

This is what Jefferson means when he says "I create a procedural generator for each task that can generate random permutations of the task."

### The ordering rule: computation before answer

This is the single most important design decision in the training data. Consider two formats:

**Format A (WRONG):**
```
What is 23 + 45? The answer is 68.
<|stack_begin|><|stack_push|> 23 <|stack_push|> 45 <|stack_add|><|stack_read|><|stack_end|>
```

**Format B (CORRECT):**
```
What is 23 + 45?
<|stack_begin|><|stack_push|> 23 <|stack_push|> 45 <|stack_add|><|stack_read|><|stack_end|>
The answer is 68.
```

Why does the order matter? Remember how training works: the model learns to predict the next token from all previous tokens. In Format A, when the model needs to predict "6" in "68", the string "23 + 45" is already in the context. The model can learn a shortcut: "when I see two numbers with a plus sign, predict their sum." It pattern-matches instead of computing. It never needs the stack instructions because the answer came first.

In Format B, the answer "68" appears *after* the stack instructions. When predicting "6" in "68", the most useful signal in the context is the result of the stack computation that just happened. The model is forced to learn: "I should use the stack to compute, then report what the stack computed."

**The rule: the answer must only be reachable through the computation.** Never give the model a shortcut.

### Curriculum learning: start easy

If you throw 5-digit multiplication at an untrained model, the gradients are too noisy - the model can't find any pattern to latch onto. Instead, build up gradually:

**Level 1: Single operation, small numbers (1-2 digits)**
```
What is 3 + 5?
<|stack_begin|><|stack_push|> 3 <|stack_push|> 5 <|stack_add|><|stack_read|><|stack_end|>
The answer is 8.
```

**Level 2: Single operation, larger numbers (2-3 digits)**
```
What is 47 * 13?
<|stack_begin|><|stack_push|> 47 <|stack_push|> 13 <|stack_mul|><|stack_read|><|stack_end|>
The answer is 611.
```

**Level 3: Multi-step expressions**
```
Calculate (12 * 5) + (8 * 3).
<|stack_begin|><|stack_push|> 12 <|stack_push|> 5 <|stack_mul|><|stack_push|> 8 <|stack_push|> 3 <|stack_mul|><|stack_add|><|stack_read|><|stack_end|>
The answer is 84.
```

**Level 4: Vector operations**
```
Compute the dot product of [2, 3] and [4, 5].
<|stack_begin|><|stack_push|> 2 <|stack_push|> 4 <|stack_mul|><|stack_push|> 3 <|stack_push|> 5 <|stack_mul|><|stack_add|><|stack_read|><|stack_end|>
The dot product is 23.
```

Start training on Level 1 only. Once accuracy is >90%, mix in Level 2. And so on.

### Teacher forcing

During training, at each position in the sequence, the model sees the *correct* previous tokens (not its own predictions). This is standard for all language model training.

For the stack computer, this means: even though the model might predict the wrong instruction, we show it the right one and move on. The stack state at each training position is what *would* result from the correct instruction sequence, not from the model's guesses.

We precompute these stack states during data generation:

```python
def generate_with_stack_states(problem_tokens):
    """For each position in the sequence, compute what the stack looks like."""
    stack = StackComputer()
    states = []
    for i, token in enumerate(problem_tokens):
        if is_instruction(token):
            stack.execute(token)
        states.append(stack.snapshot())  # stack state at position i
    return states
```

These precomputed states are loaded during training and injected into the KV cache prefix at each position.

### Two training approaches

**Approach A: Text-inlined (start here)**

Don't modify the model architecture at all. Instead, inline the stack state as text tokens:

```
<|stack_push|> 5 [stack: 5]
<|stack_push|> 3 [stack: 5, 3]
<|stack_add|> [stack: 8]
```

The model learns the relationship between instructions and stack states through pure next-token prediction. At inference time, the StackEngine executes real operations. This is simpler to train and debug.

**Approach B: KV prefix injection (upgrade to this)**

Use the real KV cache mechanism from Lesson 4. During training, inject the precomputed stack state into the KV prefix at each sequence position. The model learns to attend to the prefix for computation results. This is the "real" version - the stack truly lives inside the model's attention.

Start with Approach A to validate the concept works, then upgrade to Approach B.

### Mixing text and computation

The model needs to stay good at regular text, not just math. The training data should be a mix:

- ~50% pure text (Shakespeare or similar corpus) - maintains language ability
- ~30% simple arithmetic problems with stack solutions
- ~15% vector operations
- ~5% matrix operations

Without the text data, the model forgets how to generate coherent language. Without enough computation data, it doesn't learn to use the stack.

---

## What we'll build

### Files

| File | Purpose |
|------|---------|
| `data/generate_arithmetic.py` | Procedural generators for addition, subtraction, multiplication, division. Configurable digit ranges. Outputs (problem, stack instructions, answer) triples. |
| `data/generate_matrix.py` | Generators for dot products, cross products, small (2x2, 3x3) matrix multiplications. |
| `data/generate_mixed.py` | Mixes text corpus data with computation problems. Controls difficulty distribution. |
| `data/dataset.py` | PyTorch Dataset class. Loads generated data, tokenizes, precomputes stack state trajectories. |
| `nanochat/stack_train.py` | Training loop adapted for stack-augmented data. Handles KV prefix injection (Approach B) or plain token sequences (Approach A). |
| `nanochat/stack_eval.py` | Evaluation: exact-match accuracy on held-out problems, instruction validity rate, text perplexity. |

### Training plan

1. **Phase 1 - Language pretraining**: Train on pure text (5-10M tokens) until the model generates coherent text. This is Lessons 1-2 revisited with the final model size.

2. **Phase 2 - Stack finetuning**: Switch to mixed data (text + arithmetic). Lower learning rate to 1/10 of pretraining. Train on 20-40M tokens.

3. **Phase 3 - Curriculum**: Gradually increase problem difficulty as accuracy improves.

### Model scale for the final system

- 6-8 layers, 8 heads, 512 embedding dim = ~25M parameters
- Sequence length 512
- Stack depth 16
- Vocabulary: ~65 characters + ~12 instruction tokens + a few special tokens ≈ ~80 tokens
- Total training: ~50M tokens across all phases
- Hardware: single GPU, ~2-4 hours total

### How you'll know it works

The ultimate test: give the trained model a math problem it has never seen.

```
Input:  "What is 173 + 284?"
Output: "<|stack_begin|><|stack_push|> 173 <|stack_push|> 284 <|stack_add|><|stack_read|><|stack_end|>
         The answer is 457."
```

The model didn't memorize this. It learned:
1. "This is an addition problem" → emit stack_push for both numbers, then stack_add
2. The StackEngine executed the instructions → 457 is now in the KV cache
3. "Report the answer" → attend to the stack, generate "457"

### Evaluation metrics

- **Exact-match accuracy**: What fraction of held-out problems does the model solve correctly? Target: >90% at trained difficulty levels.
- **Instruction validity**: What fraction of generated instruction sequences are well-formed (no stack underflows, correct operand counts)? Target: >95%.
- **Generalization**: Train on 1-3 digit addition. Test on 4-digit addition. Some generalization is expected; perfect generalization is hard.
- **Text perplexity**: Does the model still generate good text? Compare against the Phase 1 model. Should not degrade significantly.

---

## Key questions to understand before moving on

1. Why procedural generation instead of having a large LM generate training data? (Procedural generation is *provably correct* - we compute the answer in Python and verify the stack instructions. LLM-generated data could contain errors, which would teach the model wrong patterns.)

2. How many training examples are enough? (This depends on the complexity. For 1-3 digit addition, a few thousand unique examples is probably enough for the model to learn the pattern. For matrix operations, you need more because the programs are longer and more varied. The key insight: you can generate unlimited data, so overfitting is not the main risk - underfitting is.)

3. What if the model learns to do arithmetic without the stack? (For small numbers, it might. "3+5=8" could be memorized. This is fine - the stack proves its value on problems too hard to memorize, like 4-digit multiplication or matrix operations. We verify by testing on numbers larger than anything in the training set.)

4. Could you skip Approach A and go straight to Approach B? (You could, but Approach A is much easier to debug. If the model isn't learning, you can inspect the text-inlined stack states and see exactly what the model sees. With KV injection, debugging is harder because the stack state is embedded in high-dimensional vectors.)

5. Why mix in text data? Why not train only on math? (Catastrophic forgetting. If you finetune a language model on only math, it rapidly loses its language abilities. The text data acts as a regularizer, keeping the language modeling pathways active while the math pathways develop.)
