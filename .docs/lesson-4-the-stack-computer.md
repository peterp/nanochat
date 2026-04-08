# Lesson 4: What is a stack computer?

## Concepts to learn

### Two kinds of computers

Most CPUs you've used are **register machines**. They have named storage locations (registers like `eax`, `r1`) and instructions that refer to them by name:

```
ADD eax, ebx    ; eax = eax + ebx
MOV ecx, eax    ; ecx = eax
```

A **stack machine** is simpler. There are no named registers. There's just a stack - a pile of numbers where you can only touch the top. All instructions implicitly operate on the top of the stack:

```
PUSH 3          ; stack: [3]
PUSH 5          ; stack: [3, 5]
ADD             ; pop 3 and 5, push 8 → stack: [8]
```

Real examples of stack machines: the Java Virtual Machine (JVM), WebAssembly, PostScript, Forth, and old HP calculators.

### Reverse Polish Notation (RPN)

Normal math notation: `(3 + 5) * 2`

The same expression in RPN (stack notation):
```
PUSH 3     → stack: [3]
PUSH 5     → stack: [3, 5]
ADD        → stack: [8]
PUSH 2     → stack: [8, 2]
MUL        → stack: [16]
```

No parentheses needed. The order of operations is explicit in the instruction sequence. Every arithmetic expression can be converted to a sequence of PUSHes and operations. This is what compilers do when they generate code for stack-based VMs.

### Why a stack machine inside a transformer?

Language models are bad at arithmetic. GPT-4 can't reliably compute 4-digit multiplication. Why? Because it was trained to predict tokens, not compute. When it sees "347 × 829 = ", it tries to pattern-match against similar multiplications it saw during training. It's not doing long multiplication - it's doing fuzzy recall.

There are three ways to fix this:

1. **External tools** (what ChatGPT does): The model generates Python code, an external interpreter runs it, and the result is pasted back. Works, but the computation happens *outside* the model.

2. **Chain-of-thought** (what reasoning models do): The model writes out intermediate steps as text: "347 × 829 = 347 × 800 + 347 × 29 = ...". Better, but the model can still make errors at each step because each step is still pattern-matching.

3. **Embedded computation** (what Jefferson does): The model generates stack instructions, and a real stack machine executes them *inside the model's own inference loop*. The model gets the **exact** result, not an approximation. The computation happens between forward passes, and the result appears in the KV cache, immediately available for the next token.

Option 3 is what we're building. The model learns *when* to compute and *what* instructions to emit. The actual arithmetic is done by the stack machine, which cannot make errors.

### The instruction set

Our stack computer supports these operations:

| Token | What it does | Stack before → after |
|-------|-------------|---------------------|
| `<\|stack_begin\|>` | Marks start of computation | (unchanged) |
| `<\|stack_end\|>` | Marks end of computation | (unchanged) |
| `<\|stack_push\|>` | Push the following number onto stack | [] → [n] |
| `<\|stack_pop\|>` | Remove top element | [a] → [] |
| `<\|stack_dup\|>` | Duplicate top element | [a] → [a, a] |
| `<\|stack_swap\|>` | Swap top two elements | [a, b] → [b, a] |
| `<\|stack_add\|>` | Pop two, push sum | [a, b] → [a+b] |
| `<\|stack_sub\|>` | Pop two, push difference | [a, b] → [a-b] |
| `<\|stack_mul\|>` | Pop two, push product | [a, b] → [a*b] |
| `<\|stack_div\|>` | Pop two, push quotient | [a, b] → [a/b] |
| `<\|stack_read\|>` | Read top without popping; result injected as tokens | (unchanged) |

### How the stack lives in the KV cache

This is the key architectural idea. Revisiting what we learned in Lesson 3:

The KV cache has positions 0, 1, 2, 3, ... We reserve positions 0 through 15 as **stack slots** (max stack depth = 16). Text tokens start at position 16.

Each stack slot needs to communicate two things to the model:
1. **"Which slot am I?"** → encoded in the Key vector. We use a learned embedding for each slot (similar to how tokens have learned embeddings). Slot 0's key always looks the same, slot 1's key always looks the same, etc.
2. **"What number am I holding?"** → encoded in the Value vector. We learn a small neural network that maps a scalar (the number) to an embedding vector.

```python
# Pseudocode for updating the KV cache with stack state
for i in range(max_stack_depth):
    k_cache[layer, :, i] = stack_slot_embedding[i]     # "I am slot i"
    if i < stack_pointer:
        v_cache[layer, :, i] = value_encoder(stack[i])  # "I hold this number"
    else:
        v_cache[layer, :, i] = empty_slot_embedding     # "I'm empty"
```

When the model processes a new text token, its attention naturally covers positions 0-15 (the stack) alongside all previous text positions. If the stack contains useful information, the model will attend to it. If not, attention weights on the stack slots will be low.

### The execution cycle

Here's what happens during generation, step by step:

```
1. Model generates token "What"      → normal text, no stack action
2. Model generates token "is"        → normal text, no stack action
3. Model generates token "23+45?"    → normal text, no stack action
4. Model generates <|stack_begin|>   → flag: we're computing now
5. Model generates <|stack_push|>    → StackEngine: "next number gets pushed"
6. Model generates "2"               → accumulating number: "2"
7. Model generates "3"               → accumulating number: "23"
8. Model generates <|stack_push|>    → push 23, start new number
   → Stack: [23]
   → Rewrite KV cache positions 0-15 with new stack state
9. Model generates "4"               → accumulating: "4"
10. Model generates "5"              → accumulating: "45"
11. Model generates <|stack_add|>    → push 45, then pop 23 and 45, push 68
    → Stack: [68]
    → Rewrite KV cache positions 0-15
12. Model generates <|stack_read|>   → inject "68" as tokens
13. Model generates <|stack_end|>    → done computing
14. Model generates "The answer"     → normal text, informed by stack result
15. Model generates "is 68."         → correct answer
```

The crucial thing: at step 14, the model doesn't need to "remember" that 23+45=68. The number 68 is right there in the KV cache at stack position 0. The model just attends to it.

---

## What we'll build

### Files

| File | Purpose |
|------|---------|
| `nanochat/stack.py` | `StackComputer` class. Pure Python, no ML. Implements push, pop, dup, swap, add, sub, mul, div. Includes `to_kv_representation()` that converts stack state to tensors. |
| `nanochat/stack_engine.py` | `StackEngine` extending `Engine`. Detects instruction tokens during generation, executes them on the `StackComputer`, rewrites KV cache prefix. |
| `nanochat/tokenizer.py` | Updated: add the ~12 instruction tokens to the vocabulary. |
| `nanochat/model.py` | Updated: add `stack_slot_embeddings` and `stack_value_encoder` parameters. Adjust RoPE so text positions start at 16. |

### How you'll know it works

1. **Unit tests for StackComputer**: `push(3), push(5), add() → top() == 8`. Cover all operations, including edge cases (pop from empty stack, division by zero).

2. **KV injection test**: Create a model, create a KV cache, manually write stack values into positions 0-15, run a forward pass. Verify the model's attention weights are non-zero at stack positions. (Output will be garbage since the model is untrained, but the plumbing works.)

3. **End-to-end smoke test**: Manually feed the token sequence `<|stack_begin|> <|stack_push|> 3 <|stack_push|> 5 <|stack_add|> <|stack_read|> <|stack_end|>` through the StackEngine. Verify: stack operations execute correctly, KV cache is updated at each step, and the read value (8) is injected back as tokens.

---

## Key questions to understand before moving on

1. Why a stack machine and not a register machine? (Stack machines have no named operands - all operations are implicit on the top of the stack. This means each instruction is a single token, no arguments needed except for PUSH. Simpler token vocabulary, shorter instruction sequences.)

2. Why 16 stack slots? (Most arithmetic expressions need at most a few stack levels. 16 is generous - even a complex nested expression rarely needs more than 5-6 levels. The number is a tradeoff: more slots = more KV cache space used = less room for text context.)

3. Could the model learn to use the stack without being explicitly trained on instruction sequences? (No. The model starts with random weights - it has no idea the stack exists. We must show it examples of problems being solved via stack instructions. It learns the *pattern* of when to use the stack and what instructions to emit.)

4. What if the model generates an invalid instruction sequence? (The StackEngine handles this gracefully: popping from an empty stack returns 0, division by zero returns 0, etc. During training, the model sees only valid sequences. Invalid sequences at inference time mean the model hasn't learned well enough.)

5. Why is this different from chain-of-thought? (Chain-of-thought writes intermediate results as text tokens - each step is still a prediction that could be wrong. Stack operations are *executed*, not predicted. PUSH 23, PUSH 45, ADD *always* gives 68. The model doesn't predict 68 - the stack machine computes it.)
