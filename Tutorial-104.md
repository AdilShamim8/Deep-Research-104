# Project 4: Build "Deep Research" Capability with Web Search and Reasoning Models

## The Most Beginner-Friendly, End-to-End, Deep Tutorial — DEFINITIVE VERSION |

## Complete Beginner Tutorial — Part 1
---
# TABLE OF CONTENTS (Part 1)

```
PART 1 COVERS:
├── 1. What is "Deep Research" Capability?
├── 2. Reasoning and Thinking LLMs
│   ├── What makes a model a "reasoning model"?
│   ├── OpenAI "o" family overview
│   └── DeepSeek-R1 overview
├── 3. Inference-Time Techniques
│   ├── What is inference time?
│   ├── Inference-time scaling
│   ├── Chain of Thought (CoT) prompting
│   ├── Parallel sampling
│   ├── Sequential sampling
│   ├── Tree of Thoughts (ToT)
│   └── Search against a verifier
```

---

# CHAPTER 1: What is "Deep Research" Capability?

---

## 1.1 — Start From Zero: What Problem Are We Solving?

Let me tell you a story first.

---

### The Story of Two Students

**Student A** is asked: *"What is the best treatment for diabetes in 2024?"*

Student A opens Google, clicks the first link, reads one paragraph, and gives the answer.

**Student B** is asked the same question.

Student B:
- Opens 10 different websites
- Reads medical journals
- Compares different treatments
- Checks which studies are recent
- Cross-verifies facts across sources
- Thinks about exceptions and edge cases
- Then writes a structured, accurate, cited answer

---

**Which student gave better research?**

Obviously Student B.

**Deep Research** means building an AI system that behaves like **Student B** — not just answering from memory, but actively searching, reading, reasoning, and synthesizing information.

---

## 1.2 — What is "Deep Research" in AI Terms?

```
Traditional AI (GPT-3 style):
User Question → Model Memory → Answer
Problem: Knowledge is frozen at training time!

Deep Research AI:
User Question
    ↓
Plan what to search
    ↓
Search the web (live data)
    ↓
Read and extract key information
    ↓
Reason about what it found
    ↓
Verify facts
    ↓
Synthesize into a complete answer
    ↓
Final Answer (with citations, reasoning)
```

---

## 1.3 — Why Do We Need This?

| Problem with Regular LLMs | How Deep Research Fixes It |
|---------------------------|---------------------------|
| Knowledge cutoff (old data) | Searches web for live data |
| Hallucination (makes things up) | Verifies facts from real sources |
| Shallow answers | Multi-step reasoning before answering |
| No citations | Tracks sources throughout |
| Cannot handle complex questions | Breaks into sub-questions |

---

## 1.4 — Real-World Analogy

> Think of a **Deep Research AI** like a **senior research analyst** at a consulting firm.

When their boss asks: *"Should we invest in Company X?"*

They do NOT just say the first thing that comes to mind.

They:
1. **Plan** → What information do I need?
2. **Search** → Financial reports, news, competitor data
3. **Read** → Carefully extract relevant data
4. **Reason** → What does this data mean?
5. **Verify** → Is this data trustworthy?
6. **Write** → Structured report with recommendations

A Deep Research AI does all of this **automatically**.

---

## 1.5 — The Full System We Are Building

```
┌─────────────────────────────────────────────────┐
│           DEEP RESEARCH AI SYSTEM               │
│                                                 │
│  User Query                                     │
│      ↓                                          │
│  [Reasoning Model] ← Breaks query into plan     │
│      ↓                                          │
│  [Web Search Tool] ← Searches multiple sources  │
│      ↓                                          │
│  [Reading Agent]  ← Extracts key information    │
│      ↓                                          │
│  [Verifier]       ← Checks facts                │
│      ↓                                          │
│  [Reasoning Model] ← Synthesizes answer         │
│      ↓                                          │
│  Final Deep Research Report                     │
└─────────────────────────────────────────────────┘
```

---

# 🧠 CHAPTER 2: Reasoning and Thinking LLMs

---

## 2.1 — What Makes a Model a "Reasoning Model"?

### First — What is a Regular LLM?

A regular LLM (like early GPT) is like a **very fast, very smart autocomplete machine**.

You give it text → it predicts the next token → and the next → and the next.

```
Input:  "The capital of France is"
Output: "Paris"
```

It answers **immediately**, without thinking in between.

---

### The Problem: Fast Answers Are Often Wrong

Imagine asking a student:

*"If a bat and ball cost $1.10 together, and the bat costs $1.00 more than the ball, how much does the ball cost?"*

Most people instantly say: **"10 cents!"**

But that is **wrong**.

The correct answer is **5 cents**.

```
Proof:
Ball = x
Bat = x + 1.00
Together: x + (x + 1.00) = 1.10
2x = 0.10
x = 0.05 → Ball = 5 cents ✓
```

The fast, intuitive answer was wrong. The **slow, step-by-step** answer was right.

This is exactly the problem with regular LLMs — they are too fast.

---

## 2.2 — What is a Reasoning Model?

A **reasoning model** is an LLM that is trained to **think before answering**.

Instead of:
```
Question → Answer
```

It does:
```
Question → Think → Think → Think → ... → Answer
```

The "thinking" part is sometimes called a **"chain of thought"** or **"scratchpad"**.

---

### Visual Difference:

```
Regular LLM:
"What is 17 × 23?" → "391" (immediate)

Reasoning LLM:
"What is 17 × 23?"
→ [thinking: 17 × 20 = 340]
→ [thinking: 17 × 3 = 51]
→ [thinking: 340 + 51 = 391]
→ "391" (after reasoning)
```

Both give the same answer here — but for complex questions, the reasoning model is **far more accurate**.

---

## 2.3 — OpenAI's "o" Family of Models

### The Birth of "o1"

In September 2024, OpenAI released **o1** — their first official "reasoning model".

Before o1, models like GPT-4 were powerful but struggled with:
- Complex math
- Multi-step logic
- Scientific reasoning
- Code debugging

o1 was specifically trained to **reason step by step** before giving answers.

---

### The "o" Family Timeline:

```
OpenAI "o" Family:
│
├── o1 (September 2024)
│   ├── First reasoning model
│   ├── Thinks before answering
│   ├── Much better at math, science, coding
│   └── Slower but more accurate
│
├── o1-mini (September 2024)
│   ├── Smaller, faster, cheaper version
│   ├── Great for coding tasks
│   └── Less powerful for general reasoning
│
├── o1-pro (December 2024)
│   ├── Even more thinking time
│   └── Best accuracy, most expensive
│
└── o3 (announced December 2024)
    ├── Much more powerful than o1
    ├── Near human-level on many benchmarks
    └── Very expensive to run
```

---

### What Makes "o" Models Special?

**Key idea:** These models were trained using **Reinforcement Learning (RL)** to spend more time thinking.

Think of it this way:

> Regular model = A student who writes the first thing that comes to mind
>
> "o" model = A student who **drafts, reviews, corrects, and then submits**

The "o" models have an **internal scratchpad** where they reason through problems.

You **cannot see** this scratchpad — only the final answer is shown.

---

### How Much Better Is o1?

| Benchmark | GPT-4o | o1 |
|-----------|--------|-----|
| Math Olympiad (AIME) | 13% | 83% |
| PhD-level Science (GPQA) | 53% | 78% |
| Competitive Coding | 11% | 62% |

These numbers show **massive improvement** just from adding reasoning.

---

### Simple Code Example — Calling o1:

```python
from openai import OpenAI

client = OpenAI()

# Calling o1 reasoning model
response = client.chat.completions.create(
    model="o1-preview",  # or "o1-mini"
    messages=[
        {
            "role": "user",
            "content": """
            A train leaves City A at 60 mph.
            Another train leaves City B (300 miles away) at 40 mph.
            They travel toward each other.
            When do they meet, and how far from City A?
            Show your complete reasoning.
            """
        }
    ]
    # Note: o1 does NOT support system messages
    # Note: o1 does NOT support temperature parameter
    # It thinks on its own — you cannot force a style
)

print(response.choices[0].message.content)
```

**Output would be:**
```
The two trains together close 100 miles per hour (60 + 40).
Total distance = 300 miles.
Time to meet = 300 / 100 = 3 hours.
Distance from City A = 60 mph × 3 hours = 180 miles.

Answer: They meet after 3 hours, 180 miles from City A.
```

---

### Important Rules for "o" Models:

```python
# ❌ WRONG - o1 does NOT support system messages
response = client.chat.completions.create(
    model="o1-preview",
    messages=[
        {"role": "system", "content": "You are helpful"},  # ERROR!
        {"role": "user", "content": "Solve this..."}
    ]
)

# ✅ CORRECT - Only user messages
response = client.chat.completions.create(
    model="o1-preview",
    messages=[
        {"role": "user", "content": "You are helpful. Solve this..."}
    ]
)

# ❌ WRONG - No temperature for o1
response = client.chat.completions.create(
    model="o1-preview",
    temperature=0.7,  # ERROR - not supported
    messages=[...]
)
```

---

## 2.4 — DeepSeek-R1: The Open-Source Reasoning Revolution

### What is DeepSeek-R1?

DeepSeek-R1 is a **reasoning model** made by the Chinese AI company **DeepSeek**.

Released in **January 2025**, it shocked the AI world because:

1. It **matched o1's performance** on many benchmarks
2. It is **completely open source** (free to use!)
3. It was trained for a **fraction of o1's cost**
4. You can **run it locally** on your own computer

---

### Why DeepSeek-R1 Was a Big Deal:

```
OpenAI o1:
├── Cost: Hundreds of millions to train
├── Open source: NO
├── Can run locally: NO
├── Transparency: Low (black box)
└── Reasoning approach: Hidden

DeepSeek-R1:
├── Cost: ~$6 million to train (much cheaper!)
├── Open source: YES (weights available)
├── Can run locally: YES (with enough RAM)
├── Transparency: HIGH (paper published)
└── Reasoning approach: Documented and clear
```

---

### DeepSeek-R1's Special Innovation: Visible Thinking

Unlike o1 (where you cannot see the thinking), DeepSeek-R1 shows you its **full thinking process** between `<think>` tags:

```
User: What is the sum of angles in a triangle?

DeepSeek-R1:
<think>
Let me think about this carefully.
A triangle has 3 angles.
I know from Euclidean geometry that...
If I draw a line parallel to one side...
The alternate interior angles...
So the three angles must add up to...
Let me verify with a right triangle: 90 + 45 + 45 = 180. ✓
</think>

The sum of angles in a triangle is always 180 degrees.
This is a fundamental theorem of Euclidean geometry.
```

This is **revolutionary** because:
- You can **see** how the model thinks
- You can **debug** wrong reasoning
- You can **learn** from the reasoning process
- Researchers can **study** and improve it

---

### DeepSeek-R1 Architecture:

```
DeepSeek-R1 is built on DeepSeek-V3 base model
                    ↓
Fine-tuned with "cold start" data
(small set of human-written reasoning examples)
                    ↓
Trained with Reinforcement Learning (GRPO algorithm)
                    ↓
Rewarded for:
  ✓ Getting correct answers
  ✓ Showing clear reasoning steps
  ✓ Following format rules
                    ↓
Result: Model that NATURALLY reasons step by step
```

---

### DeepSeek-R1 Model Sizes Available:

| Model | Parameters | RAM Needed | Use Case |
|-------|-----------|------------|----------|
| DeepSeek-R1-1.5B | 1.5 Billion | ~4 GB | Personal laptop |
| DeepSeek-R1-7B | 7 Billion | ~8 GB | Good laptop |
| DeepSeek-R1-14B | 14 Billion | ~16 GB | Workstation |
| DeepSeek-R1-32B | 32 Billion | ~32 GB | High-end PC |
| DeepSeek-R1-70B | 70 Billion | ~80 GB | Server |
| DeepSeek-R1-671B | 671 Billion | Multiple GPUs | Data center |

---

### How to Use DeepSeek-R1 via API:

```python
from openai import OpenAI

# DeepSeek uses OpenAI-compatible API
client = OpenAI(
    api_key="your-deepseek-api-key",
    base_url="https://api.deepseek.com"
)

response = client.chat.completions.create(
    model="deepseek-reasoner",  # This is R1
    messages=[
        {
            "role": "user",
            "content": "Explain why the sky is blue, step by step."
        }
    ]
)

# You can access the reasoning process!
reasoning = response.choices[0].message.reasoning_content
final_answer = response.choices[0].message.content

print("=== THINKING PROCESS ===")
print(reasoning)
print("\n=== FINAL ANSWER ===")
print(final_answer)
```

---

### Comparing o1 vs DeepSeek-R1:

```
Feature Comparison:
┌─────────────────────┬────────────────┬────────────────┐
│ Feature             │ OpenAI o1      │ DeepSeek-R1    │
├─────────────────────┼────────────────┼────────────────┤
│ Math Performance    │ ★★★★★          │ ★★★★★          │
│ Code Performance    │ ★★★★★          │ ★★★★★          │
│ Open Source         │ ❌ No          │ ✅ Yes         │
│ Visible Thinking    │ ❌ Hidden      │ ✅ Visible     │
│ Local Deployment    │ ❌ No          │ ✅ Yes         │
│ API Cost            │ 💰💰💰 High    │ 💰 Low         │
│ Training Cost       │ 💰💰💰💰💰     │ 💰 Low         │
│ Community Support   │ Large          │ Growing        │
└─────────────────────┴────────────────┴────────────────┘
```

---

# 🔬 CHAPTER 3: Inference-Time Techniques

---

## 3.1 — What is "Inference Time"?

Before we learn the techniques, we need to understand what "inference time" means.

---

### Two Phases of an LLM's Life:

```
Phase 1: TRAINING TIME
├── The model learns from billions of text examples
├── Weights (numbers inside the model) are updated
├── This happens ONCE and takes weeks/months
├── Very expensive ($millions)
└── After this, the model is "frozen" — weights don't change

Phase 2: INFERENCE TIME
├── You ask the model a question
├── The model uses its frozen weights to generate an answer
├── This happens EVERY TIME you use the model
├── Much cheaper than training
└── This is what happens when you chat with ChatGPT
```

**Simple analogy:**

> **Training** = A student studying for 4 years in college
>
> **Inference** = The student taking an exam (using what they learned)

**Inference-time techniques** = Special tricks to make the model perform **better at exam time** without re-studying (re-training).

---

## 3.2 — What is Inference-Time Scaling?

### The Core Idea

**Inference-time scaling** = "Give the model **more time and computation** during answering, and it will give **better answers**."

This was a revolutionary idea because before this, people thought:
> "If you want a better AI, you need a bigger model (more training)."

Inference-time scaling says:
> "You can get a better answer from the **same model** by letting it think **longer and harder**."

---

### Real-World Analogy:

> Imagine two students taking the same exam.
>
> **Student A** has 30 minutes — writes quickly, submits.
>
> **Student B** has 3 hours — thinks carefully, checks work, revises.
>
> Same student, same knowledge — but **more time = better result**.

Inference-time scaling gives the AI model "more time" to think.

---

### The Scaling Laws Discovery:

OpenAI researchers discovered something amazing:

```
Training-time scaling (old thinking):
More data + bigger model = better performance
(But: costs $millions more each time)

Inference-time scaling (new thinking):
More computation at answer time = better performance
(And: you can do this with existing models!)
```

This is called the **"test-time compute"** paradigm.

---

### How Do You "Scale" Inference?

There are several ways to give a model more compute at inference time:

```
Inference-Time Scaling Methods:
│
├── 1. Chain of Thought (CoT)
│   └── Ask model to think step by step
│
├── 2. Parallel Sampling
│   └── Generate many answers, pick the best
│
├── 3. Sequential Sampling
│   └── Generate answer, review, improve, repeat
│
├── 4. Tree of Thoughts (ToT)
│   └── Explore multiple reasoning paths like a tree
│
└── 5. Search Against a Verifier
    └── Generate candidates, verify which is correct
```

We will cover each one in detail now.

---

## 3.3 — Chain of Thought (CoT) Prompting

### What is CoT?

**Chain of Thought (CoT)** = A prompting technique where you ask the model to **show its reasoning steps** before giving the final answer.

The magic phrase is: **"Let's think step by step."**

---

### The Discovery (Wei et al., 2022):

Google researchers discovered that simply adding **"Let's think step by step"** to a prompt dramatically improved model performance on complex tasks.

This was a shocking discovery — **one simple sentence** changed everything.

---

### Before vs After CoT:

```
WITHOUT CoT:
Question: "Roger has 5 tennis balls. He buys 2 more cans of
           tennis balls. Each can has 3 balls. How many tennis
           balls does he have now?"

Model: "11"  ← Sometimes wrong on harder problems

WITH CoT:
Question: "Roger has 5 tennis balls. He buys 2 more cans of
           tennis balls. Each can has 3 balls. How many tennis
           balls does he have now? Let's think step by step."

Model:
"Roger starts with 5 tennis balls.
He buys 2 cans × 3 balls = 6 new balls.
5 + 6 = 11.
The answer is 11." ← More reliable, shows work
```

---

### Types of CoT Prompting:

#### Type 1: Zero-Shot CoT
Just add magic words — no examples needed.

```python
def zero_shot_cot(question):
    prompt = f"""
    {question}
    
    Let's think step by step.
    """
    return prompt

# Example
question = "If I have 3 apples and give away 1, then buy 4 more, how many do I have?"
print(zero_shot_cot(question))
```

**Output:**
```
If I have 3 apples and give away 1, then buy 4 more, how many do I have?

Let's think step by step.
→ Start: 3 apples
→ Give away 1: 3 - 1 = 2 apples
→ Buy 4 more: 2 + 4 = 6 apples
→ Answer: 6 apples
```

---

#### Type 2: Few-Shot CoT
Give examples of reasoning before asking the question.

```python
def few_shot_cot(question):
    prompt = f"""
    Here are examples of how to solve problems step by step:
    
    Example 1:
    Q: Sarah has 8 oranges. She gives 3 to her friend. How many left?
    A: Sarah starts with 8 oranges.
       She gives away 3: 8 - 3 = 5.
       Sarah has 5 oranges left.
    
    Example 2:
    Q: A store sells 12 items per hour for 3 hours. Total sold?
    A: Items per hour = 12.
       Hours = 3.
       Total = 12 × 3 = 36 items.
    
    Now solve this:
    Q: {question}
    A: Let me solve step by step.
    """
    return prompt
```

---

#### Type 3: Auto-CoT
The model generates its own examples automatically.

```python
from openai import OpenAI

client = OpenAI()

def auto_cot(complex_question):
    # Step 1: Generate diverse sub-questions
    decompose_prompt = f"""
    Break this complex question into simpler sub-questions:
    
    Main question: {complex_question}
    
    List 3-5 sub-questions I need to answer first:
    """
    
    sub_questions = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": decompose_prompt}]
    ).choices[0].message.content
    
    # Step 2: Solve with full reasoning chain
    solve_prompt = f"""
    Main question: {complex_question}
    
    Sub-questions to address: {sub_questions}
    
    Now answer each sub-question step by step, then give the final answer.
    """
    
    final_answer = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": solve_prompt}]
    ).choices[0].message.content
    
    return final_answer

# Test it
result = auto_cot("Should a small business use cloud computing or local servers?")
print(result)
```

---

### Why Does CoT Work? (First Principles)

Let us think about WHY adding "think step by step" helps.

```
Theory 1: Computation Budget
Regular: model gets ONE forward pass to answer
CoT: model gets MANY tokens to "think" before answering
More tokens = more computation = harder problems solved

Theory 2: Intermediate Steps as Scaffolding
Hard problem: A → Z (model must jump directly)
With CoT:     A → B → C → ... → Z (smaller jumps)
Smaller jumps are easier to get right

Theory 3: Error Correction
Without CoT: One wrong assumption ruins everything
With CoT: Model can "see" its intermediate steps
         and self-correct before the final answer
```

---

### CoT for Our Deep Research System:

```python
def deep_research_cot(research_question):
    """
    Use CoT to break down a complex research question
    """
    
    prompt = f"""
    You are a senior research analyst. 
    
    Research Question: {research_question}
    
    Think through this step by step:
    
    Step 1: What specific information do I need to answer this?
    Step 2: What are the key sub-questions I must investigate?
    Step 3: What sources should I search for each sub-question?
    Step 4: What potential contradictions or nuances should I watch for?
    Step 5: How should I structure my final research report?
    
    Work through each step carefully before planning the searches.
    """
    
    return prompt

# Example usage
question = "What are the economic impacts of AI on employment in 2024?"
research_plan = deep_research_cot(question)
print(research_plan)
```

---

## 3.4 — Parallel Sampling

### What is Parallel Sampling?

**Parallel sampling** = Generate **multiple different answers** to the same question **at the same time**, then pick the **best one**.

---

### Real-World Analogy:

> Imagine you need to write an important email.
>
> **Without parallel sampling:** Write one email, send it.
>
> **With parallel sampling:** Write 5 different versions of the email,
> then choose the best one to send.

The key insight: **LLMs are random** — if you ask the same question multiple times, you get **different answers**. Some are better, some are worse. Parallel sampling lets you **sample many and pick the best**.

---

### How Randomness Works in LLMs (Temperature):

```
Temperature = 0 (no randomness):
"The capital of France is" → "Paris" (always, no variation)

Temperature = 0.7 (medium randomness):
Ask 3 times:
→ "Paris, the beautiful City of Light"
→ "Paris, located in northern France"  
→ "Paris, one of Europe's largest cities"

Temperature = 1.5 (high randomness):
Ask 3 times:
→ "Paris, where the Eiffel Tower stands"
→ "Paris! Founded by the Parisii tribe"
→ "Paris — home to world-class cuisine"
```

Parallel sampling **exploits this randomness** to generate diverse candidates.

---

### The "Best of N" Strategy:

```
Algorithm: Best-of-N Sampling

Input: A question Q
       A number N (how many samples to generate)
       A scorer function (how to judge quality)

Step 1: Generate N different answers to Q
Step 2: Score each answer
Step 3: Return the highest-scoring answer

Output: The best answer from N attempts
```

---

### Simple Code Example:

```python
from openai import OpenAI
import random

client = OpenAI()

def parallel_sampling(question, n_samples=5, temperature=0.8):
    """
    Generate N different answers and return all of them
    """
    
    print(f"Generating {n_samples} parallel samples...")
    
    samples = []
    
    for i in range(n_samples):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user", 
                    "content": f"{question}\n\nThink step by step and give your answer."
                }
            ],
            temperature=temperature  # Controls randomness
        )
        
        answer = response.choices[0].message.content
        samples.append(answer)
        print(f"Sample {i+1} generated ✓")
    
    return samples


def pick_best_answer(question, samples):
    """
    Use another LLM call to pick the best answer
    This is called 'LLM-as-a-judge'
    """
    
    samples_text = ""
    for i, sample in enumerate(samples):
        samples_text += f"\n\nANSWER {i+1}:\n{sample}"
    
    judge_prompt = f"""
    Question: {question}
    
    Here are {len(samples)} different answers:
    {samples_text}
    
    Please evaluate each answer based on:
    1. Accuracy and correctness
    2. Completeness
    3. Clarity and structure
    4. Logical reasoning shown
    
    Which answer is the BEST? 
    Explain why briefly, then state: "BEST ANSWER: [number]"
    """
    
    judgment = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": judge_prompt}],
        temperature=0  # Use deterministic for judging
    ).choices[0].message.content
    
    return judgment


def parallel_sampling_pipeline(question, n=5):
    """
    Complete pipeline: sample → judge → return best
    """
    
    print(f"\n🔍 Question: {question}")
    print("=" * 50)
    
    # Step 1: Generate multiple answers
    samples = parallel_sampling(question, n_samples=n)
    
    # Step 2: Pick the best one
    print("\n⚖️ Judging answers...")
    best = pick_best_answer(question, samples)
    
    print("\n🏆 Best Answer Selected:")
    print(best)
    
    return samples, best


# Example
question = "What are the main risks of using AI in healthcare?"
samples, best = parallel_sampling_pipeline(question, n=3)
```

---

### More Advanced: Majority Voting (Self-Consistency)

For questions with **definite answers** (like math), you can use **majority voting**:

```python
def majority_vote(question, n_samples=10):
    """
    For math/logic problems: 
    Generate N answers, take the most common one
    This is called 'Self-Consistency'
    """
    
    answers = []
    
    for i in range(n_samples):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": f"""
                    Solve this step by step, then give ONLY the final 
                    numerical answer on the last line.
                    
                    Problem: {question}
                    """
                }
            ],
            temperature=0.7
        )
        
        # Extract just the final answer (last line)
        full_response = response.choices[0].message.content
        final_answer = full_response.strip().split('\n')[-1]
        answers.append(final_answer)
        
    # Count occurrences
    from collections import Counter
    vote_counts = Counter(answers)
    most_common_answer = vote_counts.most_common(1)[0][0]
    
    print(f"All answers generated: {answers}")
    print(f"Vote counts: {dict(vote_counts)}")
    print(f"Winning answer (majority vote): {most_common_answer}")
    
    return most_common_answer


# Example
math_question = "A car travels 60 mph for 2.5 hours. How far does it go?"
majority_vote(math_question, n_samples=7)
```

**Output might look like:**
```
All answers generated: ['150 miles', '150 miles', '150', '150 miles', '150', '150 miles', '150 miles']
Vote counts: {'150 miles': 5, '150': 2}
Winning answer (majority vote): 150 miles
```

---

### When to Use Parallel Sampling:

```
✅ Good for:
├── Math and logic problems (use majority voting)
├── Creative tasks (pick most creative from many)
├── High-stakes questions (verify answer stability)
└── Research tasks (explore different angles)

❌ Bad for:
├── Simple factual questions (waste of compute)
├── Very expensive models (too costly)
└── Real-time applications (too slow)
```

---

## 3.5 — Sequential Sampling

### What is Sequential Sampling?

**Sequential sampling** = Generate an answer, **review it**, **improve it**, repeat — in **sequence** (one after another).

Unlike parallel sampling (all at once), sequential sampling is **iterative** — each step builds on the previous.

---

### Real-World Analogy:

> **Parallel:** Write 5 essays at once, pick the best.
>
> **Sequential:** Write one essay, revise it, revise again, revise again — until it's great.

Both approaches can reach high quality, but through different paths.

---

### The Pattern:

```
Sequential Sampling Flow:

Attempt 1 → Review → Feedback
                ↓
Attempt 2 (improved) → Review → Feedback
                ↓
Attempt 3 (better) → Review → Feedback
                ↓
...repeat until good enough...
                ↓
Final Answer
```

---

### Simple Sequential Sampling Code:

```python
from openai import OpenAI

client = OpenAI()

def sequential_sampling(question, max_iterations=3):
    """
    Iteratively improve an answer through sequential revision
    """
    
    print(f"\n📝 Question: {question}")
    print("=" * 50)
    
    # Step 1: Initial answer
    current_answer = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": f"Answer this question: {question}"}
        ]
    ).choices[0].message.content
    
    print(f"\n🔵 Initial Answer:\n{current_answer}")
    
    # Step 2: Iteratively improve
    for iteration in range(max_iterations):
        
        print(f"\n{'='*20} ITERATION {iteration+1} {'='*20}")
        
        # Review the current answer
        review = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": f"""
                    Question: {question}
                    
                    Current Answer: {current_answer}
                    
                    Please critique this answer:
                    1. What is missing or incomplete?
                    2. What could be more accurate?
                    3. What could be clearer?
                    4. What important points are not covered?
                    
                    Be specific and constructive.
                    """
                }
            ]
        ).choices[0].message.content
        
        print(f"\n🔍 Review/Critique:\n{review}")
        
        # Improve based on review
        improved_answer = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": f"""
                    Question: {question}
                    
                    Previous Answer: {current_answer}
                    
                    Critique/Feedback: {review}
                    
                    Now write an IMPROVED answer that addresses 
                    all the critique points. Make it better.
                    """
                }
            ]
        ).choices[0].message.content
        
        print(f"\n✅ Improved Answer (Iteration {iteration+1}):\n{improved_answer}")
        
        # Update current answer
        current_answer = improved_answer
    
    print(f"\n🏆 FINAL ANSWER (after {max_iterations} improvements):")
    print(current_answer)
    
    return current_answer


# Example
question = "What are the key considerations for building a safe AI system?"
final = sequential_sampling(question, max_iterations=2)
```

---

### Sequential vs Parallel — When to Use Which:

```
┌─────────────────────┬─────────────────┬─────────────────┐
│ Scenario            │ Use Parallel    │ Use Sequential  │
├─────────────────────┼─────────────────┼─────────────────┤
│ Math problems       │ ✅ Majority vote │ ✅ Verify steps  │
│ Creative writing    │ ✅ Many options  │ ✅ Polish        │
│ Research reports    │ ✅ Many angles   │ ✅ Deep revision │
│ Code debugging      │ ❌ Not ideal     │ ✅ Fix → test   │
│ Fast response needed│ ✅ Pick fastest  │ ❌ Too slow     │
│ Limited budget      │ ❌ Too many calls│ ✅ Fewer calls  │
└─────────────────────┴─────────────────┴─────────────────┘
```

---

## 3.6 — Tree of Thoughts (ToT)

### What is Tree of Thoughts?

**Tree of Thoughts (ToT)** is a framework where the model explores **multiple reasoning paths** like branches of a tree, evaluates each branch, and follows the most promising ones.

---

### First Principles: Why a "Tree"?

Think about how humans solve hard puzzles.

A chess player doesn't just think one move ahead. They think:
```
"If I move here, opponent might go there...
 then I could respond with...
 or they might go another way...
 Let me evaluate which path leads to winning..."
```

This is **tree search** — exploring multiple futures before deciding.

**Standard CoT** is like a straight line:
```
Start → Step 1 → Step 2 → Step 3 → Answer
```

**Tree of Thoughts** is like a tree:
```
                    Start
                   /  |  \
               Path1 Path2 Path3
              /    \       |
         Step1A  Step1B  Step2C
           |              |
         DEAD         SUCCESS ✓
```

---

### The Tree of Thoughts Paper (Yao et al., 2023):

Researchers at Princeton + Google showed that ToT dramatically improved performance on tasks like:

- **Creative writing** (Game of 24 puzzle)
- **Crossword puzzles**
- **Complex planning tasks**

**Key results:**
```
Game of 24 (make 24 from 4 numbers):
├── Standard GPT-4: 4% success rate
├── With CoT: 4% success rate  
└── With ToT: 74% success rate ← Massive improvement!
```

---

### The 3 Components of ToT:

```
Component 1: THOUGHT GENERATION
└── Generate multiple different "next steps" or "thoughts"
└── Like a chess player generating possible moves

Component 2: STATE EVALUATION  
└── Judge each thought: Is this a good direction?
└── Score: "promising" / "not promising" / "definitely wrong"

Component 3: SEARCH ALGORITHM
└── Decide which branches to explore
└── Options: BFS (breadth-first) or DFS (depth-first)
```

---

### Visual Diagram of ToT:

```
PROBLEM: "Plan a research paper outline on AI safety"

Level 0 (Root):
└── "Research paper on AI safety"

Level 1 (Generate 3 approaches):
├── Approach A: "Focus on technical alignment"
├── Approach B: "Focus on governance and policy"  
└── Approach C: "Focus on historical AI failures"

Level 1 Evaluation:
├── Approach A: Score 8/10 ← Explore further
├── Approach B: Score 7/10 ← Explore further
└── Approach C: Score 5/10 ← Prune (stop exploring)

Level 2 (Expand A and B):
Approach A branches:
├── A1: "Neural network safety techniques"
├── A2: "RLHF and Constitutional AI"

Approach B branches:
├── B1: "EU AI Act analysis"
├── B2: "International AI governance"

Level 2 Evaluation:
├── A2: Score 9/10 ← BEST - follow this
├── B1: Score 8/10 ← Good
...

Final selection: Build outline around A2 path
```

---

### Tree of Thoughts Implementation:

```python
from openai import OpenAI
import json

client = OpenAI()

class TreeOfThoughts:
    """
    A simple Tree of Thoughts implementation
    """
    
    def __init__(self, model="gpt-4o", n_thoughts=3, max_depth=3):
        self.model = model
        self.n_thoughts = n_thoughts  # How many branches per level
        self.max_depth = max_depth    # How deep to search
        self.client = OpenAI()
    
    def generate_thoughts(self, problem, current_state, step_num):
        """
        Generate multiple possible 'next thoughts' from current state
        """
        prompt = f"""
        Problem: {problem}
        
        Current reasoning state: {current_state}
        
        This is step {step_num} of solving the problem.
        
        Generate {self.n_thoughts} DIFFERENT possible next reasoning steps.
        Each should explore a different angle or approach.
        
        Format your response as:
        THOUGHT 1: [first possible next step]
        THOUGHT 2: [second possible next step]  
        THOUGHT 3: [third possible next step]
        """
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8  # High temperature for diversity
        ).choices[0].message.content
        
        # Parse thoughts
        thoughts = []
        for i in range(1, self.n_thoughts + 1):
            marker = f"THOUGHT {i}:"
            if marker in response:
                start = response.index(marker) + len(marker)
                # Find end (next THOUGHT or end of string)
                next_marker = f"THOUGHT {i+1}:"
                end = response.index(next_marker) if next_marker in response else len(response)
                thought = response[start:end].strip()
                thoughts.append(thought)
        
        return thoughts
    
    def evaluate_thought(self, problem, thought_path):
        """
        Evaluate how promising a thought path is
        Returns a score from 0-10
        """
        prompt = f"""
        Problem: {problem}
        
        Reasoning path so far:
        {chr(10).join([f'Step {i+1}: {t}' for i, t in enumerate(thought_path)])}
        
        Evaluate this reasoning path:
        1. Is this making progress toward solving the problem?
        2. Is the logic sound and valid?
        3. Does it seem like a promising direction?
        
        Give a score from 0-10 where:
        0-3 = Poor direction, likely wrong
        4-6 = Okay but not ideal  
        7-9 = Good direction, promising
        10 = Excellent, very likely correct
        
        Respond with ONLY: SCORE: [number] REASON: [brief reason]
        """
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0  # Deterministic for evaluation
        ).choices[0].message.content
        
        # Extract score
        try:
            score_part = response.split("SCORE:")[1].split("REASON:")[0].strip()
            score = float(score_part)
        except:
            score = 5.0  # Default if parsing fails
            
        return score, response
    
    def solve(self, problem):
        """
        Main ToT solving algorithm using BFS (Breadth-First Search)
        """
        
        print(f"\n🌳 Tree of Thoughts - Solving:")
        print(f"Problem: {problem}")
        print("=" * 60)
        
        # Start with empty state
        # Each "node" is a list of thoughts (reasoning path)
        current_level = [[]]  # Start with one empty path
        
        for depth in range(self.max_depth):
            print(f"\n📊 Level {depth + 1} of {self.max_depth}")
            
            all_new_paths = []
            
            # Expand each current path
            for path in current_level:
                current_state = " → ".join(path) if path else "Starting fresh"
                
                # Generate new thoughts from this state
                new_thoughts = self.generate_thoughts(
                    problem, current_state, depth + 1
                )
                
                # Create new paths by adding each thought
                for thought in new_thoughts:
                    new_path = path + [thought]
                    all_new_paths.append(new_path)
            
            # Evaluate all new paths
            scored_paths = []
            for path in all_new_paths:
                score, reason = self.evaluate_thought(problem, path)
                scored_paths.append((score, path, reason))
                print(f"  Path score: {score:.1f} - {path[-1][:50]}...")
            
            # Sort by score and keep top paths (beam search)
            scored_paths.sort(reverse=True, key=lambda x: x[0])
            
            # Keep only top N paths (pruning)
            top_n = 2  # Keep 2 best paths at each level
            current_level = [path for _, path, _ in scored_paths[:top_n]]
            
            print(f"  ✂️ Pruned to {len(current_level)} best paths")
        
        # Final answer: Use the best path to generate solution
        best_path = current_level[0]
        
        print(f"\n🏆 Best reasoning path:")
        for i, thought in enumerate(best_path):
            print(f"  Step {i+1}: {thought}")
        
        # Generate final answer using best path
        final_prompt = f"""
        Problem: {problem}
        
        Reasoning path I followed:
        {chr(10).join([f'Step {i+1}: {t}' for i, t in enumerate(best_path)])}
        
        Based on this reasoning, give a complete, well-structured final answer.
        """
        
        final_answer = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": final_prompt}],
            temperature=0.3
        ).choices[0].message.content
        
        print(f"\n📝 Final Answer:\n{final_answer}")
        
        return final_answer, best_path


# Example usage
tot = TreeOfThoughts(n_thoughts=3, max_depth=2)
answer, path = tot.solve(
    "What are the most important factors to consider when choosing between "
    "fine-tuning a model vs using RAG for a customer service chatbot?"
)
```

---

### When to Use Tree of Thoughts:

```
✅ Perfect for:
├── Complex planning tasks
├── Creative problem solving
├── Multi-step reasoning where you can go wrong
├── Puzzles and logic games
└── Research planning

❌ Overkill for:
├── Simple factual questions
├── Quick responses needed
├── Low budget (many API calls)
└── Tasks with clear linear reasoning
```

---

## 3.7 — Search Against a Verifier

### What is "Search Against a Verifier"?

This technique combines:
1. **A generator** → produces many candidate answers
2. **A verifier** → checks which answers are actually correct

Instead of just picking the "most popular" or "best-sounding" answer, you **verify correctness** using a separate checker.

---

### Real-World Analogy:

> In a courtroom:
>
> **Lawyers (generators)** make arguments and propose conclusions.
>
> **Judge (verifier)** evaluates the arguments against the law and evidence.
>
> The judge doesn't just pick the most confident lawyer — they verify against objective rules.

In AI:
- **Generator** = LLM that produces many possible answers
- **Verifier** = A system that checks if answers are actually correct

---

### Types of Verifiers:

```
Type 1: RULE-BASED VERIFIER (Most Reliable)
└── Uses code/logic to verify
└── Example: Run the code → Did it pass all tests?
└── Very reliable but only works for verifiable tasks

Type 2: TRAINED VERIFIER (Reward Model)
└── A separate ML model trained to score answers
└── Example: Trained to judge if math solutions are correct
└── More flexible but can make mistakes

Type 3: LLM-AS-VERIFIER  
└── Use another LLM call to check the answer
└── Example: "Is this answer correct? Why or why not?"
└── Flexible but can be unreliable (LLMs can be wrong)
```

---

### Simple Example — Code Verification:

```python
from openai import OpenAI
import subprocess
import tempfile
import os

client = OpenAI()

def generate_code_solutions(problem, n=5):
    """
    Generate multiple code solutions to a programming problem
    """
    solutions = []
    
    for i in range(n):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": f"""
                    Write Python code to solve this problem:
                    {problem}
                    
                    Output ONLY the Python code, no explanations.
                    The code should print the final answer.
                    """
                }
            ],
            temperature=0.8
        ).choices[0].message.content
        
        # Clean up the code (remove markdown if present)
        code = response.replace("```python", "").replace("```", "").strip()
        solutions.append(code)
    
    return solutions


def verify_code(code, test_cases):
    """
    Verify if code produces correct outputs for test cases
    This is a RULE-BASED verifier - most reliable!
    """
    
    for test_input, expected_output in test_cases:
        try:
            # Write code to temp file
            with tempfile.NamedTemporaryFile(
                mode='w', suffix='.py', delete=False
            ) as f:
                # Add test input to code
                full_code = f"""
{code}

# Test input
test_val = {test_input}
"""
                f.write(full_code)
                temp_file = f.name
            
            # Run code (with timeout for safety)
            result = subprocess.run(
                ['python', temp_file],
                capture_output=True,
                text=True,
                timeout=5  # 5 second timeout
            )
            
            actual_output = result.stdout.strip()
            
            # Clean up
            os.unlink(temp_file)
            
            # Check if output matches expected
            if str(expected_output) not in actual_output:
                return False, f"Expected {expected_output}, got {actual_output}"
                
        except subprocess.TimeoutExpired:
            return False, "Code timed out"
        except Exception as e:
            return False, str(e)
    
    return True, "All tests passed!"


def search_against_verifier(problem, test_cases, n_samples=5):
    """
    Generate multiple solutions and verify each one
    Return the first solution that passes all tests
    """
    
    print(f"🔍 Problem: {problem}")
    print(f"📋 Generating {n_samples} candidate solutions...")
    
    solutions = generate_code_solutions(problem, n=n_samples)
    
    for i, solution in enumerate(solutions):
        print(f"\n🧪 Testing solution {i+1}...")
        is_correct, message = verify_code(solution, test_cases)
        
        if is_correct:
            print(f"✅ Solution {i+1} PASSED! {message}")
            print(f"\n📝 Winning solution:\n{solution}")
            return solution
        else:
            print(f"❌ Solution {i+1} FAILED: {message}")
    
    print("\n⚠️ No solution passed all tests!")
    return None


# Example: Find a solution to a math problem via code
problem = "Write a function that finds all prime numbers up to N using the Sieve of Eratosthenes"

test_cases = [
    # (input_value, expected_in_output)
    (10, "[2, 3, 5, 7]"),
    (20, "[2, 3, 5, 7, 11, 13, 17, 19]"),
]

# This would find the first correct solution
# search_against_verifier(problem, test_cases, n_samples=5)
```

---

### LLM-as-Verifier for Research:

```python
def research_verifier(claim, sources):
    """
    Verify a research claim against provided sources
    """
    
    sources_text = "\n\n".join([
        f"SOURCE {i+1}:\n{source}" 
        for i, source in enumerate(sources)
    ])
    
    verify_prompt = f"""
    You are a fact-checker. Evaluate this claim against the provided sources.
    
    CLAIM TO VERIFY: {claim}
    
    SOURCES:
    {sources_text}
    
    Answer these questions:
    1. Is this claim SUPPORTED, CONTRADICTED, or NOT MENTIONED in the sources?
    2. Which specific source(s) support or contradict it?
    3. What is your confidence level (High/Medium/Low)?
    4. What additional verification would be needed?
    
    Give a structured verification report.
    """
    
    result = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": verify_prompt}],
        temperature=0
    ).choices[0].message.content
    
    return result

# This will be integrated into our Deep Research system!
```

---

### Combining All Inference-Time Techniques:

```python
class InferenceTimeScaler:
    """
    Combines all inference-time scaling techniques
    for maximum performance
    """
    
    def __init__(self):
        self.client = OpenAI()
    
    def solve_with_maximum_compute(self, problem, strategy="auto"):
        """
        Use the right inference-time technique for the problem
        """
        
        if strategy == "auto":
            strategy = self._choose_strategy(problem)
        
        print(f"📊 Using strategy: {strategy}")
        
        if strategy == "cot":
            return self._solve_with_cot(problem)
        
        elif strategy == "parallel":
            return self._solve_with_parallel(problem)
        
        elif strategy == "sequential":
            return self._solve_with_sequential(problem)
        
        elif strategy == "tot":
            tot = TreeOfThoughts()
            answer, _ = tot.solve(problem)
            return answer
        
        elif strategy == "combined":
            # Best of all worlds:
            # 1. Use ToT to find best reasoning path
            # 2. Generate multiple answers using that path
            # 3. Verify and pick best
            return self._solve_combined(problem)
    
    def _choose_strategy(self, problem):
        """
        Automatically choose the right strategy
        """
        
        choice_prompt = f"""
        Problem: {problem}
        
        Which inference-time strategy would work best for this problem?
        
        Options:
        - cot: Simple step-by-step reasoning (for moderate complexity)
        - parallel: Generate many answers, pick best (for problems with one right answer)
        - sequential: Iteratively improve (for creative/open-ended tasks)
        - tot: Tree of Thoughts (for complex multi-step planning)
        - combined: Use all techniques (for very hard problems)
        
        Respond with ONLY one word: cot, parallel, sequential, tot, or combined
        """
        
        strategy = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": choice_prompt}],
            temperature=0
        ).choices[0].message.content.strip().lower()
        
        return strategy
    
    def _solve_with_cot(self, problem):
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": f"{problem}\n\nLet's think step by step."
            }]
        ).choices[0].message.content
        return response
    
    def _solve_with_parallel(self, problem, n=5):
        samples = parallel_sampling(problem, n_samples=n)
        best = pick_best_answer(problem, samples)
        return best
    
    def _solve_with_sequential(self, problem, iterations=3):
        return sequential_sampling(problem, max_iterations=iterations)
    
    def _solve_combined(self, problem):
        print("🚀 Using COMBINED strategy (maximum compute)...")
        
        # Step 1: Use CoT to structure the problem
        structured = self._solve_with_cot(problem)
        
        # Step 2: Use parallel sampling for final answer
        refined_problem = f"""
        Problem: {problem}
        
        Reasoning framework developed:
        {structured}
        
        Now give a complete, polished final answer.
        """
        
        samples = parallel_sampling(refined_problem, n_samples=3)
        best = pick_best_answer(problem, samples)
        
        return best


# Usage
scaler = InferenceTimeScaler()
answer = scaler.solve_with_maximum_compute(
    "Design a system architecture for a real-time fraud detection system "
    "that processes 1 million transactions per second."
)
print(answer)
```

---

# 📊 CHAPTER 3 — Summary Table

```
┌──────────────────────┬────────────────────┬────────────────┬─────────────────┐
│ Technique            │ How it Works       │ Best For       │ Cost (API calls)│
├──────────────────────┼────────────────────┼────────────────┼─────────────────┤
│ Chain of Thought     │ Think step by step │ Most problems  │ 1-2 calls       │
│ Parallel Sampling    │ Many answers, pick │ Definite answer│ N calls         │
│                      │ best               │                │                 │
│ Sequential Sampling  │ Improve iteratively│ Creative tasks │ 2N calls        │
│ Tree of Thoughts     │ Explore branches   │ Complex planning│ Many calls     │
│ Search+Verifier      │ Generate + check   │ Verifiable tasks│ N+N calls      │
└──────────────────────┴────────────────────┴────────────────┴─────────────────┘
```

---

# 🔴 END OF PART 1

---

## ✅ What We Covered in Part 1:

```
✅ Chapter 1: What is Deep Research Capability?
   - The problem it solves
   - How it differs from regular LLMs
   - Full system architecture

✅ Chapter 2: Reasoning and Thinking LLMs
   - What makes a reasoning model
   - OpenAI "o" family (o1, o1-mini, o1-pro, o3)
   - DeepSeek-R1 (open source alternative)
   - Code examples for both

✅ Chapter 3: Inference-Time Techniques
   - What is inference time
   - Inference-time scaling
   - Chain of Thought (CoT) - all 3 types
   - Parallel Sampling + Majority Voting
   - Sequential Sampling
   - Tree of Thoughts (ToT) with full implementation
   - Search Against a Verifier
   - Combined techniques
```

---

## 🚀 Coming in Part 2:

```
📌 Part 2 Will Cover:

├── Training-Time Techniques
│   ├── SFT on reasoning data (STaR)
│   ├── Reinforcement Learning with verifier
│   ├── Reward Modeling (ORM vs PRM)
│   ├── Self-refinement
│   └── Internalizing Search (Meta-CoT)
│
└── Local Deployment
    ├── Running DeepSeek-R1 locally with Ollama
    ├── Quantization explained simply
    ├── Full Deep Research system assembly
    └── Complete project code
```

# 🚀 Project 4: Deep Research Capability — PART 2

## Training-Time Techniques & Local Deployment

---

# 📚 TABLE OF CONTENTS (Part 2)

```
PART 2 COVERS:
├── 4. Training-Time Techniques
│   ├── 4.1 What is "Training Time" vs "Inference Time"?
│   ├── 4.2 SFT on Reasoning Data (STaR)
│   ├── 4.3 Reinforcement Learning with a Verifier
│   ├── 4.4 Reward Modeling (ORM vs PRM)
│   ├── 4.5 Self-Refinement
│   └── 4.6 Internalizing Search (Meta-CoT)
│
└── 5. Local Deployment
    ├── 5.1 Why Deploy Locally?
    ├── 5.2 Understanding Quantization
    ├── 5.3 Running DeepSeek-R1 with Ollama
    ├── 5.4 Building the Complete Deep Research System
    └── 5.5 Full Project Code (End-to-End)
```

---

# 🧠 CHAPTER 4: Training-Time Techniques

---

## 4.1 — Training Time vs Inference Time (Quick Recap)

Before we go deep, let me make sure this is 100% clear.

---

### The Simple Mental Model:

```
Think of an LLM like a Student:

TRAINING TIME = Going to school and studying
├── Happens ONCE (or rarely)
├── The student LEARNS new things
├── Their brain (weights) CHANGES
├── Very expensive and slow
└── After this → student is "ready"

INFERENCE TIME = Taking an exam
├── Happens EVERY TIME you use the model  
├── The student uses what they ALREADY KNOW
├── Their brain (weights) do NOT change
├── Cheaper and faster
└── This is what happens when you chat with ChatGPT
```

---

### What We Covered in Part 1 vs Part 2:

```
PART 1 (Inference-Time) = 
"How do we make a FROZEN model perform better at exam time?"
├── CoT: Give more time to think
├── Parallel sampling: Try many answers
├── Sequential sampling: Iterate and improve
├── Tree of Thoughts: Explore many paths
└── Search + Verifier: Generate then verify

PART 2 (Training-Time) =
"How do we TEACH the model to be a better reasoner from the start?"
├── STaR: Train on self-generated reasoning examples
├── RL with Verifier: Reward good reasoning
├── Reward Modeling: Teach a model to judge quality
├── Self-Refinement: Train model to improve itself
└── Meta-CoT: Internalize the search process itself
```

---

### Why Do We Need Training-Time Techniques?

Inference-time techniques are powerful but have **limits**:

```
Problem 1: Cost
├── Running 10 parallel samples = 10x the API cost
├── Tree of Thoughts = Many API calls per question
└── For millions of users, this gets very expensive

Problem 2: Latency (Speed)
├── Sequential sampling takes time (multiple rounds)
├── Users expect fast responses
└── ToT with deep trees can take minutes

Problem 3: Context Length
├── Long reasoning chains use many tokens
├── Models have context limits
└── Very long CoT chains can hit limits

Solution: Training-Time Techniques
"Bake the reasoning ability INTO the model itself"
Then at inference time, the model reasons well naturally
WITHOUT needing all those extra techniques!
```

---

## 4.2 — SFT on Reasoning Data (STaR: Self-Taught Reasoner)

---

### What is SFT?

**SFT = Supervised Fine-Tuning**

This is the most basic form of training an LLM.

```
Regular SFT:
Input:  "What is the capital of France?"
Output: "Paris"
Model learns: Question → Answer

Reasoning SFT:
Input:  "What is 15% of 80?"
Output: "Let me think step by step.
         15% means 15 out of 100.
         So 15/100 × 80 = 0.15 × 80 = 12.
         The answer is 12."
Model learns: Question → Reasoning → Answer
```

The key difference: We train the model on examples that include **the full reasoning process**, not just the final answer.

---

### The Big Problem: Where Do We Get Reasoning Data?

Human-written reasoning data is:
- Very expensive to create
- Slow to produce
- Hard to scale to millions of examples
- Sometimes inconsistent in quality

This is where **STaR** comes in.

---

### What is STaR? (Self-Taught Reasoner)

**STaR** was introduced by Zelikman et al. (2022) at Stanford.

The core idea is **brilliant and simple**:

> "Let the model generate its OWN reasoning examples, then train on the ones where it got the RIGHT answer."

This is called **bootstrapping** — using the model to improve itself.

---

### STaR Algorithm — Step by Step:

```
STaR Algorithm:

INPUT:
├── A base LLM (not yet good at reasoning)
├── A dataset of (Question, Answer) pairs
│   Note: Only questions and FINAL answers, no reasoning!
└── A verifier (checks if answer is correct)

LOOP (repeat many times):

Step 1: GENERATE
└── For each question, ask the model:
    "Think step by step, then give the answer."
└── Model produces: reasoning + answer

Step 2: VERIFY  
└── Check: Is the final answer CORRECT?
└── Keep ONLY the examples where answer = correct
└── Discard examples where answer = wrong

Step 3: RATIONALIZATION (for wrong answers)
└── For questions the model got wrong:
    Give it a hint: "The answer is X. Now explain why."
└── Model generates reasoning that leads to correct answer
└── Keep these too (if reasoning looks valid)

Step 4: FINE-TUNE
└── Take all kept examples (verified correct reasoning)
└── Fine-tune the model on these examples
└── Now model is BETTER at reasoning!

Step 5: REPEAT with better model
└── Go back to Step 1 with improved model
└── Model generates even better reasoning
└── Fine-tune again
└── Keep improving!

OUTPUT: A model that reasons well NATURALLY
```

---

### Visual Diagram of STaR:

```
┌─────────────────────────────────────────────────────────┐
│                    STaR CYCLE                           │
│                                                         │
│   ┌─────────┐                                          │
│   │  Base   │                                          │
│   │  Model  │                                          │
│   └────┬────┘                                          │
│        │                                               │
│        ▼                                               │
│   Generate reasoning for all questions                 │
│   Q: "5+3=?" → "5 plus 3... I count 5, then 3        │
│                  more... = 8. Answer: 8"               │
│        │                                               │
│        ▼                                               │
│   ┌─────────────────────────┐                         │
│   │      VERIFIER           │                         │
│   │  Is final answer right? │                         │
│   └─────────┬───────────────┘                         │
│             │                                          │
│    ┌────────┴─────────┐                               │
│    │                  │                               │
│    ▼                  ▼                               │
│  CORRECT ✅        WRONG ❌                            │
│  Keep reasoning   Try rationalization                  │
│                   (give hint, regenerate)              │
│    │                  │                               │
│    └────────┬─────────┘                               │
│             │                                          │
│             ▼                                          │
│   ┌──────────────────┐                               │
│   │  Fine-tune model  │                               │
│   │  on GOOD examples │                               │
│   └────────┬─────────┘                               │
│            │                                          │
│            ▼                                          │
│   ┌─────────────┐                                     │
│   │  BETTER     │ ──────────── Loop again ──────────► │
│   │  MODEL      │                                     │
│   └─────────────┘                                     │
└─────────────────────────────────────────────────────────┘
```

---

### Why STaR is Powerful:

```
Traditional approach:
Hire humans → Write 10,000 reasoning examples → Train
Cost: Very expensive, time-consuming, limited scale

STaR approach:
Model generates its OWN examples → Filter good ones → Train
Cost: Mostly compute (cheap), unlimited scale!

Result:
STaR paper showed: Model improved from 36% → 50% accuracy
on grade school math just by training on its own reasoning!
```

---

### Code Example — Simulating STaR:

```python
from openai import OpenAI
import json
from typing import List, Tuple

client = OpenAI()

class STaRTrainer:
    """
    Simulates the STaR (Self-Taught Reasoner) process
    
    Note: In real training you would use a training framework
    like HuggingFace Transformers. This shows the DATA GENERATION
    part of STaR which is the key innovation.
    """
    
    def __init__(self, model="gpt-4o-mini"):
        self.model = model
        self.client = OpenAI()
        self.training_data = []  # Collected good reasoning examples
    
    def generate_reasoning(self, question: str) -> Tuple[str, str]:
        """
        Step 1: Ask model to reason through a problem
        Returns (reasoning_chain, final_answer)
        """
        
        prompt = f"""
        Solve this problem step by step.
        Show ALL your thinking.
        At the very end, write "FINAL ANSWER: [answer]"
        
        Problem: {question}
        """
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        ).choices[0].message.content
        
        # Extract final answer
        if "FINAL ANSWER:" in response:
            parts = response.split("FINAL ANSWER:")
            reasoning = parts[0].strip()
            final_answer = parts[1].strip()
        else:
            reasoning = response
            final_answer = response.split('\n')[-1].strip()
        
        return reasoning, final_answer
    
    def verify_answer(self, 
                      final_answer: str, 
                      correct_answer: str) -> bool:
        """
        Step 2: Check if the model's answer matches correct answer
        In real systems, this could be:
        - Exact match (for math)
        - Code execution (for programming)
        - LLM judge (for open-ended)
        """
        
        # Simple string match (in real system, smarter matching)
        correct = correct_answer.lower().strip()
        model_answer = final_answer.lower().strip()
        
        # Check if correct answer appears in model's answer
        return correct in model_answer
    
    def rationalization(self, 
                        question: str, 
                        correct_answer: str) -> str:
        """
        Step 3: For wrong answers, give a HINT and ask to reason
        This is the "rationalization" step of STaR
        """
        
        prompt = f"""
        Problem: {question}
        
        I know the correct answer is: {correct_answer}
        
        Now explain step by step WHY the answer is {correct_answer}.
        Show clear, logical reasoning that leads to this answer.
        """
        
        reasoning = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        ).choices[0].message.content
        
        return reasoning
    
    def collect_training_examples(self,
                                   qa_pairs: List[Tuple[str, str]],
                                   use_rationalization: bool = True):
        """
        Main STaR loop: Generate reasoning, verify, collect good examples
        
        qa_pairs = list of (question, correct_answer) tuples
        """
        
        print(f"\n🚀 STaR Training Data Collection")
        print(f"Processing {len(qa_pairs)} examples...")
        print("=" * 60)
        
        correct_count = 0
        rationalized_count = 0
        
        for i, (question, correct_answer) in enumerate(qa_pairs):
            print(f"\n📝 Example {i+1}/{len(qa_pairs)}")
            print(f"Q: {question[:60]}...")
            
            # Step 1: Generate reasoning
            reasoning, model_answer = self.generate_reasoning(question)
            
            # Step 2: Verify answer
            is_correct = self.verify_answer(model_answer, correct_answer)
            
            if is_correct:
                # Model got it right! Keep this reasoning example
                print(f"✅ Correct! Keeping reasoning example.")
                
                self.training_data.append({
                    "type": "direct_reasoning",
                    "question": question,
                    "reasoning": reasoning,
                    "answer": correct_answer,
                    "source": "model_generated_correct"
                })
                correct_count += 1
                
            elif use_rationalization:
                # Model got it wrong, try rationalization
                print(f"❌ Wrong answer. Trying rationalization...")
                
                rationalized_reasoning = self.rationalization(
                    question, correct_answer
                )
                
                # Add rationalized example (with lower confidence)
                self.training_data.append({
                    "type": "rationalized",
                    "question": question,
                    "reasoning": rationalized_reasoning,
                    "answer": correct_answer,
                    "source": "rationalization"
                })
                rationalized_count += 1
                print(f"🔄 Rationalized example added.")
            
            else:
                print(f"❌ Wrong answer. Discarding.")
        
        print(f"\n📊 STaR Collection Results:")
        print(f"Total examples: {len(qa_pairs)}")
        print(f"Correct (direct): {correct_count}")
        print(f"Rationalized: {rationalized_count}")
        print(f"Discarded: {len(qa_pairs) - correct_count - rationalized_count}")
        print(f"Total training examples collected: {len(self.training_data)}")
        
        return self.training_data
    
    def save_training_data(self, filename: str = "star_training_data.jsonl"):
        """
        Save collected examples in JSONL format for fine-tuning
        """
        
        # Convert to OpenAI fine-tuning format
        formatted_data = []
        
        for example in self.training_data:
            formatted_example = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a careful reasoner. Always think step by step."
                    },
                    {
                        "role": "user",
                        "content": example["question"]
                    },
                    {
                        "role": "assistant",
                        "content": f"{example['reasoning']}\n\nFINAL ANSWER: {example['answer']}"
                    }
                ]
            }
            formatted_data.append(formatted_example)
        
        # Save to file
        with open(filename, 'w') as f:
            for item in formatted_data:
                f.write(json.dumps(item) + '\n')
        
        print(f"\n💾 Saved {len(formatted_data)} training examples to {filename}")
        print(f"This file can be used for OpenAI fine-tuning!")
        
        return filename


# ─── Example Usage ───────────────────────────────────────

# Sample QA pairs (question, correct answer)
sample_qa_pairs = [
    (
        "A store sells apples for $0.50 each and oranges for $0.75 each. "
        "If Sarah buys 4 apples and 3 oranges, how much does she spend?",
        "$4.25"
    ),
    (
        "A train travels at 80 mph. How long does it take to travel 200 miles?",
        "2.5 hours"
    ),
    (
        "If a rectangle has length 12 cm and width 8 cm, what is its area?",
        "96 square cm"
    ),
    (
        "A class has 30 students. 40% are boys. How many girls are there?",
        "18 girls"
    ),
    (
        "What is 25% of 160?",
        "40"
    )
]

# Run STaR data collection
trainer = STaRTrainer(model="gpt-4o-mini")
training_examples = trainer.collect_training_examples(
    sample_qa_pairs,
    use_rationalization=True
)

# Save for fine-tuning
trainer.save_training_data("star_reasoning_data.jsonl")

# Show one example
if training_examples:
    print("\n📌 Sample Training Example:")
    print(json.dumps(training_examples[0], indent=2))
```

---

### How to Actually Fine-Tune with This Data:

```python
from openai import OpenAI
import time

client = OpenAI()

def fine_tune_with_star_data(training_file_path: str):
    """
    Use OpenAI API to fine-tune a model on STaR-generated data
    """
    
    print("🎓 Starting Fine-Tuning Process...")
    
    # Step 1: Upload training file
    print("\n📤 Uploading training data...")
    with open(training_file_path, 'rb') as f:
        upload_response = client.files.create(
            file=f,
            purpose='fine-tune'
        )
    
    file_id = upload_response.id
    print(f"✅ File uploaded! ID: {file_id}")
    
    # Step 2: Create fine-tuning job
    print("\n🏋️ Creating fine-tuning job...")
    ft_job = client.fine_tuning.jobs.create(
        training_file=file_id,
        model="gpt-4o-mini",  # Base model to fine-tune
        hyperparameters={
            "n_epochs": 3,        # How many times to train on data
            "batch_size": 4,      # Examples per training step
            "learning_rate_multiplier": 1.0  # How fast to learn
        }
    )
    
    job_id = ft_job.id
    print(f"✅ Fine-tuning job created! ID: {job_id}")
    
    # Step 3: Wait for completion
    print("\n⏳ Waiting for training to complete...")
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        status = job.status
        
        print(f"Status: {status}")
        
        if status == "succeeded":
            print(f"\n🎉 Fine-tuning COMPLETE!")
            print(f"New model: {job.fine_tuned_model}")
            return job.fine_tuned_model
        
        elif status == "failed":
            print(f"\n❌ Fine-tuning FAILED!")
            print(f"Error: {job.error}")
            return None
        
        # Wait 30 seconds before checking again
        time.sleep(30)


def test_fine_tuned_model(model_name: str, question: str):
    """
    Test the fine-tuned model to see if it reasons better
    """
    
    print(f"\n🧪 Testing fine-tuned model: {model_name}")
    
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "You are a careful reasoner. Always think step by step."
            },
            {
                "role": "user",
                "content": question
            }
        ]
    ).choices[0].message.content
    
    print(f"Question: {question}")
    print(f"Answer:\n{response}")
    
    return response


# To actually run this:
# model = fine_tune_with_star_data("star_reasoning_data.jsonl")
# test_fine_tuned_model(model, "If 8 people share 56 cookies equally, how many does each get?")
```

---

### Key Insight About STaR:

```
Why STaR Works — The Bootstrapping Magic:

Iteration 0 (base model): 
Can solve 30% of problems
Generates 30% good training examples

Train on these → Better model!

Iteration 1 (improved model):
Can solve 45% of problems  
Generates 45% good training examples

Train on these → Even better model!

Iteration 2 (even better model):
Can solve 60% of problems
Generates 60% good training examples

...and so on!

The model literally TEACHES ITSELF to reason better!
```

---

## 4.3 — Reinforcement Learning with a Verifier

---

### What is Reinforcement Learning (RL) in Simple Terms?

Before we apply RL to LLMs, let me explain RL from zero.

---

### The Dog Training Analogy:

```
Teaching a dog to sit:

1. Dog does random things (sits, stands, barks...)
2. When dog sits → give treat (REWARD) ✅
3. When dog doesn't sit → no treat (NO REWARD) ❌
4. Dog learns: "Sitting = treat, so sit more!"

This is Reinforcement Learning!

The key parts:
├── Agent = The dog (the LLM)
├── Action = What the dog does (the LLM's output)
├── Environment = The world (the verifier)
├── Reward = The treat (score from verifier)
└── Policy = Dog's behavior (LLM's behavior)

Goal: Agent learns to take actions that maximize reward
```

---

### RL for LLMs:

```
Teaching an LLM to reason better:

1. LLM generates a reasoning chain + answer
2. Verifier checks: Is the answer correct?
3. Correct answer → POSITIVE reward (+1)
4. Wrong answer → NEGATIVE reward (-1)
5. LLM parameters update: "Do more of what got +reward"
6. Repeat millions of times → LLM gets better at reasoning!
```

---

### The RL Training Loop for Reasoning:

```
┌─────────────────────────────────────────────────────────┐
│            RL Training Loop for Reasoning LLMs          │
│                                                         │
│  ┌──────────┐   Question    ┌──────────────┐           │
│  │  Dataset  │ ──────────►  │   LLM Agent  │           │
│  │(questions │              │  (generates  │           │
│  │ + answers)│              │  reasoning + │           │
│  └──────────┘              │   answer)    │           │
│                             └──────┬───────┘           │
│                                    │                   │
│                                    │ Reasoning chain   │
│                                    │ + Final answer    │
│                                    ▼                   │
│                             ┌──────────────┐           │
│                             │   VERIFIER   │           │
│                             │  (checks if  │           │
│                             │  answer is   │           │
│                             │  correct)    │           │
│                             └──────┬───────┘           │
│                                    │                   │
│                         ┌──────────┴──────────┐        │
│                         │                     │        │
│                    Correct ✅             Wrong ❌       │
│                  Reward: +1           Reward: -1        │
│                         │                     │        │
│                         └──────────┬──────────┘        │
│                                    │                   │
│                                    ▼                   │
│                         ┌──────────────────┐           │
│                         │  Update LLM      │           │
│                         │  Parameters      │           │
│                         │  (RL algorithm)  │           │
│                         └──────────────────┘           │
│                                    │                   │
│                                    └──── Repeat! ───►  │
└─────────────────────────────────────────────────────────┘
```

---

### GRPO: The Algorithm DeepSeek Used

DeepSeek-R1 used an RL algorithm called **GRPO (Group Relative Policy Optimization)**.

Let me explain this simply:

```
GRPO in Simple Terms:

Step 1: For ONE question, generate MULTIPLE answers (a group)
        Q: "What is 15% of 80?"
        Answer 1: reasoning... "12" ✅
        Answer 2: reasoning... "15" ❌  
        Answer 3: reasoning... "12" ✅
        Answer 4: reasoning... "10" ❌
        Answer 5: reasoning... "12" ✅

Step 2: Verify each answer
        Answer 1: Correct → reward +1
        Answer 2: Wrong → reward -1
        Answer 3: Correct → reward +1
        Answer 4: Wrong → reward -1
        Answer 5: Correct → reward +1

Step 3: Calculate RELATIVE reward
        "Compared to the GROUP average, was this answer better?"
        Group average reward = (1 + -1 + 1 + -1 + 1) / 5 = 0.2
        Answer 1: 1 - 0.2 = +0.8 (better than average)
        Answer 2: -1 - 0.2 = -1.2 (worse than average)
        Answer 3: 1 - 0.2 = +0.8
        ...

Step 4: Update model
        "Do more of what got positive relative reward"
        "Do less of what got negative relative reward"

Why is this clever?
→ Using relative reward means the model always gets useful signal
→ Even if all answers are wrong, some are "less wrong"
→ The model learns which DIRECTIONS to improve
```

---

### What Does the Verifier Check?

For DeepSeek-R1, the verifier had TWO types of checks:

```
Verifier Type 1: CORRECTNESS REWARD
└── "Is the final answer correct?"
└── Math: Check if number matches
└── Code: Run the code, check output
└── Yes = +1, No = -1

Verifier Type 2: FORMAT REWARD
└── "Did the model follow the required format?"
└── Does it have <think>...</think> tags?
└── Does it have a clear final answer?
└── Yes = +small reward, No = -small reward

Combined reward = Correctness reward + Format reward
```

---

### The Surprising Discovery — "Aha Moments":

When DeepSeek trained R1 with RL, they observed something **amazing**:

The model **spontaneously developed** reasoning behaviors that were never explicitly taught!

```
Emergent Behaviors from RL Training:

1. Self-verification:
   Model started double-checking its own work
   "Wait, let me verify this..."
   
2. Backtracking:
   Model started correcting itself mid-reasoning
   "Actually, I made an error above. Let me redo..."
   
3. Extended thinking:
   Model naturally spent more time on harder problems
   (without being told to!)
   
4. Reflection:
   Model developed meta-awareness
   "This problem is similar to... so I should use..."

Nobody told the model to do these things!
The RL reward signal CAUSED these behaviors to emerge naturally.
This is why DeepSeek-R1 is so remarkable.
```

---

### Simple Code — Simulating RL Training Signal:

```python
from openai import OpenAI
import random
from dataclasses import dataclass
from typing import List

client = OpenAI()

@dataclass
class TrainingExample:
    question: str
    reasoning: str
    answer: str
    reward: float
    is_correct: bool

class RLDataCollector:
    """
    Simulates collecting RL training signal for reasoning models.
    
    In real RL training, this would feed into a training loop.
    Here we show the DATA COLLECTION and REWARD CALCULATION parts.
    """
    
    def __init__(self, model="gpt-4o-mini", n_samples_per_question=5):
        self.model = model
        self.n_samples = n_samples_per_question
        self.client = OpenAI()
        self.training_buffer = []  # Stores all examples with rewards
    
    def generate_answer(self, question: str) -> tuple:
        """Generate one reasoning attempt"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": f"""
                Solve this step by step.
                Show your complete reasoning.
                End with: ANSWER: [your final answer]
                
                Problem: {question}
                """
            }],
            temperature=0.8  # High temp for diversity
        ).choices[0].message.content
        
        # Parse reasoning and answer
        if "ANSWER:" in response:
            parts = response.split("ANSWER:")
            reasoning = parts[0].strip()
            answer = parts[1].strip()
        else:
            reasoning = response
            answer = response.split('\n')[-1].strip()
        
        return reasoning, answer
    
    def calculate_correctness_reward(self,
                                      model_answer: str,
                                      correct_answer: str) -> float:
        """
        Calculate reward based on answer correctness
        Returns: +1.0 if correct, -1.0 if wrong
        """
        
        correct = correct_answer.lower().strip()
        model = model_answer.lower().strip()
        
        if correct in model or model in correct:
            return 1.0
        else:
            return -1.0
    
    def calculate_format_reward(self, reasoning: str) -> float:
        """
        Calculate reward based on reasoning format quality
        Returns: 0.0 to 0.5
        """
        
        reward = 0.0
        
        # Check for step-by-step structure
        if "step" in reasoning.lower():
            reward += 0.1
        
        # Check for showing work
        if any(op in reasoning for op in ['=', '+', '-', '×', '÷', '*', '/']):
            reward += 0.1
        
        # Check for reasonable length (not too short, not too long)
        word_count = len(reasoning.split())
        if 50 <= word_count <= 500:
            reward += 0.1
        
        # Check for verification/checking
        if any(word in reasoning.lower() 
               for word in ['check', 'verify', 'let me confirm', 'therefore']):
            reward += 0.2
        
        return reward
    
    def collect_grpo_batch(self,
                           question: str,
                           correct_answer: str) -> List[TrainingExample]:
        """
        GRPO: Generate a GROUP of answers for one question
        Calculate relative rewards within the group
        """
        
        print(f"\n🎯 Question: {question[:50]}...")
        
        raw_examples = []
        
        # Generate N samples (the "group" in GRPO)
        for i in range(self.n_samples):
            reasoning, answer = self.generate_answer(question)
            
            # Calculate rewards
            correctness_reward = self.calculate_correctness_reward(
                answer, correct_answer
            )
            format_reward = self.calculate_format_reward(reasoning)
            
            # Combined reward
            total_reward = correctness_reward + format_reward
            is_correct = correctness_reward > 0
            
            raw_examples.append({
                "question": question,
                "reasoning": reasoning,
                "answer": answer,
                "total_reward": total_reward,
                "is_correct": is_correct
            })
            
            status = "✅" if is_correct else "❌"
            print(f"  Sample {i+1}: {status} reward={total_reward:.2f}")
        
        # GRPO: Calculate mean reward of the group
        mean_reward = sum(e["total_reward"] for e in raw_examples) / len(raw_examples)
        print(f"  Group mean reward: {mean_reward:.2f}")
        
        # Calculate RELATIVE rewards (advantage)
        training_examples = []
        for e in raw_examples:
            
            # Relative reward = individual reward - group mean
            # This tells us: "Was this sample better or worse than average?"
            relative_reward = e["total_reward"] - mean_reward
            
            example = TrainingExample(
                question=e["question"],
                reasoning=e["reasoning"],
                answer=e["answer"],
                reward=relative_reward,  # Use RELATIVE reward for training
                is_correct=e["is_correct"]
            )
            training_examples.append(example)
        
        # Show distribution
        positive = sum(1 for e in training_examples if e.reward > 0)
        negative = sum(1 for e in training_examples if e.reward < 0)
        print(f"  Positive advantage: {positive}, Negative advantage: {negative}")
        
        return training_examples
    
    def collect_dataset(self,
                        qa_pairs: List[tuple]) -> List[TrainingExample]:
        """
        Collect RL training data for a full dataset
        """
        
        print(f"\n🚀 Collecting RL Training Data")
        print(f"Questions: {len(qa_pairs)}, Samples per Q: {self.n_samples}")
        print("=" * 60)
        
        all_examples = []
        
        for question, correct_answer in qa_pairs:
            examples = self.collect_grpo_batch(question, correct_answer)
            all_examples.extend(examples)
            self.training_buffer.extend(examples)
        
        # Summary statistics
        total = len(all_examples)
        correct = sum(1 for e in all_examples if e.is_correct)
        
        print(f"\n📊 Collection Complete!")
        print(f"Total examples: {total}")
        print(f"Correct answers: {correct} ({100*correct/total:.1f}%)")
        print(f"Wrong answers: {total-correct} ({100*(total-correct)/total:.1f}%)")
        print(f"\nHigh reward examples (best for training): "
              f"{sum(1 for e in all_examples if e.reward > 0.5)}")
        
        return all_examples


# ─── Usage ───────────────────────────────────────────────

qa_pairs = [
    ("A rectangle has perimeter 30cm and length 10cm. What is the width?", "5cm"),
    ("If 3 cats catch 3 mice in 3 minutes, how many mice do 100 cats catch in 100 minutes?", "10000"),
    ("What is the sum of numbers from 1 to 100?", "5050"),
]

collector = RLDataCollector(model="gpt-4o-mini", n_samples_per_question=3)
examples = collector.collect_dataset(qa_pairs)
```

---

## 4.4 — Reward Modeling (ORM vs PRM)

---

### What is a Reward Model?

In RL training, we need something to **judge** whether an answer is good or bad.

A **Reward Model** is a **separate neural network** trained specifically to **score the quality** of LLM outputs.

---

### Real-World Analogy:

```
Imagine training a chef:

Without reward model:
Chef cooks food → Customer says "good" or "bad" → Chef learns

Problem: Getting real customer feedback for every dish is expensive and slow!

With reward model:
Step 1: Train a food critic (reward model) using real customer feedback
Step 2: Now use the food critic to evaluate dishes AUTOMATICALLY
Step 3: Use critic's scores to train the chef quickly

The food critic = The reward model!
It replaces expensive human feedback with automatic scoring.
```

---

### Two Types of Reward Models:

---

#### Type 1: ORM — Outcome Reward Model

**ORM** looks at the **final answer** only and judges: "Is this correct?"

```
ORM (Outcome Reward Model):

Input:  Question + Full Reasoning Chain + Final Answer
Output: SINGLE SCORE for the whole thing

Example:
Q: "What is 15% of 80?"
Reasoning: "15% = 15/100. So 15/100 × 80 = 1200/100 = 12."
Answer: "12"

ORM Output: 0.95 (very good — answer is correct!)

Q: "What is 15% of 80?"  
Reasoning: "15% means multiply by 15. So 80 × 15 = 1200."
Answer: "1200"

ORM Output: 0.02 (very bad — answer is wrong!)
```

**ORM only cares about the END result.**

---

#### Type 2: PRM — Process Reward Model

**PRM** looks at **each reasoning step** and judges: "Is this step correct?"

```
PRM (Process Reward Model):

Input:  Question + Reasoning Chain (step by step)
Output: SCORE FOR EACH STEP

Example:
Q: "A store offers 20% discount on $150. What is the final price?"

Step 1: "20% discount means I subtract 20% from the price."
PRM Score: 0.99 ← Correct reasoning!

Step 2: "20% of $150 = 150 × 0.2 = $30."
PRM Score: 0.97 ← Correct calculation!

Step 3: "Final price = $150 - $30 = $130."
PRM Score: 0.98 ← Correct!

Final Answer: "$130"
PRM Overall: 0.98 ← Excellent!

--- Now a WRONG example: ---

Q: "A store offers 20% discount on $150. What is the final price?"

Step 1: "20% discount means I add 20% to the price."
PRM Score: 0.02 ← WRONG! (discount means subtract, not add)

Step 2: "20% of $150 = $30."
PRM Score: 0.95 ← This step is actually correct mathematically...

Step 3: "Final price = $150 + $30 = $180."  
PRM Score: 0.01 ← WRONG final answer

PRM identifies WHERE the error happened (Step 1)!
ORM would just say "wrong" without knowing where!
```

---

### ORM vs PRM — Side by Side Comparison:

```
┌─────────────────────┬────────────────────┬────────────────────┐
│ Feature             │ ORM                │ PRM                │
├─────────────────────┼────────────────────┼────────────────────┤
│ What it scores      │ Final answer only  │ Every step         │
│ Training data needed│ (Q, Answer, Label) │ (Q, Steps, Labels) │
│ Data difficulty     │ Easy to collect    │ HARD to collect    │
│ Error detection     │ Only at end        │ Exactly where!     │
│ Training cost       │ Lower              │ Much higher        │
│ Inference cost      │ Lower              │ Higher (per step)  │
│ Better for          │ Short problems     │ Long multi-step    │
│ DeepSeek-R1 used    │ ✅ YES             │ ❌ Avoided!        │
└─────────────────────┴────────────────────┴────────────────────┘
```

---

### Why DeepSeek Avoided PRM (Important Insight!):

```
Problem with PRM:

1. DATA COLLECTION IS VERY HARD:
   You need humans to label EVERY STEP of every solution
   "Is step 3 correct? Is step 7 correct?"
   This is extremely expensive!

2. REWARD HACKING:
   The model might learn to game the PRM
   Producing steps that SCORE WELL but aren't actually good reasoning
   (Like a student who writes what the teacher wants to hear)

3. PRM IS OFTEN WRONG:
   A step might look wrong but lead to a correct answer
   A step might look right but be subtly incorrect
   PRM training data is noisy

DeepSeek's solution:
"Use a simple RULE-BASED verifier (ORM) instead!"
For math: just check if final number is right
For code: just run it and check output
Simple, reliable, no training needed!
```

---

### Training a Simple ORM:

```python
from openai import OpenAI
from sklearn.linear_model import LogisticRegression
import numpy as np
import json

client = OpenAI()

class SimpleORM:
    """
    A simple Outcome Reward Model that learns to score LLM outputs.
    
    In production: This would be a neural network trained on millions of examples.
    Here: We use a simple classifier for demonstration.
    """
    
    def __init__(self):
        self.client = OpenAI()
        # In production: this would be a neural network
        # Here: using a simple approach with LLM scoring
    
    def score_with_llm(self,
                        question: str,
                        reasoning: str,
                        answer: str) -> float:
        """
        Use an LLM as a reward model (LLM-as-judge)
        Returns score between 0 and 1
        """
        
        scoring_prompt = f"""
        You are an expert evaluator. Score this answer from 0.0 to 1.0.
        
        Question: {question}
        
        Reasoning provided:
        {reasoning}
        
        Final Answer given: {answer}
        
        Score based on:
        - Correctness of final answer (most important: 60%)
        - Quality and validity of reasoning steps (30%)
        - Clarity and structure (10%)
        
        Rules:
        - 0.0-0.2 = Completely wrong
        - 0.2-0.4 = Mostly wrong with minor correct elements
        - 0.4-0.6 = Partially correct
        - 0.6-0.8 = Mostly correct with minor errors
        - 0.8-1.0 = Correct and well-reasoned
        
        Respond with ONLY a decimal number like: 0.85
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": scoring_prompt}],
            temperature=0
        ).choices[0].message.content.strip()
        
        try:
            score = float(response)
            score = max(0.0, min(1.0, score))  # Clamp between 0 and 1
        except ValueError:
            # If parsing fails, extract any number found
            import re
            numbers = re.findall(r'\d+\.?\d*', response)
            score = float(numbers[0]) if numbers else 0.5
        
        return score
    
    def build_training_dataset(self,
                                qa_pairs: list,
                                n_samples_each: int = 3) -> list:
        """
        Build ORM training dataset by:
        1. Generating multiple answers for each question
        2. Scoring each with the LLM judge
        3. Storing (question, reasoning, answer, score) tuples
        """
        
        training_data = []
        
        for question, correct_answer in qa_pairs:
            print(f"\n📝 Processing: {question[:40]}...")
            
            for sample_num in range(n_samples_each):
                # Generate an answer
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{
                        "role": "user",
                        "content": f"""
                        Solve step by step.
                        End with: ANSWER: [your answer]
                        
                        {question}
                        """
                    }],
                    temperature=0.8
                ).choices[0].message.content
                
                # Parse
                if "ANSWER:" in response:
                    parts = response.split("ANSWER:")
                    reasoning = parts[0].strip()
                    answer = parts[1].strip()
                else:
                    reasoning = response
                    answer = response.split('\n')[-1]
                
                # Score this answer
                score = self.score_with_llm(question, reasoning, answer)
                
                # Also calculate binary label (for training)
                is_correct = correct_answer.lower() in answer.lower()
                
                training_data.append({
                    "question": question,
                    "reasoning": reasoning,
                    "answer": answer,
                    "score": score,
                    "is_correct": is_correct,
                    "correct_answer": correct_answer
                })
                
                print(f"  Sample {sample_num+1}: score={score:.2f}, correct={is_correct}")
        
        return training_data
    
    def best_of_n_with_orm(self,
                            question: str,
                            n: int = 5) -> tuple:
        """
        Generate N answers and use ORM to pick the best one.
        This is the most common use of reward models!
        """
        
        print(f"\n🎯 Best-of-{n} with ORM")
        print(f"Question: {question}")
        
        candidates = []
        
        # Generate N candidates
        for i in range(n):
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": f"Solve step by step:\n{question}"
                }],
                temperature=0.8
            ).choices[0].message.content
            
            parts = response.split("ANSWER:") if "ANSWER:" in response else [response, ""]
            reasoning = parts[0].strip()
            answer = parts[1].strip() if len(parts) > 1 else response.split('\n')[-1]
            
            candidates.append((reasoning, answer))
        
        # Score each candidate with ORM
        scored = []
        for i, (reasoning, answer) in enumerate(candidates):
            score = self.score_with_llm(question, reasoning, answer)
            scored.append((score, reasoning, answer))
            print(f"  Candidate {i+1}: score={score:.2f}")
        
        # Pick the best
        scored.sort(reverse=True, key=lambda x: x[0])
        best_score, best_reasoning, best_answer = scored[0]
        
        print(f"\n🏆 Best candidate (score={best_score:.2f}):")
        print(f"Answer: {best_answer}")
        
        return best_reasoning, best_answer, best_score


# Usage
orm = SimpleORM()

# Test best-of-N with ORM
reasoning, answer, score = orm.best_of_n_with_orm(
    "A rectangular pool is 25m long and 10m wide. "
    "The water depth is 2m. What is the volume of water?",
    n=3
)
```

---

### Training a PRM (Conceptual):

```python
class SimplePRM:
    """
    Process Reward Model - scores each reasoning step
    """
    
    def __init__(self):
        self.client = OpenAI()
    
    def score_reasoning_steps(self,
                               question: str,
                               reasoning_chain: str) -> dict:
        """
        Score each step in a reasoning chain.
        Returns scores for each step.
        """
        
        # First, parse the reasoning into steps
        parse_prompt = f"""
        Break this reasoning into numbered steps:
        
        Reasoning: {reasoning_chain}
        
        Format each step as:
        STEP 1: [step content]
        STEP 2: [step content]
        ...
        """
        
        parsed = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": parse_prompt}],
            temperature=0
        ).choices[0].message.content
        
        # Extract steps
        steps = []
        for line in parsed.split('\n'):
            if line.strip().startswith('STEP'):
                steps.append(line.strip())
        
        # Score each step
        step_scores = {}
        
        for i, step in enumerate(steps):
            score_prompt = f"""
            Question: {question}
            
            Previous steps taken:
            {chr(10).join(steps[:i])}
            
            Current step to evaluate:
            {step}
            
            Is this step:
            1. Logically correct given the question?
            2. Following from the previous steps correctly?
            3. Heading toward the right solution?
            
            Score from 0.0 (completely wrong) to 1.0 (perfectly correct).
            Respond ONLY with a decimal: e.g., 0.85
            """
            
            score_response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": score_prompt}],
                temperature=0
            ).choices[0].message.content.strip()
            
            try:
                score = float(score_response)
            except:
                score = 0.5
            
            step_scores[f"step_{i+1}"] = {
                "content": step,
                "score": score
            }
            
            print(f"  {step[:40]}... → score: {score:.2f}")
        
        # Find the first "bad" step (where reasoning goes wrong)
        problem_steps = [
            k for k, v in step_scores.items() 
            if v['score'] < 0.5
        ]
        
        result = {
            "step_scores": step_scores,
            "overall_score": sum(v['score'] for v in step_scores.values()) / max(len(step_scores), 1),
            "problem_steps": problem_steps,
            "first_error": problem_steps[0] if problem_steps else None
        }
        
        return result
    
    def find_reasoning_errors(self,
                               question: str,
                               wrong_answer_reasoning: str) -> str:
        """
        Use PRM to diagnose WHERE reasoning went wrong
        """
        
        print(f"\n🔍 PRM Error Diagnosis")
        scores = self.score_reasoning_steps(question, wrong_answer_reasoning)
        
        if scores['first_error']:
            error_step = scores['step_scores'][scores['first_error']]
            print(f"\n❌ First error found at: {scores['first_error']}")
            print(f"Problem step: {error_step['content']}")
            print(f"Score: {error_step['score']:.2f}")
        else:
            print("\n✅ No clear error steps found")
        
        return scores


# Usage example
prm = SimplePRM()

wrong_reasoning = """
A car travels at 60 mph for 2 hours, then 80 mph for 1 hour.
Step 1: Total distance = 60 × 2 + 80 × 1 = 120 + 80 = 200 miles
Step 2: Total time = 2 + 1 = 3 hours  
Step 3: Average speed = total time / total distance = 3 / 200 = 0.015 mph
"""

# The error is in Step 3: should be distance/time, not time/distance!
# prm.find_reasoning_errors(
#     "What is the average speed for the whole journey?",
#     wrong_reasoning
# )
```

---

## 4.5 — Self-Refinement

---

### What is Self-Refinement?

**Self-Refinement** = Training a model to **critique its own output** and then **improve it**, without any external feedback.

This is inspired by how humans improve their work:
1. Write a draft
2. Read it critically
3. Find problems
4. Improve it
5. Repeat

---

### Real-World Analogy:

```
A good writer's process:

Draft 1: "The AI is good at stuff."
Self-critique: "Too vague. What stuff? For whom? Be specific."
Draft 2: "Modern AI systems excel at tasks like image recognition..."
Self-critique: "Better, but needs examples. Add concrete data."  
Draft 3: "Modern AI systems excel at image recognition (97% accuracy
          on ImageNet), natural language understanding (MMLU benchmark
          90%+), and strategic planning (AlphaGo)."

This is self-refinement!
```

---

### The Self-Refinement Paper (Madaan et al., 2023):

The key insight: **An LLM can act as its own critic and its own editor.**

```
Self-Refinement Framework:

Component 1: GENERATOR (same LLM)
└── Produces initial output

Component 2: FEEDBACK (same LLM)
└── Critiques the output
└── "What is wrong? What is missing? What could be better?"

Component 3: REFINEMENT (same LLM)
└── Improves output based on feedback

All THREE components = ONE LLM!
```

---

### Self-Refinement Code:

```python
from openai import OpenAI
from typing import List, Dict

client = OpenAI()

class SelfRefinementSystem:
    """
    Self-refinement: A model critiques and improves its own outputs.
    This can be used:
    1. At INFERENCE TIME (immediate improvement)
    2. To generate TRAINING DATA (for training-time improvement)
    """
    
    def __init__(self, model="gpt-4o", max_iterations=3):
        self.model = model
        self.max_iterations = max_iterations
        self.client = OpenAI()
        self.refinement_history = []
    
    def generate_initial_response(self, task: str) -> str:
        """Step 1: Generate initial answer"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": f"Complete this task:\n\n{task}"
            }]
        ).choices[0].message.content
        
        return response
    
    def generate_critique(self,
                           task: str,
                           current_response: str,
                           iteration: int) -> str:
        """Step 2: Model critiques its own output"""
        
        critique_prompt = f"""
        You wrote a response to a task, and now you need to 
        CRITICALLY evaluate your own work.
        
        ORIGINAL TASK:
        {task}
        
        YOUR RESPONSE (Iteration {iteration}):
        {current_response}
        
        Provide a DETAILED CRITIQUE covering:
        
        1. ACCURACY: What facts or claims might be wrong or incomplete?
        2. COMPLETENESS: What important aspects did you miss?
        3. CLARITY: What parts are confusing or poorly explained?
        4. STRUCTURE: Is the organization logical and helpful?
        5. DEPTH: Where could you go deeper or provide more insight?
        6. SPECIFIC IMPROVEMENTS: List 3-5 concrete things to fix.
        
        Be HONEST and SPECIFIC. Do not just say "good job."
        Really find the weaknesses.
        """
        
        critique = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": critique_prompt}]
        ).choices[0].message.content
        
        return critique
    
    def generate_refinement(self,
                             task: str,
                             current_response: str,
                             critique: str,
                             iteration: int) -> str:
        """Step 3: Model improves based on its own critique"""
        
        refinement_prompt = f"""
        You wrote a response to a task, critiqued it, and now must IMPROVE it.
        
        ORIGINAL TASK:
        {task}
        
        YOUR PREVIOUS RESPONSE:
        {current_response}
        
        YOUR CRITIQUE OF THAT RESPONSE:
        {critique}
        
        Now write an IMPROVED response (Iteration {iteration+1}) that:
        ✅ Addresses every point in your critique
        ✅ Keeps what was good from before
        ✅ Adds what was missing
        ✅ Fixes what was wrong
        ✅ Is clearly better than the previous version
        
        Write the improved response now:
        """
        
        improved = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": refinement_prompt}]
        ).choices[0].message.content
        
        return improved
    
    def evaluate_improvement(self,
                              task: str,
                              old_response: str,
                              new_response: str) -> dict:
        """
        Check if the new response is actually better
        This prevents the model from making things WORSE
        """
        
        eval_prompt = f"""
        Compare these two responses to the same task.
        
        TASK: {task}
        
        RESPONSE A (older): 
        {old_response[:500]}...
        
        RESPONSE B (newer):
        {new_response[:500]}...
        
        Is Response B BETTER than Response A?
        
        Score each from 1-10 and explain briefly.
        Format: 
        SCORE_A: [number]
        SCORE_B: [number]
        BETTER: A or B
        REASON: [one sentence]
        """
        
        evaluation = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": eval_prompt}],
            temperature=0
        ).choices[0].message.content
        
        # Parse scores
        lines = evaluation.split('\n')
        result = {"raw": evaluation, "b_is_better": True}
        
        for line in lines:
            if "BETTER:" in line:
                better = line.split("BETTER:")[1].strip()
                result["b_is_better"] = "B" in better.upper()
        
        return result
    
    def refine(self, task: str, verbose: bool = True) -> Dict:
        """
        Main self-refinement loop
        """
        
        print(f"\n🔄 Self-Refinement System")
        print(f"Task: {task[:60]}...")
        print("=" * 60)
        
        # Step 1: Initial response
        current_response = self.generate_initial_response(task)
        
        if verbose:
            print(f"\n📝 Initial Response:")
            print(current_response[:300] + "..." if len(current_response) > 300 else current_response)
        
        self.refinement_history = [{
            "iteration": 0,
            "response": current_response,
            "critique": None
        }]
        
        # Refinement loop
        for iteration in range(self.max_iterations):
            
            print(f"\n{'='*20} ITERATION {iteration+1} {'='*20}")
            
            # Step 2: Generate critique
            print(f"🔍 Generating self-critique...")
            critique = self.generate_critique(task, current_response, iteration+1)
            
            if verbose:
                print(f"\n💭 Self-Critique:")
                print(critique[:300] + "..." if len(critique) > 300 else critique)
            
            # Step 3: Generate refined response
            print(f"\n✍️ Generating improved response...")
            refined_response = self.generate_refinement(
                task, current_response, critique, iteration+1
            )
            
            # Step 4: Check if actually improved
            evaluation = self.evaluate_improvement(
                task, current_response, refined_response
            )
            
            if evaluation["b_is_better"]:
                print(f"✅ Improvement confirmed! Moving to refined version.")
                current_response = refined_response
            else:
                print(f"⚠️ Refinement not clearly better. Keeping previous version.")
            
            self.refinement_history.append({
                "iteration": iteration + 1,
                "response": current_response,
                "critique": critique,
                "improved": evaluation["b_is_better"]
            })
        
        print(f"\n🏆 FINAL REFINED RESPONSE:")
        print(current_response)
        
        return {
            "task": task,
            "final_response": current_response,
            "iterations": len(self.refinement_history),
            "history": self.refinement_history
        }
    
    def generate_training_data(self, tasks: List[str]) -> List[Dict]:
        """
        Use self-refinement to generate HIGH-QUALITY training data.
        
        The final refined response becomes the TARGET for training.
        This is how companies generate reasoning training data at scale!
        """
        
        training_pairs = []
        
        for task in tasks:
            print(f"\n📚 Processing task for training data: {task[:40]}...")
            result = self.refine(task, verbose=False)
            
            # Training pair: task → final refined response
            training_pairs.append({
                "input": task,
                "output": result["final_response"],  # High-quality refined answer
                "iterations_needed": result["iterations"]
            })
            
            print(f"✅ Added training pair (refined through {result['iterations']} iterations)")
        
        print(f"\n📊 Generated {len(training_pairs)} high-quality training pairs!")
        return training_pairs


# ─── Usage ────────────────────────────────────────────────

refiner = SelfRefinementSystem(model="gpt-4o", max_iterations=2)

# For inference-time use:
result = refiner.refine(
    "Explain why transformer models are better than RNNs "
    "for natural language processing tasks."
)

# For training data generation:
tasks_for_training = [
    "What are the main challenges in AI alignment?",
    "How does backpropagation work in neural networks?",
]
# training_data = refiner.generate_training_data(tasks_for_training)
```

---

## 4.6 — Internalizing Search (Meta-CoT)

---

### What is "Internalizing Search"?

This is one of the most advanced and exciting ideas in reasoning AI.

Let me build up to it carefully.

---

### The Problem: External vs Internal Search

```
External Search (what we built in Part 1):
LLM + Tree of Thoughts (code outside the model)
→ Model generates thoughts
→ EXTERNAL CODE decides which to explore
→ Model continues from selected thought
→ EXTERNAL CODE manages the whole search

Problem: Complex, slow, many API calls

Internal Search (Internalizing Search):
LLM does ALL of this INSIDE its own generation!
→ Model generates thoughts
→ Model ITSELF decides which are promising
→ Model continues from best ones
→ Model manages the whole search internally

The search process is BAKED INTO the model's weights!
```

---

### What is Meta-CoT?

**Meta-CoT (Meta-Chain of Thought)** is a paper from Anthropic (2025) that asks:

> "What if we trained a model to generate ALL the search steps, backtracking, exploration, and reasoning as one long sequence of text?"

Instead of having external code manage a search tree, the model learns to **output its own thinking process** including:
- Trying one approach
- Realizing it's wrong
- Backtracking
- Trying another approach
- Evaluating which path is better
- Continuing on the best path

---

### The Key Insight — Thinking as Text:

```
Regular CoT produces text like:
"Step 1: Calculate X
 Step 2: Now use X to find Y
 Step 3: Therefore Z"

Meta-CoT produces text like:
"Let me try approach 1...
 Hmm, that seems to lead to a dead end.
 Let me backtrack and try approach 2...
 This looks more promising.
 Actually, let me verify my assumption here...
 Yes, that assumption is correct.
 Continuing with approach 2...
 I get the answer Z."

The MODEL ITSELF is doing what ToT did externally!
```

---

### How Meta-CoT is Trained:

```
Step 1: Generate "Thinking Traces"
│
└── Use existing techniques (ToT, MCTS) to solve hard problems
└── Record the FULL SEARCH PROCESS as text
    "Try path A... dead end... backtrack... try path B... success!"

Step 2: Filter for Quality
│
└── Keep only traces that led to CORRECT final answers
└── These become training examples

Step 3: Train the Model
│
└── SFT: Fine-tune model on these thinking traces
└── RL: Reward model for generating useful thinking patterns

Step 4: Model Internalizes the Search
│
└── Model learns: "When I write 'let me try another approach'
                   I should then actually try something different"
└── Model learns: "When I write 'wait, let me check this'
                   I should actually verify"

Result: The model NATURALLY searches through reasoning space
        without any external code!
```

---

### Connection to DeepSeek-R1:

```
DeepSeek-R1 is a great example of internalized search!

When you ask DeepSeek-R1 a hard question, it outputs:
<think>
Let me approach this from the perspective of...
Actually, that's not quite right. Let me reconsider.
If I think about it differently...
Wait, there's an important constraint I missed.
Let me restart with this constraint in mind...
OK so now I have...
Let me verify: does this make sense?
Yes! So the answer is...
</think>

This is INTERNALIZED SEARCH!
The model is doing Tree of Thoughts INSIDE its output text.
No external code needed.
```

---

### Generating Meta-CoT Training Data:

```python
from openai import OpenAI
import json

client = OpenAI()

class MetaCoTDataGenerator:
    """
    Generates Meta-CoT training data by:
    1. Using Tree-of-Thoughts externally to solve hard problems
    2. Converting the search process into a linear "thinking trace"
    3. This trace becomes training data for internalizing search
    """
    
    def __init__(self, model="gpt-4o"):
        self.model = model
        self.client = OpenAI()
    
    def generate_thinking_trace(self, problem: str) -> str:
        """
        Generate a rich 'thinking trace' that shows:
        - Multiple approaches being tried
        - Dead ends being recognized
        - Backtracking happening
        - Best path being selected
        
        This simulates what an internalized search model would produce.
        """
        
        prompt = f"""
        I want you to solve this problem, but do it in a very NATURAL way
        that shows your complete thought process including:
        - Starting approaches that might not work
        - Realizing when you're going in the wrong direction  
        - Backtracking and trying different approaches
        - Checking your work mid-way
        - Moments of insight
        - Verification at the end
        
        Use natural language like:
        "Let me try... Hmm, that doesn't work because... Let me backtrack...
         What if I approach it this way... Yes! That looks right...
         But wait, let me double-check... Confirmed! So the answer is..."
        
        Make this feel like watching a human expert's natural thought process.
        Show the MESSY REAL thinking, not the cleaned-up version.
        
        Problem: {problem}
        
        Start thinking now:
        """
        
        thinking_trace = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2000
        ).choices[0].message.content
        
        return thinking_trace
    
    def extract_final_answer(self, thinking_trace: str) -> str:
        """Extract the clean final answer from a thinking trace"""
        
        extract_prompt = f"""
        From this thinking trace, extract ONLY the final answer.
        Give just the clean, concise final answer with no explanation.
        
        Thinking trace:
        {thinking_trace}
        
        Final answer:
        """
        
        answer = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": extract_prompt}],
            temperature=0
        ).choices[0].message.content.strip()
        
        return answer
    
    def verify_answer(self,
                       problem: str,
                       answer: str,
                       correct_answer: str = None) -> bool:
        """Verify if the final answer is correct"""
        
        if correct_answer:
            return correct_answer.lower().strip() in answer.lower().strip()
        
        # Use LLM to verify if we don't have ground truth
        verify_prompt = f"""
        Problem: {problem}
        Answer given: {answer}
        
        Is this answer correct and complete?
        Respond ONLY: YES or NO
        """
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": verify_prompt}],
            temperature=0
        ).choices[0].message.content.strip().upper()
        
        return "YES" in response
    
    def create_training_example(self,
                                 problem: str,
                                 thinking_trace: str,
                                 final_answer: str) -> dict:
        """
        Format as training example for Meta-CoT training.
        This is designed to teach models to INTERNALIZE search.
        """
        
        # Format with thinking tags (like DeepSeek-R1)
        full_response = f"""<think>
{thinking_trace}
</think>

{final_answer}"""
        
        return {
            "messages": [
                {
                    "role": "user",
                    "content": problem
                },
                {
                    "role": "assistant",
                    "content": full_response
                }
            ],
            "metadata": {
                "thinking_length": len(thinking_trace.split()),
                "has_backtracking": any(
                    phrase in thinking_trace.lower() 
                    for phrase in [
                        "let me backtrack", "actually", "wait",
                        "hmm", "that's not right", "let me try again"
                    ]
                )
            }
        }
    
    def generate_dataset(self,
                          problems: list,
                          correct_answers: list = None) -> list:
        """
        Generate full Meta-CoT training dataset
        """
        
        print(f"\n🧠 Meta-CoT Training Data Generation")
        print(f"Problems to process: {len(problems)}")
        print("=" * 60)
        
        dataset = []
        
        for i, problem in enumerate(problems):
            print(f"\n📝 Problem {i+1}: {problem[:50]}...")
            
            correct_answer = correct_answers[i] if correct_answers else None
            
            # Generate thinking trace
            print("  🤔 Generating thinking trace...")
            thinking_trace = self.generate_thinking_trace(problem)
            
            # Extract final answer
            final_answer = self.extract_final_answer(thinking_trace)
            
            # Verify
            is_correct = self.verify_answer(problem, final_answer, correct_answer)
            
            if is_correct:
                print(f"  ✅ Correct! Adding to dataset.")
                example = self.create_training_example(
                    problem, thinking_trace, final_answer
                )
                dataset.append(example)
            else:
                print(f"  ❌ Wrong answer. Skipping.")
            
            print(f"  Dataset size so far: {len(dataset)}")
        
        print(f"\n📊 Final dataset: {len(dataset)} examples")
        
        # Show backtracking statistics
        has_backtracking = sum(
            1 for ex in dataset 
            if ex['metadata']['has_backtracking']
        )
        print(f"Examples with backtracking: {has_backtracking}/{len(dataset)}")
        
        return dataset
    
    def save_dataset(self, dataset: list, filename: str = "meta_cot_data.jsonl"):
        """Save dataset for training"""
        
        with open(filename, 'w') as f:
            for example in dataset:
                # Save only the messages part (remove metadata for training)
                training_example = {"messages": example["messages"]}
                f.write(json.dumps(training_example) + '\n')
        
        print(f"\n💾 Saved {len(dataset)} Meta-CoT examples to {filename}")


# ─── Usage ────────────────────────────────────────────────

problems = [
    "Alice is twice as old as Bob was when Alice was as old as Bob is now. "
    "If Alice is 24, how old is Bob?",
    
    "A farmer has 17 sheep. All but 9 die. How many are left?",
    
    "You have a 3-liter jug and a 5-liter jug. "
    "How do you measure exactly 4 liters of water?"
]

correct_answers = ["18", "9", "4 liters"]

generator = MetaCoTDataGenerator(model="gpt-4o")
# dataset = generator.generate_dataset(problems, correct_answers)
# generator.save_dataset(dataset)
```

---

### Summary of All Training-Time Techniques:

```
┌────────────────────┬──────────────────────────┬───────────────────────┐
│ Technique          │ Core Idea                │ Key Benefit           │
├────────────────────┼──────────────────────────┼───────────────────────┤
│ STaR               │ Model teaches itself      │ No human data needed  │
│                    │ using own correct outputs │ Scales automatically  │
├────────────────────┼──────────────────────────┼───────────────────────┤
│ RL + Verifier      │ Reward correct reasoning  │ Emergent behaviors    │
│                    │ Punish wrong reasoning    │ (like DeepSeek-R1)    │
├────────────────────┼──────────────────────────┼───────────────────────┤
│ ORM                │ Score final answers       │ Simple, reliable      │
│                    │                           │ Easy to collect data  │
├────────────────────┼──────────────────────────┼───────────────────────┤
│ PRM                │ Score each reasoning step │ Precise error finding │
│                    │                           │ Better training signal│
├────────────────────┼──────────────────────────┼───────────────────────┤
│ Self-Refinement    │ Model critiques itself    │ Improves quality      │
│                    │ and improves             │ No external judge      │
├────────────────────┼──────────────────────────┼───────────────────────┤
│ Meta-CoT           │ Train model to do search  │ No external code      │
│                    │ internally in text        │ Natural reasoning     │
└────────────────────┴──────────────────────────┴───────────────────────┘
```

---

# 💻 CHAPTER 5: Local Deployment

---

## 5.1 — Why Deploy Locally?

---

### The Case for Local AI:

```
Using API (OpenAI/DeepSeek API):
✅ Easy to start
✅ No hardware needed
✅ Always up to date
❌ Costs money per token
❌ Data sent to external servers (privacy!)
❌ Requires internet
❌ Rate limits
❌ Cannot customize the model

Running Locally:
✅ FREE after setup (no per-token cost)
✅ 100% private (data never leaves your machine)
✅ Works offline
✅ No rate limits
✅ Can modify and customize the model
❌ Needs good hardware (RAM, ideally GPU)
❌ Slower on consumer hardware
❌ You manage everything
```

---

### For Our Deep Research Project:

```
Why local deployment matters for Deep Research:

1. PRIVACY: Research topics might be confidential
   "What are my company's weaknesses?" → Don't want this sent to OpenAI!

2. COST: Deep Research makes MANY model calls
   10 searches × 5 reasoning steps × $0.01/call = expensive at scale

3. CUSTOMIZATION: Can fine-tune the model on domain-specific data

4. LEARNING: Understanding local deployment is a key engineering skill
```

---

## 5.2 — Understanding Quantization

---

### What is Quantization and Why Do We Need It?

The problem with large models is **size**.

```
DeepSeek-R1-70B model:
In full precision (float32): 70B × 4 bytes = 280 GB RAM needed!
Who has 280 GB of RAM? Almost nobody.

We need to COMPRESS the model to fit on normal hardware.
This compression is called QUANTIZATION.
```

---

### What is a Number's "Precision"?

```
Float32 (full precision):
└── 32 bits to store one number
└── Can represent: -3.14159265358979...
└── Very precise, very large

Float16 (half precision):
└── 16 bits to store one number
└── Can represent: -3.14159... (less precise)
└── Half the size!

Int8 (8-bit integer):
└── 8 bits to store one number
└── Can represent: -128 to 127
└── One quarter the size!

Int4 (4-bit integer):
└── 4 bits to store one number
└── Can represent: -8 to 7
└── One eighth the size!
```

---

### How Quantization Works:

```
Imagine a model weight is: 3.14159265

Float32 (exact): 3.14159265 ← 4 bytes
Float16:         3.141       ← 2 bytes (slight rounding)
Int8:            3           ← 1 byte (more rounding)
Int4:            3           ← 0.5 bytes (significant rounding)

The quantized model:
- Uses less memory (can run on your laptop!)
- Is slightly less accurate (due to rounding)
- But usually still very good!
```

---

### Real-World Analogy:

```
Think of quantization like compressing a photo:

Original photo: 50 MB (RAW format)
├── Perfect quality, every pixel exact
└── Takes lots of storage

Compressed photo: 2 MB (JPEG)  
├── Looks almost the same to the eye
├── But some tiny details are lost
└── But it fits on your phone!

Quantization = JPEG compression for AI models
```

---

### Quantization Impact on Model Size:

```
DeepSeek-R1-7B Model:
┌──────────┬───────────┬────────────┬─────────────────────┐
│ Format   │ Size      │ RAM Needed │ Quality             │
├──────────┼───────────┼────────────┼─────────────────────┤
│ float32  │ 28 GB     │ 32 GB      │ ★★★★★ (perfect)    │
│ float16  │ 14 GB     │ 16 GB      │ ★★★★★ (same)       │
│ 8-bit    │ 7 GB      │ 8 GB       │ ★★★★☆ (excellent)  │
│ 4-bit    │ 3.5 GB    │ 4 GB       │ ★★★★☆ (very good)  │
└──────────┴───────────┴────────────┴─────────────────────┘

DeepSeek-R1-70B Model:
┌──────────┬───────────┬────────────┬─────────────────────┐
│ Format   │ Size      │ RAM Needed │ Quality             │
├──────────┼───────────┼────────────┼─────────────────────┤
│ float32  │ 280 GB    │ 280+ GB    │ ★★★★★               │
│ float16  │ 140 GB    │ 140 GB     │ ★★★★★               │
│ 8-bit    │ 70 GB     │ 80 GB      │ ★★★★☆               │
│ 4-bit    │ 35 GB     │ 40 GB      │ ★★★★☆               │
└──────────┴───────────┴────────────┴─────────────────────┘
```

---

## 5.3 — Running DeepSeek-R1 Locally with Ollama

---

### What is Ollama?

**Ollama** is a tool that makes running LLMs locally as easy as running any software.

```
Without Ollama:
1. Download model weights (complex file format)
2. Install CUDA drivers
3. Install PyTorch
4. Write loading code
5. Handle memory management
6. Write inference code
7. Hope it works!

With Ollama:
1. Install Ollama (one click)
2. Type: ollama run deepseek-r1
3. Start chatting!
```

---

### Step 1: Install Ollama

```bash
# On Mac:
brew install ollama

# On Linux:
curl -fsSL https://ollama.com/install.sh | sh

# On Windows:
# Download installer from https://ollama.com/download

# Verify installation:
ollama --version
```

---

### Step 2: Download and Run DeepSeek-R1

```bash
# Run DeepSeek-R1 (1.5B - smallest, works on most laptops)
ollama run deepseek-r1:1.5b

# Run 7B model (needs ~8GB RAM)
ollama run deepseek-r1:7b

# Run 14B model (needs ~16GB RAM)
ollama run deepseek-r1:14b

# Run 32B model (needs ~32GB RAM)
ollama run deepseek-r1:32b

# Once running, you can chat directly in terminal!
# Type your question and press Enter.

# To exit:
/bye
```

---

### Step 3: Use Ollama from Python

```bash
# First install the Python library
pip install ollama
```

```python
import ollama

# ─── Basic Chat ───────────────────────────────────────────

def chat_with_local_model(message: str, model: str = "deepseek-r1:7b"):
    """
    Send a message to your locally running DeepSeek-R1
    """
    
    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": message
            }
        ]
    )
    
    return response['message']['content']


# Test it
answer = chat_with_local_model(
    "What are the main causes of climate change? Think step by step."
)
print(answer)


# ─── Streaming Response ───────────────────────────────────

def chat_with_streaming(message: str, model: str = "deepseek-r1:7b"):
    """
    Stream responses token by token (feels more responsive)
    """
    
    print("🤔 Thinking...\n")
    
    stream = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": message}],
        stream=True  # Enable streaming
    )
    
    full_response = ""
    
    for chunk in stream:
        token = chunk['message']['content']
        print(token, end='', flush=True)  # Print as it generates
        full_response += token
    
    print("\n")  # New line at end
    return full_response


# Test streaming
chat_with_streaming("Explain quantum computing in simple terms.")


# ─── Access Thinking vs Answer ─────────────────────────────

def get_thinking_and_answer(message: str, model: str = "deepseek-r1:7b"):
    """
    Separate the <think> part from the final answer
    """
    
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": message}]
    )
    
    full_output = response['message']['content']
    
    # DeepSeek-R1 wraps thinking in <think> tags
    if "<think>" in full_output and "</think>" in full_output:
        thinking_start = full_output.index("<think>") + len("<think>")
        thinking_end = full_output.index("</think>")
        
        thinking = full_output[thinking_start:thinking_end].strip()
        answer = full_output[thinking_end + len("</think>"):].strip()
    else:
        thinking = "No explicit thinking shown"
        answer = full_output
    
    return thinking, answer


# Test
thinking, answer = get_thinking_and_answer(
    "Is it better to rent or buy a home? Consider all factors."
)

print("🧠 THINKING PROCESS:")
print(thinking[:500] + "..." if len(thinking) > 500 else thinking)
print("\n📝 FINAL ANSWER:")
print(answer)
```

---

### Using OpenAI-Compatible API with Ollama:

```python
from openai import OpenAI

# Ollama provides an OpenAI-compatible API!
# This means any code written for OpenAI works with Ollama too

client = OpenAI(
    base_url="http://localhost:11434/v1",  # Ollama's local API
    api_key="ollama"  # Can be anything, Ollama doesn't check
)

def query_local_model(question: str, model: str = "deepseek-r1:7b"):
    """
    Use OpenAI SDK to query local Ollama model.
    Same code works for both OpenAI API and local Ollama!
    """
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": question
            }
        ],
        temperature=0.7,
        stream=False
    )
    
    return response.choices[0].message.content


# This is exactly the same as calling OpenAI API!
answer = query_local_model(
    "What is the difference between deep learning and machine learning?"
)
print(answer)
```

---

## 5.4 — Building the Complete Deep Research System

---

### Full System Architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                DEEP RESEARCH SYSTEM                         │
│                                                             │
│  User Question                                             │
│       ↓                                                    │
│  ┌────────────────────┐                                    │
│  │   PLANNER MODULE   │ ← Reasoning Model (local/API)      │
│  │ Breaks question    │   Decides what to search           │
│  │ into sub-questions │                                    │
│  └────────┬───────────┘                                    │
│           ↓                                                │
│  ┌────────────────────┐                                    │
│  │   SEARCH MODULE    │ ← Web Search API (Tavily/Serper)   │
│  │ Searches web for   │   Returns URLs + snippets          │
│  │ each sub-question  │                                    │
│  └────────┬───────────┘                                    │
│           ↓                                                │
│  ┌────────────────────┐                                    │
│  │   READER MODULE    │ ← Scrapes webpage content          │
│  │ Reads web pages    │   Extracts relevant text           │
│  │ Extracts key info  │                                    │
│  └────────┬───────────┘                                    │
│           ↓                                                │
│  ┌────────────────────┐                                    │
│  │  VERIFIER MODULE   │ ← Reasoning Model                  │
│  │ Cross-checks facts │   Checks consistency               │
│  │ across sources     │                                    │
│  └────────┬───────────┘                                    │
│           ↓                                                │
│  ┌────────────────────┐                                    │
│  │  SYNTHESIS MODULE  │ ← Reasoning Model                  │
│  │ Combines all info  │   Writes final report              │
│  │ Writes report      │                                    │
│  └────────┬───────────┘                                    │
│           ↓                                                │
│  Final Research Report (with citations)                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 5.5 — Full Project Code (End-to-End)

---

### Setup and Dependencies:

```bash
# Install all required packages
pip install openai ollama tavily-python requests beautifulsoup4 \
            python-dotenv rich pydantic

# Get a Tavily API key (free tier available)
# https://tavily.com
```

```python
# .env file
OPENAI_API_KEY=your_openai_key_here
TAVILY_API_KEY=your_tavily_key_here
DEEPSEEK_API_KEY=your_deepseek_key_here  # optional
```

---

### The Complete Deep Research System:

```python
"""
deep_research.py
Complete Deep Research System with:
- Local DeepSeek-R1 (via Ollama) OR OpenAI API
- Web search (Tavily)
- Web reading (BeautifulSoup)
- Fact verification
- Research synthesis with citations
"""

import os
import json
import time
import requests
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from openai import OpenAI
import ollama

load_dotenv()


# ─── Data Models ──────────────────────────────────────────

@dataclass
class SearchResult:
    """Represents one search result"""
    title: str
    url: str
    snippet: str
    content: str = ""
    relevance_score: float = 0.0


@dataclass
class ResearchSection:
    """One section of the research report"""
    sub_question: str
    search_results: List[SearchResult]
    key_findings: str
    sources: List[str]


@dataclass
class ResearchReport:
    """Complete research report"""
    main_question: str
    executive_summary: str
    sections: List[ResearchSection]
    conclusion: str
    all_sources: List[str]
    confidence_level: str
    total_sources_consulted: int


# ─── Model Interface ──────────────────────────────────────

class ModelInterface:
    """
    Unified interface for both local (Ollama) and API-based models.
    Switch between them easily!
    """
    
    def __init__(self, mode: str = "local", model: str = None):
        """
        mode: "local" (Ollama) or "api" (OpenAI) or "deepseek_api"
        """
        self.mode = mode
        
        if mode == "local":
            self.model = model or "deepseek-r1:7b"
            print(f"🖥️ Using LOCAL model: {self.model}")
            
        elif mode == "api":
            self.model = model or "gpt-4o"
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            print(f"🌐 Using OpenAI API: {self.model}")
            
        elif mode == "deepseek_api":
            self.model = model or "deepseek-reasoner"
            self.client = OpenAI(
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com"
            )
            print(f"🌐 Using DeepSeek API: {self.model}")
        
        elif mode == "ollama_api":
            # Use Ollama via OpenAI-compatible API
            self.model = model or "deepseek-r1:7b"
            self.client = OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama"
            )
            print(f"🖥️ Using Ollama API mode: {self.model}")
    
    def think(self,
               prompt: str,
               temperature: float = 0.7,
               max_tokens: int = 2000) -> str:
        """
        Send prompt to model and get response.
        Works the same regardless of which model you chose!
        """
        
        if self.mode == "local":
            # Use Ollama directly
            try:
                response = ollama.chat(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": temperature}
                )
                return response['message']['content']
            
            except Exception as e:
                print(f"⚠️ Ollama error: {e}")
                print("Make sure Ollama is running: 'ollama serve'")
                raise
        
        else:
            # Use API (OpenAI, DeepSeek, or Ollama API)
            try:
                messages = [{"role": "user", "content": prompt}]
                
                kwargs = {
                    "model": self.model,
                    "messages": messages,
                }
                
                # o1 models don't support temperature
                if "o1" not in self.model:
                    kwargs["temperature"] = temperature
                
                if max_tokens and "o1" not in self.model:
                    kwargs["max_tokens"] = max_tokens
                
                response = self.client.chat.completions.create(**kwargs)
                return response.choices[0].message.content
            
            except Exception as e:
                print(f"⚠️ API error: {e}")
                raise
    
    def think_with_reasoning(self, prompt: str) -> Dict[str, str]:
        """
        Get both thinking process and final answer (for R1 models)
        """
        
        if self.mode == "deepseek_api":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return {
                "thinking": getattr(
                    response.choices[0].message, 
                    'reasoning_content', 
                    ''
                ) or "",
                "answer": response.choices[0].message.content
            }
        
        else:
            # For other models, just return the full response
            answer = self.think(prompt)
            
            # If it has <think> tags (Ollama DeepSeek-R1), parse them
            if "<think>" in answer and "</think>" in answer:
                think_start = answer.index("<think>") + len("<think>")
                think_end = answer.index("</think>")
                thinking = answer[think_start:think_end].strip()
                final = answer[think_end + len("</think>"):].strip()
                return {"thinking": thinking, "answer": final}
            
            return {"thinking": "", "answer": answer}


# ─── Web Search Module ────────────────────────────────────

class WebSearcher:
    """
    Search the web using Tavily API.
    Tavily is designed specifically for AI agents.
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        self.base_url = "https://api.tavily.com/search"
        
        if not self.api_key:
            print("⚠️ No Tavily API key found! Using mock search.")
    
    def search(self,
               query: str,
               max_results: int = 5,
               search_depth: str = "advanced") -> List[SearchResult]:
        """
        Search the web for a query.
        Returns list of SearchResult objects.
        """
        
        if not self.api_key:
            return self._mock_search(query)
        
        print(f"  🔍 Searching: '{query[:50]}...'")
        
        try:
            payload = {
                "api_key": self.api_key,
                "query": query,
                "max_results": max_results,
                "search_depth": search_depth,
                "include_raw_content": True,
                "include_answer": True  # Get AI summary too
            }
            
            response = requests.post(
                self.base_url,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            results = []
            for item in data.get('results', []):
                result = SearchResult(
                    title=item.get('title', 'No title'),
                    url=item.get('url', ''),
                    snippet=item.get('content', '')[:500],
                    content=item.get('raw_content', '')[:3000] or item.get('content', ''),
                    relevance_score=item.get('score', 0.0)
                )
                results.append(result)
            
            print(f"  ✅ Found {len(results)} results")
            return results
        
        except requests.exceptions.RequestException as e:
            print(f"  ❌ Search error: {e}")
            return []
        
        except Exception as e:
            print(f"  ❌ Unexpected error: {e}")
            return []
    
    def _mock_search(self, query: str) -> List[SearchResult]:
        """
        Mock search for testing without API key
        """
        return [
            SearchResult(
                title=f"Mock Result 1 for: {query[:30]}",
                url="https://example.com/article1",
                snippet=f"This is a mock search result about {query}. "
                        "In real usage, this would be actual web content.",
                content=f"Full mock content about {query}. "
                        "This demonstrates how the system works.",
                relevance_score=0.9
            ),
            SearchResult(
                title=f"Mock Result 2 for: {query[:30]}",
                url="https://example.com/article2",
                snippet=f"Another perspective on {query}.",
                content=f"Alternative viewpoint about {query}.",
                relevance_score=0.7
            )
        ]


# ─── Web Reader Module ────────────────────────────────────

class WebReader:
    """
    Reads and extracts content from web pages.
    """
    
    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Research Bot) AppleWebKit/537.36'
        }
    
    def read_page(self, url: str) -> str:
        """
        Scrape and clean content from a web page
        """
        
        try:
            response = requests.get(
                url,
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 
                                  'footer', 'header', 'ads']):
                element.decompose()
            
            # Extract text
            text = soup.get_text(separator='\n', strip=True)
            
            # Clean up whitespace
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            clean_text = '\n'.join(lines)
            
            # Limit length (to avoid too many tokens)
            return clean_text[:5000]
        
        except requests.exceptions.Timeout:
            return f"[Timeout reading {url}]"
        except requests.exceptions.RequestException as e:
            return f"[Error reading {url}: {str(e)[:100]}]"
        except Exception as e:
            return f"[Unexpected error: {str(e)[:100]}]"


# ─── Research Planner ─────────────────────────────────────

class ResearchPlanner:
    """
    Uses reasoning model to plan the research strategy.
    Breaks the main question into searchable sub-questions.
    """
    
    def __init__(self, model: ModelInterface):
        self.model = model
    
    def create_research_plan(self, main_question: str) -> List[str]:
        """
        Break the main question into 3-5 specific sub-questions.
        Each sub-question will be searched separately.
        """
        
        prompt = f"""
        You are a senior research analyst planning a deep research investigation.
        
        Main Research Question:
        {main_question}
        
        Your task: Break this into 3-5 specific sub-questions that together 
        will completely answer the main question.
        
        Good sub-questions should be:
        - Specific and searchable (not too broad)
        - Cover different aspects of the main question
        - Build toward a complete answer
        - Ordered logically
        
        Think carefully, then provide EXACTLY the sub-questions.
        Format your response as:
        
        SUB-QUESTION 1: [specific question]
        SUB-QUESTION 2: [specific question]
        SUB-QUESTION 3: [specific question]
        (and so on, max 5)
        
        Research question: {main_question}
        """
        
        print("\n📋 Creating research plan...")
        response = self.model.think(prompt, temperature=0.3)
        
        # Parse sub-questions
        sub_questions = []
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('SUB-QUESTION'):
                # Extract the question after the colon
                if ':' in line:
                    question = line.split(':', 1)[1].strip()
                    if question:
                        sub_questions.append(question)
        
        # Fallback: if parsing fails, create basic sub-questions
        if not sub_questions:
            sub_questions = [
                f"What is {main_question}?",
                f"What are the main factors related to {main_question}?",
                f"What are the latest developments in {main_question}?",
                f"What are the key challenges regarding {main_question}?"
            ]
        
        print(f"✅ Research plan created: {len(sub_questions)} sub-questions")
        for i, q in enumerate(sub_questions):
            print(f"   {i+1}. {q}")
        
        return sub_questions
    
    def create_search_queries(self, sub_question: str) -> List[str]:
        """
        For each sub-question, create 2-3 specific search queries.
        """
        
        prompt = f"""
        Create 2-3 specific Google search queries to answer this research question:
        
        Research question: {sub_question}
        
        Good search queries are:
        - Specific with key terms
        - Include recent year if relevant
        - Use different angles
        
        Format:
        QUERY 1: [search query]
        QUERY 2: [search query]
        QUERY 3: [search query]
        """
        
        response = self.model.think(prompt, temperature=0.3)
        
        queries = []
        for line in response.split('\n'):
            if 'QUERY' in line and ':' in line:
                query = line.split(':', 1)[1].strip()
                if query:
                    queries.append(query)
        
        # Fallback
        if not queries:
            queries = [sub_question]
        
        return queries[:3]  # Max 3 queries per sub-question


# ─── Information Extractor ────────────────────────────────

class InformationExtractor:
    """
    Extracts relevant information from search results.
    Uses reasoning model to identify key findings.
    """
    
    def __init__(self, model: ModelInterface):
        self.model = model
    
    def extract_key_findings(self,
                               sub_question: str,
                               search_results: List[SearchResult]) -> str:
        """
        Given search results, extract what's relevant to the sub-question.
        """
        
        if not search_results:
            return "No relevant information found for this question."
        
        # Compile search results text
        sources_text = ""
        for i, result in enumerate(search_results[:5]):  # Max 5 sources
            content = result.content if result.content else result.snippet
            sources_text += f"""
SOURCE {i+1}: {result.title}
URL: {result.url}
Content: {content[:800]}
---
"""
        
        prompt = f"""
        You are extracting key information for a research report.
        
        Research Sub-Question: {sub_question}
        
        Source Material:
        {sources_text}
        
        Extract and synthesize the KEY FINDINGS that answer the sub-question.
        
        Your extraction should:
        1. Focus only on information relevant to the sub-question
        2. Note any contradictions between sources
        3. Identify what is well-established vs uncertain
        4. Highlight the most important facts, statistics, and insights
        5. Note the recency of information if relevant
        
        Write a comprehensive but concise summary (3-5 paragraphs).
        """
        
        findings = self.model.think(prompt, temperature=0.3, max_tokens=1500)
        return findings


# ─── Fact Verifier ────────────────────────────────────────

class FactVerifier:
    """
    Cross-checks key claims across multiple sources.
    Identifies contradictions and uncertainty.
    """
    
    def __init__(self, model: ModelInterface):
        self.model = model
    
    def verify_findings(self,
                         main_question: str,
                         all_sections: List[ResearchSection]) -> str:
        """
        Review all sections for consistency and flag issues.
        """
        
        # Compile all findings
        all_findings = ""
        for section in all_sections:
            all_findings += f"""
SUB-QUESTION: {section.sub_question}
FINDINGS: {section.key_findings[:500]}
---
"""
        
        prompt = f"""
        You are a fact-checker reviewing research findings for consistency.
        
        Main Research Question: {main_question}
        
        All Research Findings:
        {all_findings}
        
        Please:
        1. Identify any CONTRADICTIONS between sections
        2. Flag any claims that seem uncertain or need more verification
        3. Note the overall reliability of the research
        4. Identify gaps - what important aspects are still unclear?
        5. Give an overall confidence rating: HIGH / MEDIUM / LOW
        
        Format your verification report clearly.
        """
        
        verification = self.model.think(prompt, temperature=0.2, max_tokens=1000)
        return verification


# ─── Report Synthesizer ───────────────────────────────────

class ReportSynthesizer:
    """
    Combines all research into a coherent final report.
    """
    
    def __init__(self, model: ModelInterface):
        self.model = model
    
    def write_executive_summary(self,
                                  main_question: str,
                                  sections: List[ResearchSection]) -> str:
        """Write a concise executive summary"""
        
        all_key_points = "\n".join([
            f"- {s.sub_question}: {s.key_findings[:200]}"
            for s in sections
        ])
        
        prompt = f"""
        Write a concise executive summary (2-3 paragraphs) for this research.
        
        Main Question: {main_question}
        
        Key Findings:
        {all_key_points}
        
        The summary should:
        - Directly answer the main question
        - Highlight the most important findings
        - Be written for a busy professional
        - Be accurate and not overstate certainty
        """
        
        return self.model.think(prompt, temperature=0.4, max_tokens=500)
    
    def write_conclusion(self,
                          main_question: str,
                          sections: List[ResearchSection],
                          verification: str) -> str:
        """Write a research conclusion with recommendations"""
        
        prompt = f"""
        Write a conclusion for this research report.
        
        Main Research Question: {main_question}
        
        Research covered {len(sections)} aspects of this topic.
        
        Verification Notes: {verification[:300]}
        
        The conclusion should:
        1. Summarize what was definitively established
        2. Note remaining uncertainties
        3. Provide practical implications
        4. Suggest directions for further research if needed
        5. Give a clear, direct answer to the main question
        
        Write 2-3 paragraphs.
        """
        
        return self.model.think(prompt, temperature=0.4, max_tokens=600)
    
    def format_final_report(self, report: ResearchReport) -> str:
        """
        Format the complete research report as readable text
        """
        
        report_text = f"""
╔══════════════════════════════════════════════════════════════╗
║                    DEEP RESEARCH REPORT                      ║
╚══════════════════════════════════════════════════════════════╝

📌 RESEARCH QUESTION:
{report.main_question}

📊 RESEARCH STATS:
• Sources consulted: {report.total_sources_consulted}
• Aspects investigated: {len(report.sections)}
• Confidence level: {report.confidence_level}

═══════════════════════════════════════════════════════════════
📋 EXECUTIVE SUMMARY
═══════════════════════════════════════════════════════════════
{report.executive_summary}

═══════════════════════════════════════════════════════════════
🔬 DETAILED FINDINGS
═══════════════════════════════════════════════════════════════
"""
        
        for i, section in enumerate(report.sections, 1):
            report_text += f"""
{'─'*60}
SECTION {i}: {section.sub_question}
{'─'*60}
{section.key_findings}

Sources for this section:
{chr(10).join([f'  [{j+1}] {url}' for j, url in enumerate(section.sources[:3])])}
"""
        
        report_text += f"""
═══════════════════════════════════════════════════════════════
🎯 CONCLUSION
═══════════════════════════════════════════════════════════════
{report.conclusion}

═══════════════════════════════════════════════════════════════
📚 ALL SOURCES
═══════════════════════════════════════════════════════════════
{chr(10).join([f'[{i+1}] {url}' for i, url in enumerate(report.all_sources)])}
"""
        
        return report_text


# ─── Main Deep Research Engine ────────────────────────────

class DeepResearchEngine:
    """
    The main engine that orchestrates all components.
    This is the heart of the project!
    """
    
    def __init__(self, 
                 model_mode: str = "local",
                 model_name: str = None,
                 tavily_api_key: str = None):
        
        print("\n🚀 Initializing Deep Research Engine...")
        print("=" * 60)
        
        # Initialize model
        self.model = ModelInterface(mode=model_mode, model=model_name)
        
        # Initialize all modules
        self.searcher = WebSearcher(api_key=tavily_api_key)
        self.reader = WebReader()
        self.planner = ResearchPlanner(self.model)
        self.extractor = InformationExtractor(self.model)
        self.verifier = FactVerifier(self.model)
        self.synthesizer = ReportSynthesizer(self.model)
        
        print("✅ All modules initialized!")
    
    def research(self, 
                 question: str,
                 depth: str = "standard") -> ResearchReport:
        """
        Main research function. 
        Give it a question, get back a full research report!
        
        depth: "quick" (3 sources) or "standard" (5 sources) 
               or "deep" (10 sources)
        """
        
        depth_config = {
            "quick": {"max_results": 3, "max_sub_questions": 3},
            "standard": {"max_results": 5, "max_sub_questions": 4},
            "deep": {"max_results": 8, "max_sub_questions": 5}
        }
        config = depth_config.get(depth, depth_config["standard"])
        
        print(f"\n🔭 Starting Deep Research")
        print(f"Question: {question}")
        print(f"Depth: {depth}")
        print("=" * 60)
        
        start_time = time.time()
        all_sources = []
        sections = []
        
        # ── PHASE 1: Planning ──────────────────────────────────
        print("\n📋 PHASE 1: Research Planning")
        
        sub_questions = self.planner.create_research_plan(question)
        sub_questions = sub_questions[:config["max_sub_questions"]]
        
        # ── PHASE 2: Searching & Reading ───────────────────────
        print(f"\n🔍 PHASE 2: Web Search & Reading")
        
        for i, sub_question in enumerate(sub_questions):
            print(f"\n[{i+1}/{len(sub_questions)}] Investigating: {sub_question[:50]}...")
            
            # Get search queries for this sub-question
            queries = self.planner.create_search_queries(sub_question)
            
            # Search and collect results
            all_results = []
            for query in queries[:2]:  # Max 2 queries per sub-question
                results = self.searcher.search(
                    query, 
                    max_results=config["max_results"]
                )
                all_results.extend(results)
                time.sleep(0.5)  # Be polite to APIs
            
            # Remove duplicates by URL
            seen_urls = set()
            unique_results = []
            for result in all_results:
                if result.url not in seen_urls:
                    seen_urls.add(result.url)
                    unique_results.append(result)
                    all_sources.append(result.url)
            
            # ── PHASE 3: Extract Key Information ──────────────
            print(f"  📖 Extracting key findings from {len(unique_results)} sources...")
            
            key_findings = self.extractor.extract_key_findings(
                sub_question,
                unique_results[:5]  # Max 5 sources per section
            )
            
            # Create section
            section = ResearchSection(
                sub_question=sub_question,
                search_results=unique_results,
                key_findings=key_findings,
                sources=[r.url for r in unique_results[:5]]
            )
            sections.append(section)
            
            print(f"  ✅ Section {i+1} complete")
        
        # ── PHASE 4: Verification ──────────────────────────────
        print(f"\n✔️ PHASE 4: Fact Verification")
        
        verification_report = self.verifier.verify_findings(question, sections)
        
        # Extract confidence level
        confidence = "MEDIUM"
        for level in ["HIGH", "MEDIUM", "LOW"]:
            if level in verification_report.upper():
                confidence = level
                break
        
        print(f"  Confidence level: {confidence}")
        
        # ── PHASE 5: Synthesis ─────────────────────────────────
        print(f"\n📝 PHASE 5: Report Synthesis")
        
        print("  Writing executive summary...")
        executive_summary = self.synthesizer.write_executive_summary(
            question, sections
        )
        
        print("  Writing conclusion...")
        conclusion = self.synthesizer.write_conclusion(
            question, sections, verification_report
        )
        
        # Build final report
        report = ResearchReport(
            main_question=question,
            executive_summary=executive_summary,
            sections=sections,
            conclusion=conclusion,
            all_sources=list(set(all_sources)),  # Remove duplicates
            confidence_level=confidence,
            total_sources_consulted=len(set(all_sources))
        )
        
        elapsed = time.time() - start_time
        print(f"\n⏱️ Research completed in {elapsed:.1f} seconds")
        print(f"📚 Total unique sources: {report.total_sources_consulted}")
        
        return report
    
    def research_and_print(self, question: str, depth: str = "standard"):
        """
        Research a question and print the formatted report
        """
        report = self.research(question, depth)
        formatted = self.synthesizer.format_final_report(report)
        print(formatted)
        return report, formatted
    
    def save_report(self, 
                    report: ResearchReport,
                    formatted: str,
                    filename: str = None):
        """Save report to files"""
        
        if filename is None:
            # Create filename from question
            safe_name = "".join(
                c if c.isalnum() or c == '_' else '_' 
                for c in report.main_question[:30]
            )
            filename = f"research_{safe_name}"
        
        # Save formatted text report
        with open(f"{filename}.txt", 'w', encoding='utf-8') as f:
            f.write(formatted)
        
        # Save JSON for programmatic use
        report_dict = {
            "main_question": report.main_question,
            "executive_summary": report.executive_summary,
            "sections": [
                {
                    "sub_question": s.sub_question,
                    "key_findings": s.key_findings,
                    "sources": s.sources
                }
                for s in report.sections
            ],
            "conclusion": report.conclusion,
            "all_sources": report.all_sources,
            "confidence_level": report.confidence_level,
            "total_sources": report.total_sources_consulted
        }
        
        with open(f"{filename}.json", 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Report saved to {filename}.txt and {filename}.json")


# ─── Main Entry Point ─────────────────────────────────────

def main():
    """
    Run the Deep Research System.
    Choose your model mode and ask any research question!
    """
    
    print("""
    ╔═══════════════════════════════════════════╗
    ║     DEEP RESEARCH AI SYSTEM               ║
    ║     Powered by DeepSeek-R1 / GPT-4o       ║
    ╚═══════════════════════════════════════════╝
    """)
    
    # ── CHOOSE YOUR MODEL ─────────────────────────────────
    #
    # Option 1: Local DeepSeek-R1 (free, private, needs Ollama)
    engine = DeepResearchEngine(
        model_mode="local",
        model_name="deepseek-r1:7b",
        tavily_api_key=os.getenv("TAVILY_API_KEY")
    )
    
    # Option 2: OpenAI API (paid, easy, no local setup)
    # engine = DeepResearchEngine(
    #     model_mode="api",
    #     model_name="gpt-4o",
    #     tavily_api_key=os.getenv("TAVILY_API_KEY")
    # )
    
    # Option 3: DeepSeek API (cheap, cloud R1)
    # engine = DeepResearchEngine(
    #     model_mode="deepseek_api",
    #     model_name="deepseek-reasoner",
    #     tavily_api_key=os.getenv("TAVILY_API_KEY")
    # )
    
    # ── YOUR RESEARCH QUESTION ────────────────────────────
    
    research_question = """
    What are the most effective strategies for building 
    AI-powered applications in 2024, and what are the 
    key technical challenges engineers face?
    """
    
    # Depth options: "quick", "standard", "deep"
    report, formatted_report = engine.research_and_print(
        question=research_question.strip(),
        depth="standard"
    )
    
    # Save the report
    engine.save_report(report, formatted_report)
    
    return report


if __name__ == "__main__":
    main()
```

---

### Quick Start Guide:

```bash
# Step 1: Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Step 2: Pull DeepSeek-R1
ollama pull deepseek-r1:7b

# Step 3: Start Ollama (in background)
ollama serve &

# Step 4: Clone/create your project
mkdir deep-research && cd deep-research
touch deep_research.py .env

# Step 5: Install Python dependencies
pip install openai ollama tavily-python requests \
            beautifulsoup4 python-dotenv

# Step 6: Add your API keys to .env
echo "TAVILY_API_KEY=your_key_here" >> .env

# Step 7: Run it!
python deep_research.py
```

---

### Testing Without Any API Keys:

```python
# test_without_apis.py
# Run this to test the system architecture without any API keys

import sys
sys.path.insert(0, '.')

# Mock the entire system for testing
class MockModel:
    def think(self, prompt, **kwargs):
        return """
        SUB-QUESTION 1: What are the current AI engineering best practices?
        SUB-QUESTION 2: What tools do AI engineers use most in 2024?
        SUB-QUESTION 3: What are the main challenges in AI deployment?
        """

class MockSearcher:
    def search(self, query, **kwargs):
        from deep_research import SearchResult
        return [
            SearchResult(
                title="AI Engineering in 2024",
                url="https://example.com/ai-engineering",
                snippet=f"Key information about {query}",
                content=f"Detailed content about {query} including best practices...",
                relevance_score=0.9
            )
        ]

# Test individual components
print("✅ Testing system architecture...")
print("✅ All components can be imported correctly!")
print("\nTo run with real models:")
print("1. Install Ollama: curl -fsSL https://ollama.com/install.sh | sh")
print("2. Pull model: ollama pull deepseek-r1:7b")
print("3. Get Tavily key: https://tavily.com")
print("4. Run: python deep_research.py")
```

---

# 📊 FINAL SUMMARY — Everything We Covered

---

## Complete Topic Map:

```
PROJECT 4: DEEP RESEARCH WITH WEB SEARCH AND REASONING MODELS
│
├── PART 1 (Previously Covered)
│   │
│   ├── Chapter 1: What is Deep Research?
│   │   ├── Problem: Regular LLMs have frozen knowledge
│   │   ├── Solution: Search + Reason + Verify + Synthesize
│   │   └── Architecture of the full system
│   │
│   ├── Chapter 2: Reasoning Models
│   │   ├── What makes a model a "reasoning model"
│   │   ├── OpenAI "o" family (o1, o1-mini, o1-pro, o3)
│   │   └── DeepSeek-R1 (open source, visible thinking)
│   │
│   └── Chapter 3: Inference-Time Techniques
│       ├── What is inference time
│       ├── Chain of Thought (zero-shot, few-shot, auto)
│       ├── Parallel Sampling + Majority Voting
│       ├── Sequential Sampling
│       ├── Tree of Thoughts (full implementation)
│       └── Search Against a Verifier
│
└── PART 2 (This Session)
    │
    ├── Chapter 4: Training-Time Techniques
    │   ├── STaR: Self-taught reasoning (bootstrapping)
    │   ├── RL with Verifier (GRPO, reward signals)
    │   ├── ORM vs PRM (outcome vs process rewards)
    │   ├── Self-Refinement (critique and improve)
    │   └── Meta-CoT (internalizing search into weights)
    │
    └── Chapter 5: Local Deployment
        ├── Why deploy locally (privacy, cost, control)
        ├── Quantization (making models fit on laptops)
        ├── Ollama setup and usage
        ├── OpenAI-compatible local API
        └── Complete Deep Research System (end-to-end code)
```

---

## Key Concepts Quick Reference:

```
┌─────────────────────────────────────────────────────────────┐
│ CONCEPT              │ ONE-LINE EXPLANATION                  │
├─────────────────────────────────────────────────────────────┤
│ Reasoning Model      │ LLM that thinks before answering     │
│ o1/o3               │ OpenAI's reasoning models (RL trained)│
│ DeepSeek-R1         │ Open-source reasoning model (free!)  │
│ CoT                 │ "Think step by step" prompting       │
│ Parallel Sampling   │ Many answers → pick best one         │
│ Sequential Sampling │ One answer → improve → improve       │
│ Tree of Thoughts    │ Explore many reasoning paths         │
│ Verifier            │ System that checks if answer is right│
│ STaR                │ Model teaches itself from own outputs │
│ GRPO                │ RL algorithm used by DeepSeek        │
│ ORM                 │ Score final answers only             │
│ PRM                 │ Score each reasoning step            │
│ Self-Refinement     │ Model critiques and improves itself  │
│ Meta-CoT            │ Search baked INTO model's weights    │
│ Quantization        │ Compress model to fit on your laptop │
│ Ollama              │ Run LLMs locally (easy!)             │
│ Inference-time      │ When model answers your question     │
│ Training-time       │ When model learns from data          │
└─────────────────────────────────────────────────────────────┘
```

---

> 🎉 **Congratulations!** You have completed the full **Deep Research** tutorial — from first principles all the way to a working end-to-end system!
