# Scallop-Titans Agent: Master Development Plan

> **Version:** 1.0 | **Last Updated:** 2024-12-08

---

## Part A: Architecture

### 1. System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SCALLOP-TITANS AGENT                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────┐    ┌─────────────────────────────────────────────────┐   │
│   │    USER     │───▶│         CHAT INTERFACE (Gradio/vLLM)            │   │
│   └─────────────┘    └───────────────────┬─────────────────────────────┘   │
│                                          │                                  │
│                                          ▼                                  │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    LLM (Qwen2.5-1.5B-Instruct)                      │   │
│   │  - Generates text including <think> and <call_scallop> tokens       │   │
│   └───────────────────────────────┬─────────────────────────────────────┘   │
│                                   │                                         │
│                    ┌──────────────┴──────────────┐                          │
│                    ▼                             ▼                          │
│   ┌────────────────────────────┐   ┌────────────────────────────────────┐   │
│   │   Text Generation Path    │   │   Tool Call Path                   │   │
│   │   (Normal Response)       │   │   (Scallop Invocation)             │   │
│   └────────────────────────────┘   └───────────────┬────────────────────┘   │
│                                                    │                        │
│                                                    ▼                        │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                        TITANS MEMORY MODULE                         │   │
│   │  - MLP with Fast Weights (torch.func)                               │   │
│   │  - "Surprise" based update: high surprise = remember                │   │
│   │  - Outputs: Probabilistic facts P(relation | entity_a, entity_b)    │   │
│   └───────────────────────────────┬─────────────────────────────────────┘   │
│                                   │                                         │
│                                   ▼                                         │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                     SCALLOP REASONING ENGINE                        │   │
│   │  - Datalog rules (kinship, transitivity, math axioms)               │   │
│   │  - Provenance semiring for differentiable reasoning                 │   │
│   │  - Outputs: Answer entity with probability                          │   │
│   └───────────────────────────────┬─────────────────────────────────────┘   │
│                                   │                                         │
│                                   ▼                                         │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                  RESULT INJECTION INTO LLM                          │   │
│   │  - <scallop_result> token + structured output                       │   │
│   │  - LLM continues generation                                          │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2. Component Specifications

#### A. LLM (The "Brain")
| Attribute | Specification |
|---|---|
| **Base Model** | `Qwen/Qwen3-32B` (Recommended) — fits on 1x H100, leaves 2 GPUs for Scallop/spare |
| **Why Qwen3-32B?** | Outperforms Qwen2.5-72B on reasoning. Native "Thinking Mode". 36T training tokens. Superior agentic capabilities. |
| **Special Tokens** | Leverage Qwen3's native `<think>`/`</think>`. Add only: `<|call_scallop|>`, `<|scallop_result|>` |
| **Context Length** | 32K native, extendable to 256K (via YaRN). Supports very long reasoning chains. |
| **Fallback** | If 32B is slow for RL iteration, drop to Qwen3-14B for faster cycles. |

> **Note:** Qwen3's built-in Thinking Mode reduces training burden—the model already knows how to "think out loud" before answering.

#### B. Titans Memory (The "Episodic Memory")
| Attribute | Specification |
|---|---|
| **Architecture** | Multi-Layer Perceptron (2 layers, hidden_dim=256) |
| **Fast Weights** | Implemented via `torch.func.functional_call` for in-context weight updates |
| **Surprise Mechanism** | `surprise = ||grad(loss)||` where loss = prediction error on current input |
| **Update Rule** | `W_new = W_old + lr * surprise * grad(W)` (higher surprise = larger update) |
| **Forgetting** | Weight decay `W = W * (1 - decay_rate)` applied each step |

> **Research Spike (RS-1): Titans Layer Depth**
> - *Question:* How many MLP layers are optimal for relation encoding?
> - *Experiment:* Test 1, 2, 4 layers on CLUTRR validation set.
> - *Metrics:* Accuracy vs. Memory Size vs. Training Speed.

#### C. Scallop Engine (The "Logic Core")
| Attribute | Specification |
|---|---|
| **Language** | Datalog with negation, aggregation, and recursion |
| **Python Binding** | `scallopy` (PyTorch-compatible) |
| **Differentiability** | Via provenance semiring (`difftopkproofs` recommended) |
| **Rule Bank** | Kinship (CLUTRR), Transitivity, Inheritance, Math Axioms |

> **Research Spike (RS-2): Gradient Flow Verification**
> - *Question:* Does `scallopy` propagate gradients through recursive rules back to a foreign predicate?
> - *Experiment:* Create minimal test: `foreign_pred -> rule -> loss.backward()`. Check if foreign_pred's weights update.
> - *Priority:* **CRITICAL** (Blocks entire architecture if fails).
> - *Fallback:* If fails, use Scallop inference-only; train Titans separately via distillation.

---

## Part B: Data Strategy

### 1. The "Four Pillars" of Training Data

| Pillar | Source | Size | Purpose |
|---|---|---|---|
| **I. Synthetic Logic Traces** | Custom Generator | 200k | Teach LLM to emit `<call_scallop>` with correct syntax |
| **II. CLUTRR (Kinship)** | [HuggingFace CLUTRR/v1](https://huggingface.co/datasets/CLUTRR/v1) | ~10k | Benchmark for relational reasoning |
| **III. GSM8K (Math)** | [OpenAI GSM8K](https://huggingface.co/datasets/gsm8k) | 8.5k | Test generalization to math reasoning |
| **IV. Near-Miss Negatives** | Generated | 50k | Robustness to contradictions and red herrings |

### 2. Data Format (SFT)
```json
{
  "messages": [
    {"role": "system", "content": "You are a reasoning agent. Use <call_scallop> to invoke logic."},
    {"role": "user", "content": "Alice's mother is Betty. Betty's sister is Carol. Who is Alice's aunt?"},
    {"role": "assistant", "content": "<|start_thought|>I need to find the sister of Alice's mother.<|call_scallop|>add_fact(mother, alice, betty). add_fact(sister, betty, carol). query(aunt, alice, ?)<|end_thought|><|scallop_result|>[(carol, 0.98)]<|end_scallop_result|>Alice's aunt is Carol."}
  ]
}
```

### 3. Data Generation Pipeline
1.  **CLUTRR Conversion:** Parse CLUTRR stories -> Extract entities -> Generate Scallop traces.
2.  **GSM8K Conversion:** Parse word problems -> Identify quantities -> Generate arithmetic Scallop rules.
3.  **Negative Mining:** For each positive, create 2 negatives:
    *   **Contradiction:** "Alice's mother is Betty. Alice's mother is Denise." -> Query should return Contradiction.
    *   **Distraction:** Add irrelevant facts. Correct answer should be unchanged.

> **Research Spike (RS-3): GSM8K to Scallop Feasibility**
> - *Question:* What percentage of GSM8K problems can be expressed in Datalog?
> - *Experiment:* Manually convert 50 problems. Track success/failure reasons.
> - *Decision:* If <50% convertible, deprioritize GSM8K; focus on CLUTRR for PoC.

---

## Part C: Training Pipeline

### Phase 0: Pre-Training the Titans Memory (Standalone)
**Goal:** Prime the memory module to encode relations before LLM integration.
| Attribute | Specification |
|---|---|
| **Task** | Masked Relation Prediction: Given (A, ?, B), predict the masked relation. |
| **Data** | 100k synthetic relation triplets from CLUTRR generator. |
| **Loss** | Cross-Entropy on relation type. |
| **Duration** | ~2 hours on single GPU. |

### Phase 1: Supervised Fine-Tuning (SFT)
**Goal:** Teach the LLM to generate `<call_scallop>` tokens and interpret `<scallop_result>` tokens.
| Attribute | Specification |
|---|---|
| **Data** | Pillar I + Pillar II (200k + 10k examples, mixed). |
| **Method** | LoRA (rank=16, alpha=32) via `unsloth` or `peft`. |
| **Trainer** | `SFTTrainer` from `trl`. |
| **Epochs** | 1-2 (LLMs overtrain quickly on synthetic data). |
| **Batch Size** | 8 (with gradient accumulation = 4). |

### Phase 2: Reinforcement Learning (GRPO)
**Goal:** Reward correct *reasoning paths*, not just final answers.
| Attribute | Specification |
|---|---|
| **Algorithm** | GRPO (Group Relative Policy Optimization) via `trl.GRPOTrainer`. |
| **Group Size** | 4 completions per prompt. |
| **Reward Function** | See table below. |
| **KL Penalty** | 0.05 (prevent policy collapse). |
| **Epochs** | 1-3 (careful: RL is unstable). |

**Reward Function:**
```python
def compute_reward(completion, ground_truth):
    score = 0.0
    # 1. Correctness
    if extract_answer(completion) == ground_truth:
        score += 1.0
    # 2. Format (valid Scallop syntax)
    if is_valid_scallop_syntax(completion):
        score += 0.2
    # 3. Efficiency (penalize excessive tokens)
    score -= 0.01 * len(tokenize(completion))
    # 4. Logic Verification (run Scallop, check if derivation holds)
    if scallop_verify(completion):
        score += 0.3
    return score
```

> **Research Spike (RS-4): GRPO Hyperparameters**
> - *Question:* What are optimal KL penalty and learning rate for 1.5B model?
> - *Reference:* DeepSeek-R1 paper (Section 4.2).
> - *Experiment:* Grid search on small subset (1k examples).

---

## Part D: Inference & Deployment

### 1. Session State Management
The Titans Memory must persist across conversation turns within a session.

| Strategy | Description | Trade-off |
|---|---|---|
| **In-Memory** | Hold Titans weights in GPU memory per session. | Fast, but memory-intensive for many users. |
| **Serialization** | Save weights to disk/Redis after each turn. | Slower, but scalable. |
| **Prefix Caching** | Use vLLM prefix caching for LLM context. | Faster repeated prompts. |

**Recommended:** Hybrid. Use In-Memory for Titans (small ~100KB footprint). Use vLLM prefix caching for LLM.

### 2. The Inference Loop (Pseudocode)
```python
class ScallopTitansAgent:
    def __init__(self, llm, titans_memory, scallop_engine):
        self.llm = llm
        self.titans = titans_memory
        self.scallop = scallop_engine
    
    def chat(self, user_message: str, history: List[Dict]) -> str:
        # 1. Construct prompt
        prompt = self._build_prompt(user_message, history)
        
        # 2. Generate LLM response (may contain tool calls)
        response = ""
        while True:
            chunk = self.llm.generate(prompt + response, stop=["<|call_scallop|>", "<|end_response|>"])
            response += chunk
            
            # 3. Check for tool call
            if "<|call_scallop|>" in chunk:
                # Extract Scallop command
                scallop_cmd = self._extract_scallop_cmd(response)
                
                # 4. Update Titans Memory (may add new facts)
                self.titans.update(scallop_cmd)
                
                # 5. Query Scallop with Titans as fact source
                result = self.scallop.query(scallop_cmd, fact_source=self.titans)
                
                # 6. Inject result
                response += f"<|scallop_result|>{result}<|end_scallop_result|>"
            else:
                break  # No more tool calls
        
        return response.split("<|end_thought|>")[-1].strip()  # Return final answer
```

### 3. Serving Options
| Option | Use Case | Notes |
|---|---|---|
| **vLLM + Custom Tool Parser** | Production | Best performance. Requires custom tool parser registration. |
| **Ollama + Python Wrapper** | Local Dev | Easy setup. Slightly slower. |
| **Gradio Demo** | Interactive Demo | Good for visualization. |

> **Research Spike (RS-5): vLLM Tool Parser Integration**
> - *Question:* How to register a custom tool that invokes external Python code (Scallop engine)?
> - *Reference:* vLLM docs: [Tool Calling](https://docs.vllm.ai/en/stable/features/tool_calling.html).
> - *Experiment:* Create minimal vLLM server with dummy tool. Verify round-trip.

---

## Part E: Evaluation & Benchmarks

### 1. Primary Benchmarks
| Benchmark | Metric | Target (PoC) | Target (Production) |
|---|---|---|---|
| **CLUTRR (k=2)** | Accuracy | 85% | 95% |
| **CLUTRR (k=4)** | Accuracy | 70% | 85% |
| **GSM8K** | Accuracy | 50% | 70% |

### 2. Secondary Metrics
| Metric | Description | Why It Matters |
|---|---|---|
| **Proof Depth Correlation** | Does accuracy drop linearly or exponentially with hop count? | Measures multi-hop robustness. |
| **Surprise Dynamics** | Does Titans "surprise" spike on new entities and decay on repeats? | Validates memory mechanism. |
| **Reasoning Validity** | % of answers where Scallop derivation is logically sound. | Measures logical rigor. |

---

## Part F: Risk Register & Contingencies

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| **RS-2 Fails (No gradient flow)** | Medium | Critical | Use Scallop inference-only. Train Titans via distillation. |
| **Scallop is too slow for RL** | Medium | High | Batch Scallop calls. Use `top-k` proofs (smaller graph). |
| **LLM forgets tool syntax during RL** | High | Medium | Add format reward (+0.2). Use conservative KL penalty. |
| **Titans Memory explodes** | Low | Medium | Implement weight decay. Cap max memory size. |

---

## Part G: Timeline & Resources

### Timeline (10 Weeks)
| Week | Phase | Deliverable |
|---|---|---|
| 1 | Setup & Spikes | Environment. **RS-1, RS-2** resolved. |
| 2 | Titans Core | `NeuralMemory` class with surprise mechanism. Unit tests. |
| 3 | Scallop Bridge | `titans_foreign_predicate`. Integration test. |
| 4 | Data Generation | 200k synthetic traces. CLUTRR conversion. |
| 5-6 | SFT | Fine-tuned LLM. Eval on CLUTRR (k=2). |
| 7-8 | GRPO | RL-tuned model. Eval on CLUTRR (k=4). |
| 9 | Inference & Demo | vLLM server or Gradio demo. |
| 10 | Documentation | Paper draft. Model card. Code cleanup. |

### Compute (Updated: 3x NVIDIA H100 NVL Available)
| Resource | Available | Usage Plan |
|---|---|---|
| **GPU** | 3x NVIDIA H100 NVL (96GB VRAM each, 288GB total) | Far exceeds requirements. Use multi-GPU for faster training. |
| **Storage** | 50GB (model checkpoints, data). | Unchanged. |
| **Cloud** | Not needed - local hardware sufficient. | N/A |

**Hardware-Adjusted Model Recommendations:**

| Original Plan | Updated Plan | Rationale |
|---|---|---|
| Qwen2.5-1.5B | **Qwen2.5-7B** or **Qwen2.5-14B** | With 3x H100, we can comfortably train a 7B-14B model. Larger = better reasoning. |
| 1 GPU | **2 GPUs for training** (1 for Scallop overhead) | Use `accelerate` or FSDP for distributed training. Keep 1 GPU for Scallop batch inference. |
| LoRA (rank=16) | **Full Fine-Tuning** or LoRA (rank=64) | With 96GB+ VRAM, full fine-tuning is feasible for 7B. |

**Revised Training Speed Estimates:**
| Phase | Original (1x A100-80GB) | Updated (2x H100 NVL) |
|---|---|---|
| SFT (200k examples) | ~8 hours | ~2-3 hours |
| GRPO (4 samples/prompt) | ~24 hours | ~6-8 hours |


### Key Dependencies
| Library | Version | Purpose |
|---|---|---|
| `transformers` | >=4.40 | Base LLM loading. |
| `trl` | >=0.10 | `GRPOTrainer`, `SFTTrainer`. |
| `scallopy` | >=0.2 | Differentiable logic. |
| `unsloth` | latest | Fast LoRA fine-tuning. |
| `vllm` | >=0.5 | Inference serving. |

---

## Part H: Final Deliverables

1.  **`ScallopTitansAgent` Class:** Single PyTorch module wrapping LLM, Titans, and Scallop.
2.  **Trained Model Weights:** LoRA adapters for SFT and GRPO.
3.  **Inference Server:** vLLM-based with custom tool parser.
4.  **Demo Notebook:** Interactive Gradio demo showing reasoning traces.
5.  **Technical Report:** Documenting architecture, results, and ablations.
