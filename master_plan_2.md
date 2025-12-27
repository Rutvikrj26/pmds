# Scallop-Titans Agent: Master Development Plan 2.0 (Strategic Pivot)

> **Version:** 2.1 | **Date:** December 2024
> **Theme:** "Small Model + Big Tool = Fast Intelligence"

---

## Part A: Experimental Findings (Model Selection)

We tested 4 models on 5 complex "Secure Routing" samples (k=10 hops):

| Model | Accuracy | Tool Usage | Inference Speed | Decision |
|---|---|---|---|---|
| Mistral-24B (SFT v1) | 60% | 0% | Fast | ❌ Too smart, bypasses tools |
| Qwen 2.5-7B (base) | 40% | 40% | ~35s/sample | ❌ Lazy, selective tool use |
| Qwen 2.5-3B (base) | 60% | 0% | ~31s/sample | ❌ Too smart, ignores tools |
| Qwen 2.5-1.5B (base) | 40% | 60% | ~100s/sample | ⚠️ Uses tools but chatty |
| **Salesforce xLAM-1B** | 40% | 0% | **~3s/sample** | ✅ Fast, concise, needs SFT |

**Conclusion:** xLAM-1B is **30x faster** than Qwen 1.5B and produces concise outputs. It doesn't use our `<|call_scallop|>` token because it wasn't trained on it — but SFT will fix this.

---

## Part B: Architecture 2.0

#### A. LLM (The "Fast Executor")
| Attribute | Specification |
|---|---|
| **Base Model** | **Salesforce/xLAM-1B-fc-r** |
| **Parameters** | 1 Billion |
| **Strength** | Purpose-built for function calling, ultra-concise outputs |
| **Weakness** | Doesn't know `<|call_scallop|>` → Requires SFT |
| **Inference (vLLM)** | Est. **~0.3-0.5s/sample** with KV caching |

#### B. Titans Memory + Scallop Engine
*   No changes from v1.

---

## Part C: Data Strategy (200k Samples)

| Category | Size | Domains |
|---|---|---|
| **Robotics/Physical** | 70k | Safety Interlocks, Navigation, Object Manipulation |
| **Enterprise** | 60k | Logistics, Supply Chain, RBAC, Compliance |
| **Societal/Legal** | 30k | Healthcare Protocols, Legal Contracts, Privacy |
| **Academic** | 20k | CLUTRR, GSM8K |
| **Negatives** | 20k | Distractors and contradictions |

---

## Part D: Training Pipeline 2.0

### Phase 1: SFT (Tool Syntax)
*   **Goal:** Teach xLAM-1B to recognize `<|call_scallop|>` and emit valid Datalog queries.
*   **Config:** Full Fine-Tuning on 1B model (~30 minutes on 3x H100).
*   **Milestone:** 90%+ Syntax Validity on tool calls.

### Phase 2: GRPO (Reinforcement Learning)
*   **Goal:** Maximize tool usage + correctness, minimize verbosity.
*   **Reward Function (Revised):**

| Component | Weight | Rationale |
|---|---|---|
| `+1.0` Correct Answer | High | Core objective |
| `+0.8` **Valid Scallop Tool Call** | **Very High** | Key change: strongly incentivize tool use |
| `+0.2` Concise Output (<100 tokens) | Medium | Leverage xLAM's natural brevity |
| `-0.2` No Tool Call Attempted | Penalty | Punish pure CoT on hard tasks |
| `-0.05` Per token after 50 | Low | Soft length penalty |

---

## Part E: Execution Plan (Timeline: 3 Weeks)

| Week | Task | Milestone |
|---|---|---|
| **1** | Multi-Domain Generator + Data Gen | 200k samples generated |
| **2** | SFT on xLAM-1B | 90%+ tool syntax validity |
| **3** | GRPO + Final Eval | >70% Tool Usage, >60% Accuracy |

---

## Part F: Risk Register

| Risk | Mitigation |
|---|---|
| xLAM-1B "Semantic Blindness" | Use **"Tools as Ontology"** prompting. Map complex sentences to atomic tool calls. Use Scallop to catch logical contradictions. |
| License Restrictions (CC-BY-NC) | Acknowledge this is a **Research PoC**. If commercializing later, swap for Qwen 2.5-1.5B (Apache 2.0). |
| Tool overhead slows inference | Implement vLLM batching + Scallop caching. |
