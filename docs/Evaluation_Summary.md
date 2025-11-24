# Evaluation Summary for FinR1 and Qwen7B-Instruct  
This document summarizes the behavior of **FinR1** and **Qwen7B-Instruct** across three prompt variants (P1, P2, P3) on a 20-sample FinQA dev subset.  
Metrics evaluated:

- **Formatting Accuracy** → % outputs with exactly one valid `\boxed{...}`  
- **Rationale Length** → tokens inside `<think>...</think>`

---

## FinR1 Results  
FinR1 naturally produces long chain-of-thought explanations. Prompt structure strongly influences depth, with reasoning growing from **P1 → P2 → P3**.

| Model | Prompt | Formatting Accuracy | Avg Rationale Tokens | Min | Max  |
|--------|--------|----------------------|------------------------|-----|------|
| FinR1 | P1 | **55%** | **539.8** | 0 | 1636 |
| FinR1 | P2 | **65%** | **642.0** | 0 | 1604 |
| FinR1 | P3 | **65%** | **825.5** | 0 | 2160 |

### **FinR1 Interpretation**  
- P1 produces shorter and less stable reasoning.  
- P2 improves structure, with moderate reasoning length.  
- **P3 generates the longest and most complete reasoning**, allowing FinR1 to use its full multi-step capabilities.  
  - This is why **P3 is recommended over P2**, even though formatting accuracy is tied — P3 encourages deeper, more consistent reasoning.

---

## Qwen7B-Instruct Results  
Qwen prefers short, compact reasoning. It is stable on P1 and P3, but P2 disrupts formatting.

| Model | Prompt | Formatting Accuracy | Avg Rationale Tokens | Min | Max |
|--------|--------|----------------------|------------------------|-----|-----|
| Qwen7B | P1 | **95%** | **56.3** | 0 | 121 |
| Qwen7B | P2 | **55%** | **78.5** | 0 | 291 |
| Qwen7B | P3 | **95%** | **56.3** | 0 | 121 |

### **Qwen Interpretation**  
- P1 and P3 give extremely stable formatting (95%).  
- P2 increases verbosity and reduces structure.  
- Qwen does not benefit from long reasoning, and performs similarly at 2k vs 4k output tokens.

---

## Recommended Prompt Choices

| Model | Best Prompt(s) | Reason |
|--------|------------------|--------|
| **FinR1** | **P3** | Produces deeper and more complete reasoning than P2 |
| **Qwen7B** | **P1 or P3** | Most stable formatting (95%) with concise reasoning |

---

This evaluation shows that **FinR1 thrives with richer multi-step reasoning**, while **Qwen7B performs best with concise, lightweight prompts**.
