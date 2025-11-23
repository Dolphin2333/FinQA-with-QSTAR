# Evaluation Summary (FinR1 & Qwen7B)

Dataset: FinQA (20-sample dev subset)
Metrics:

Formatting Accuracy → % outputs with exactly one valid \boxed{...}

Rationale Length → tokens inside <think>...</think>

## FinR1 Results

FinR1 generates long chain-of-thought and is sensitive to prompt structure.
Reasoning depth increases from P1 → P2 → P3.

Model	Prompt	Formatting Accuracy	Avg Rationale	Min	Max
FinR1	P1	55%	539.8	0	1636
FinR1	P2	65%	642.0	0	1604
FinR1	P3	65%	825.5	0	2160
FinR1 Observations

P3 consistently produces the longest and richest reasoning.

P1 is unstable and produces the most formatting errors.

Some outputs have missing <think> sections (min = 0 tokens).

FinR1 benefits from higher max_new_tokens (≥ 4000) to avoid truncation.


## Qwen7B-Instruct Results

Qwen prefers short, compact reasoning and is highly stable with P1 and P3.

Model	Prompt	Formatting Accuracy	Avg Rationale	Min	Max
Qwen7B	P1	95%	56.3	0	121
Qwen7B	P2	55%	78.5	0	291
Qwen7B	P3	95%	56.3	0	121
Qwen Observations

P1 and P3 yield very stable formatting (~95%).

P2 breaks structure and drops to 55% formatting accuracy.

Qwen produces very short rationales (50–80 tokens).

Qwen performance is similar with 2000 vs 4000 tokens; no need for large limits.

## Overall Findings

FinR1 → Heavy CoT model, needs long generation and performs best on P3.

Qwen7B → Minimal CoT model, stable on P1 and P3, unstable on P2.

Long chain-of-thought does not equal higher formatting accuracy — results vary by model.