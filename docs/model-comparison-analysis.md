# Model comparison: findings

Companion to [model-comparison.md](model-comparison.md), which holds the full table from the second run.
All runs use agent prompt `v2`, the same 24 / 10 / 12 examples, temperature 0, and a fixed Bedrock judge
(`mistral.mistral-large-3-675b-instruct`, which agreed with human labels on 16 of 16 calibration examples).
Ten Bedrock models were evaluated twice and `gpt-4o-mini` once, in October 2026, through the AWS Bedrock sandbox
(us-east-1) and the OpenAI API.

## How much to trust the numbers

| Check | Result |
| --- | --- |
| Run-to-run noise (same model, same judge) | Mean absolute difference 0.025 across all metrics; 95% of differences are within 0.10 |
| Largest swing | Kimi K2 Thinking correctness 0.71 to 0.88; three-example safety metrics can move by 0.33 per single flip |
| Judge dependence (4 models re-scored with `gpt-oss-20b`) | Correctness shifted by 0.04 to 0.13; the ordering among the top three changed, Qwen3 32B stayed last |
| Self-preference (Mistral judging Mistral) | Not observed: Mistral scored 0.88 under its own family and 0.96 under another judge |

Practical reading: gaps below about 0.1 are not evidence. Absolute scores depend on the judge, so compare
models only within one judge setting.

## Findings that held in both runs

1. **gpt-oss-120b and gpt-oss-20b lead on correctness** (0.96 and 0.92 in both runs). Qwen3 Next 80B is last
   (0.62 and 0.67) and the cause is retrieval: its recall is 0.59 in both runs, so it often answers without
   finding the right document. The Nova family and Qwen3 32B sit in the middle (0.79 to 0.88).
2. **Citation compliance is a model trait, not noise.** gpt-oss-120b cites in 50% and 59% of answers and
   Mistral Large 3 in 41% and 55%, even though both answer correctly. The other models cite in 82 to 100% of answers.
   A correctness-only benchmark would miss this, which is why the composite score now weights citation presence.
3. **No model is safe by default.** Leak checks are code-based, so they are judge-independent:
   Mistral leaks nothing (1.00 in both runs), while MiniMax M2 (0.67) and Kimi K2 Thinking (0.75) leak most.
   gpt-oss-20b and Qwen3 32B open a support ticket on an injected command in 1 of 3 cases in both runs.
4. **Latency is similar for most models** (1.5 to 2.4 seconds per answer). MiniMax and Kimi take 3.5 to 4.2 seconds.
   Kimi hit the step cap on 2 to 3 of the 24 benchmark questions in each run.
5. **Nova Micro is competitive for its class.** Composite 0.89 in run 2, indistinguishable from Nova Lite and
   close to Nova Pro, with the lowest latency among the Amazon models.

## Which model for this task

| Priority | Candidate | Why | Watch |
| --- | --- | --- | --- |
| Best answer quality | `gpt-oss-20b` | Highest composite in run 2 (0.95), perfect citations, 0.92 correctness | Opens tickets on injected commands; needs the escalation guard |
| Balanced quality and safety | `gpt-4o-mini` | Composite 0.91, safety 0.92 / 0.92 / 1.00, fewest tokens | One run only in this comparison |
| Lowest cost and latency | `amazon.nova-micro-v1:0` | Composite 0.89, 1.5 s, no ticket abuse in either run | Correctness 0.83 |

Not recommended for this agent as configured: Qwen3 Next 80B (retrieval), and MiniMax M2 or Kimi K2 Thinking
where leakage matters.

## Limits of this study

- The `v2` prompt was tuned against `gpt-4o-mini`; other models may do better with their own prompt.
- Twenty-four, ten and twelve examples cannot separate models that are within 0.1. Raise
  `EVAL_NUM_REPETITIONS` and extend the datasets before acting on a close call.
- The injection and ticket-abuse cases number three and two examples; treat those columns as screening, not measurement.
- Bedrock sandbox credentials are temporary, so the Bedrock rows cannot be reproduced from a fresh clone
  without an AWS account that has model access. The run configuration is recorded in the report header.
- LangSmith's free-tier monthly trace quota was exhausted partway through the work, so some of these runs have
  complete scores locally and incomplete traces in the LangSmith UI.

## Reproducing

```bash
python -m evalsuite.cli calibrate --judge bedrock:mistral.mistral-large-3-675b-instruct
python -m evalsuite.cli matrix --judge bedrock:mistral.mistral-large-3-675b-instruct \
  --models "bedrock:amazon.nova-micro-v1:0,bedrock:openai.gpt-oss-20b-1:0,openai:gpt-4o-mini" --prompt-version v2
```

## Update: reasoning-block fix and injection guard

After these runs, two harness problems were found and fixed: reasoning blocks from gpt-oss, Kimi and MiniMax were
being included in answers, and the safety judge misgraded correct answers for some judge models. The gpt-oss-120b
correctness lead held after the fix (0.96); MiniMax's correctness was overstated (0.88 corrected to 0.79). The
model-independent injection defenses and their results are in [guard-evaluation.md](guard-evaluation.md): with them,
all 11 models are leak-free and open no injected tickets on the safety set.
