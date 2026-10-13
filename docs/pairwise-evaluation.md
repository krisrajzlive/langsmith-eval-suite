# Pairwise evaluation: prompt v1 vs tightened v2

A pairwise evaluation asks a judge which of two answers is better for the same question, instead of scoring each answer
on its own. It is the right tool when "better" is relative and a numeric rubric is hard to write. This note records the
first live run of the pairwise evaluator and what it showed.

## Setup

- **Compared experiments:** `regression-v1-a653b454` (prompt v1) and `tightened-v2-8fe294a8` (the tightened v2 prompt).
  Both ran `gpt-4o-mini` over the same 24 documentation questions, so the only difference is the system prompt.
- **Judge:** `gpt-oss-120b` on Bedrock, with structured output. The answer order shown to the judge is randomized for
  every example to reduce position bias.
- **Two rubrics** (`--rubric`): `default` prefers the answer that is accurate, well grounded (cites sources) and concise;
  `accuracy` prefers the answer that is accurate, complete and concise and ignores citations.
- **Repetition:** each rubric was run three times, with a fresh random order each time.

```bash
python -m evalsuite.cli pairwise regression-v1-a653b454 tightened-v2-8fe294a8 \
  --judge bedrock:openai.gpt-oss-120b-1:0 --rubric accuracy
```

Only experiments whose runs were fully uploaded to LangSmith can be compared. Experiments created after the free-tier
monthly trace quota ran out have scores locally but no stored runs, so the Bedrock model experiments could not be used.

## Results

| Rubric | v2 preferred | v1 preferred | Ties | Verdicts identical across 3 passes |
| --- | --- | --- | --- | --- |
| default (rewards citations) | 21 | 3 | 0 | 24 of 24 |
| accuracy only | 6 | 13 | 5 | 15 of 24 |

(Counts are the majority verdict over the three passes. Single passes under the default rubric were 21 to 3 every time.)

## What this means

1. **The default result is mostly a citation result.** v2 was tuned to cite sources, and the default rubric rewards
   citing, so a 21 to 3 preference measures prompt compliance, not better answers. The verdicts are perfectly stable
   because the criterion is easy to apply, which is not the same as the comparison being informative.
2. **On accuracy alone there is no evidence v2 is better.** The majority leans to v1 (13 to 6, with 5 ties), but only 15 of
   24 verdicts repeat across passes. When both answers are correct, the judge is choosing on small differences, so this
   rubric is close to a coin flip on those questions. Read it as "no demonstrated improvement", not "v1 wins".
3. **The tightened prompt over-refuses one legitimate question.** For "What prefix do production secret API keys have?" v1
   answers correctly (`mk_live_`), while v2 replies that it cannot provide information about secret API keys. The rule
   against revealing or requesting secrets was applied to a documentation question. This also explains why that
   question (qa-001) failed correctness in every v2 benchmark run. The next prompt revision should state that documented
   key formats are fine to explain.

## Limits

- The pairwise judge was not calibrated against human preferences, unlike the correctness judge. Treat it as a screen.
- Twenty-four questions and one judge model. A second judge from a different family would show whether the accuracy-only
  verdicts are judge-specific.
- LangSmith stores only the preference scores for a comparison, not the judge's reasoning, so individual verdicts cannot be
  audited there.
- The two prompts differ in many ways at once (citation rules, scope, safety, length), so a pairwise win or loss cannot be
  attributed to any single rule.
