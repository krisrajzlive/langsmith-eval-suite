# Evaluation strategy

This suite follows the offline workflow described in the LangSmith evaluation docs: curate a dataset,
define evaluators, run an experiment, and compare experiments. It deliberately omits online evaluation of
production traffic.

## Principles

1. **Deterministic first.** Anything checkable with code (format, citations, tool selection, leakage, math)
   is a code evaluator. Judges are reserved for qualities that need judgment: correctness, grounding, tone,
   refusal behavior.
2. **Cheap checks gate, expensive checks inform.** The unit and trajectory checks run on every change.
   Judge-based benchmarks run on prompt, model or retrieval changes.
3. **Judges are measured, not trusted.** `make calibrate` scores the correctness judge against human labels.
   Judge scores are only meaningful when kappa is acceptable.
4. **Separate model families.** The judge model differs from the application model to limit self-preference.
5. **Skipped is not failed.** Evaluators return no score when they do not apply (for example citations on an
   unanswerable question), and aggregations exclude them rather than counting zero.

## Evaluation types

### Unit tests (`tests/`)
Dataset integrity (unique IDs, reference documents exist, tool names valid), evaluator logic, tool arithmetic
against the figures in the datasets, retrieval quality of the offline retriever on the QA set, and the agent
graph plumbing with a scripted model. These need no credentials and run on every pull request.
`tests/live` runs the real agent on a smoke subset and the full safety set through the LangSmith pytest
integration, with response caching to keep CI cost low.

### Correctness benchmark
LLM judge (openevals `CORRECTNESS_PROMPT`) comparing the answer with the reference answer on `qa.jsonl`.
Categories (factual, how-to, troubleshooting, calculation, multi-hop, unanswerable) are stored as example
metadata so results can be sliced in the LangSmith comparison view.

### RAG quality
Three reference-free judges from the LangSmith RAG evaluation pattern: groundedness (answer vs retrieved
context), helpfulness (answer vs question) and retrieval relevance (context vs question). Groundedness and
retrieval relevance are skipped when the answer came from a deterministic tool and no passages were retrieved.

### Retrieval metrics
Recall, precision and reciprocal rank per example from document IDs, plus an experiment-level summary
evaluator reporting MRR and hit rate. A failure here with a passing correctness score usually means the model
answered from parametric memory rather than the corpus.

### Agent trajectory
`agentevals` trajectory match with the mode declared per example: `superset` (at least these tools),
`subset` (no tools beyond these, used for out-of-scope requests), `unordered`, or `strict`. Tool arguments
are ignored. Separate checks enforce forbidden tools (for example no escalation for a routine question) and a
cap on tool calls. An LLM trajectory judge scores overall path quality.

### Safety and robustness
Code checks for the confidential canary in the system prompt, injected payloads echoed back, secrets and card
numbers repeated, and forbidden tool use. A judge grades whether the reply refused or answered as the example
expects. Injection-in-question cases require the legitimate part of the question to still be answered.

### Regression gate
`regression` runs the benchmark for a candidate prompt and compares per-example scores with
`baselines/qa_baseline.json`. It fails when a gated metric's mean drops by more than 0.05 or when more than one
example flips from passing to failing. Refresh the baseline deliberately with `make baseline` after an
accepted change; review the diff in code review.

### Pairwise comparison
For subjective quality, two finished experiments are compared with a judge that picks the better answer per
example. Order is randomized to counter position bias.

### Composite score
`composite_quality` is a weighted average of correctness (0.35), groundedness (0.20), citation presence (0.15),
helpfulness (0.10), citation grounding (0.10) and concision (0.10), renormalized over the metrics present for each example.
Weights live in `evaluators/composite.py`. Use it for a single trend line, not instead of the component metrics.

### Judge calibration
Sixteen labeled answers (balanced correct and incorrect, including near-miss numeric errors) are scored by the
correctness judge. The summary evaluator reports accuracy, true positive and true negative rates and Cohen's
kappa. Expand the set whenever a judge disagreement is found during review.

## Extending

- **New metric:** add a function with the `(inputs, outputs, reference_outputs)` signature in
  `evaluators/`, include it in the relevant runner, and add a unit test.
- **New dataset example:** append to the JSONL file with a unique ID and run `make seed`.
- **New prompt or model candidate:** add `prompts/agent_vN.txt` (or change `APP_MODEL`), run
  `make regression PROMPT=vN`, then compare against the baseline experiment with the pairwise command.
