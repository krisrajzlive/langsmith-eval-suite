# langsmith-eval-suite

Offline evaluation suite for a documentation-grounded support agent, built on
[LangSmith](https://docs.langchain.com/langsmith/evaluation), LangGraph, `openevals` and `agentevals`.

The system under test is a LangGraph agent that answers questions about a payments API from a versioned
documentation corpus. It retrieves passages, looks up error codes, computes fees and can open a support
ticket. Every evaluation runs **before deployment** against curated datasets with reference outputs;
there is no dependency on production traces.

## Evaluation coverage

| Type | What it verifies | Scoring | Entry point |
| --- | --- | --- | --- |
| Unit tests | Format, citation discipline, latency/token budgets, tool math, dataset integrity | Code evaluators, `pytest` | `make test` |
| Correctness benchmark | Answer agrees with the reference answer | LLM judge, reference-based | `make benchmark` |
| RAG quality | Groundedness, helpfulness, retrieval relevance | LLM judges, reference-free | `make benchmark` |
| Retrieval metrics | Recall, precision, reciprocal rank; experiment-level MRR and hit rate | Code + summary evaluator | `make benchmark` |
| Agent trajectory | Correct tools, no forbidden tools, bounded tool calls, trajectory quality | `agentevals` match modes + LLM judge | `make trajectory` |
| Safety and robustness | Prompt-leak canary, injected instructions, secret/PII handling, scope refusal | Code checks + LLM judge | `make safety` |
| Regression gate | A candidate prompt/model must not break cases the baseline passes | Per-example comparison to a stored baseline | `make regression` |
| Pairwise comparison | Version A vs version B on answer quality | Position-randomized pairwise judge, two rubrics | `python -m evalsuite.cli pairwise A B --rubric accuracy` ([results](docs/pairwise-evaluation.md)) |
| Composite score | One weighted quality number for gating and dashboards | Weighted feedback aggregation | `python -m evalsuite.cli composite EXPERIMENT` |
| Judge calibration | Whether the LLM judge agrees with human labels | Accuracy, TPR/TNR, Cohen's kappa | `make calibrate` |

Prompt-injection defenses (an input guard and an escalation policy) and their measured effect are described in
[docs/guard-evaluation.md](docs/guard-evaluation.md).

See [docs/eval-strategy.md](docs/eval-strategy.md) for the rationale, thresholds and how to extend each type.

## Layout

```
src/evalsuite/
  agent/        LangGraph agent, tools, normalized result record
  rag/          corpus loading; offline TF-IDF and FAISS dense retrievers
  evaluators/   code, retrieval, trajectory, safety, judges, composite
  datasets.py   JSONL loading and idempotent LangSmith seeding
  runners.py    one runner per evaluation type
  regression.py baseline storage and the regression gate
  calibration.py judge-vs-human statistics
  cli.py        python -m evalsuite.cli <command>
data/corpus/    documentation corpus (markdown)
datasets/       versioned evaluation sets (JSONL)
prompts/        versioned agent prompts (agent_v1, agent_v2)
baselines/      committed per-example baseline scores
tests/          deterministic tests; tests/live for real-model CI gates
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env                                # then fill in your own keys
```

Providers are selected through environment variables (see `.env.example`): the application model, the judge
model and the embeddings can each use OpenAI, Ollama (hosted API), Hugging Face or AWS Bedrock. Use a judge from a
different model family than the application. The default embeddings are an offline TF-IDF retriever, so
retrieval evals need no embedding quota.

## Running

```bash
make test           # deterministic tests, no credentials needed
make seed           # upsert datasets into LangSmith
make benchmark      # correctness, RAG quality, retrieval, format
make trajectory
make safety
make calibrate      # run this before trusting judge scores
make baseline       # record the baseline for the current prompt
make regression PROMPT=v1   # gate a candidate against the baseline
```

Without `make`, use `python -m evalsuite.cli <command>`; see `--help`.

### Agents without retrieval

A second agent, an order-operations assistant with no documents, is evaluated on the state it leaves behind:
task success against an expected end state, unintended side effects, truthfulness of its own reports, tool and
trajectory discipline, clarification, fault recovery, and consistency across repetitions (pass^k). See
[docs/agent-evaluation.md](docs/agent-evaluation.md). Run it with `python -m evalsuite.cli ops --repetitions 3`.

### Comparing application models

```bash
python -m evalsuite.cli calibrate --judge bedrock:mistral.mistral-large-3-675b-instruct   # vet a judge first
python -m evalsuite.cli matrix \
  --models "bedrock:amazon.nova-lite-v1:0,bedrock:amazon.nova-pro-v1:0,openai:gpt-4o-mini" \
  --prompt-version v2
```

`matrix` runs the benchmark, trajectory and safety suites for every model with the judge held fixed, records each
run as a LangSmith experiment, and writes [docs/model-comparison.md](docs/model-comparison.md). A model that cannot
complete a suite is reported as a failure instead of aborting the run. Set `EVAL_TRACE_EVALUATORS=false` on free
LangSmith tiers to cut trace volume. Keep the judge in a different model family from the model being scored, and
treat gaps under about 0.1 as noise unless you raise `EVAL_NUM_REPETITIONS`.

Free-tier providers are rate limited. Concurrency and repetitions are controlled by `EVAL_MAX_CONCURRENCY` and
`EVAL_NUM_REPETITIONS`, and `make test-live` caches model responses between runs.

## Datasets

| File | Examples | Purpose |
| --- | --- | --- |
| `qa.jsonl` | 24 | Reference answers and reference documents; includes multi-hop, calculation and unanswerable cases |
| `trajectory.jsonl` | 10 | Expected tools, match mode and forbidden tools per request |
| `safety.jsonl` | 15 | Leak, injection, secret, PII and scope probes plus a benign control |
| `judge_calibration.jsonl` | 16 | Balanced correct/incorrect answers with human labels |

Examples carry stable IDs, so re-seeding updates in place. Change a dataset by editing the JSONL and running
`make seed`; the dataset integrity tests run in CI.
