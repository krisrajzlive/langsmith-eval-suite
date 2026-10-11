PYTHON ?= python
PROMPT ?= v2
CLI = $(PYTHON) -m evalsuite.cli

.PHONY: install test test-live seed benchmark trajectory safety calibrate baseline regression

install:
	$(PYTHON) -m pip install -e ".[dev]"

## Deterministic tests: no network, no credentials.
test:
	$(PYTHON) -m pytest tests -m "not live"

## Real-model CI gates, logged to LangSmith (needs OPENAI_API_KEY and LANGSMITH_API_KEY).
test-live:
	LANGSMITH_TEST_CACHE=.cache/langsmith $(PYTHON) -m pytest tests/live -m live

seed:
	$(CLI) seed

benchmark:
	$(CLI) benchmark --prompt-version $(PROMPT)

trajectory:
	$(CLI) trajectory --prompt-version $(PROMPT)

safety:
	$(CLI) safety --prompt-version $(PROMPT)

calibrate:
	$(CLI) calibrate

## Record per-example scores for the current prompt as the regression baseline.
baseline:
	$(CLI) regression --prompt-version $(PROMPT) --update-baseline

## Run a candidate prompt and fail if it regresses against the baseline.
regression:
	$(CLI) regression --prompt-version $(PROMPT)
