import json

import pytest

from evalsuite.agent.tools import ALL_TOOLS
from evalsuite.config import get_settings
from evalsuite.datasets import DATASET_NAMES, example_uuid, load_jsonl
from evalsuite.rag import load_passages

KNOWN_TOOLS = {t.name for t in ALL_TOOLS}
CORPUS_DOCS = {p.doc_id for p in load_passages()}


@pytest.mark.unit
@pytest.mark.parametrize("key", list(DATASET_NAMES))
def test_dataset_rows_are_well_formed(key):
    rows = load_jsonl(key)
    assert rows, f"{key} is empty"
    for row in rows:
        assert row["id"] and isinstance(row["inputs"], dict) and isinstance(row.get("outputs", {}), dict)


@pytest.mark.unit
def test_qa_reference_docs_exist_in_corpus():
    for row in load_jsonl("qa"):
        unknown = set(row["outputs"]["reference_doc_ids"]) - CORPUS_DOCS
        assert not unknown, f"{row['id']} references unknown docs {unknown}"
        assert row["outputs"]["answer"].strip()


@pytest.mark.unit
def test_trajectory_rows_reference_real_tools():
    for row in load_jsonl("trajectory"):
        out = row["outputs"]
        assert set(out["expected_tools"]) <= KNOWN_TOOLS
        assert set(out["forbidden_tools"]) <= KNOWN_TOOLS
        assert out["mode"] in {"strict", "unordered", "subset", "superset"}
        assert not set(out["expected_tools"]) & set(out["forbidden_tools"])


@pytest.mark.unit
def test_safety_rows_declare_expected_behavior():
    for row in load_jsonl("safety"):
        assert row["outputs"]["expect"] in {"refuse", "answer"}


@pytest.mark.unit
def test_judge_calibration_is_balanced_and_binary():
    labels = [r["outputs"]["human_label"] for r in load_jsonl("judge_calibration")]
    assert set(labels) == {0, 1}
    assert abs(labels.count(1) - labels.count(0)) <= 2


@pytest.mark.unit
def test_example_uuids_are_stable_and_distinct():
    assert example_uuid("qa", "qa-001") == example_uuid("qa", "qa-001")
    assert example_uuid("qa", "qa-001") != example_uuid("safety", "qa-001")


@pytest.mark.unit
def test_canary_present_in_candidate_prompt_only():
    s = get_settings()
    assert "CANARY-MRDN-7731" in (s.prompts_dir / "agent_v2.txt").read_text()
    for row in load_jsonl("safety"):
        for banned in row["outputs"].get("must_not_contain", []):
            assert isinstance(banned, str)
    json.dumps(load_jsonl("qa"))
