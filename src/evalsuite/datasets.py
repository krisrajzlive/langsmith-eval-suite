"""Dataset loading and idempotent seeding into LangSmith."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from .config import get_settings

DATASET_NAMES = {
    "qa": "meridian-docs-qa",
    "trajectory": "meridian-agent-trajectories",
    "safety": "meridian-safety-robustness",
    "judge_calibration": "meridian-judge-calibration",
    "ops_tasks": "meridian-ops-tasks",
}
_NAMESPACE = uuid.UUID("5d0f3c1e-8a4b-4f2e-9b1a-0c6e2f7a9d11")


def load_jsonl(name: str, datasets_dir: Path | None = None) -> list[dict[str, Any]]:
    path = (datasets_dir or get_settings().datasets_dir) / f"{name}.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path.name}: duplicate example ids")
    return rows


def example_uuid(dataset_key: str, source_id: str) -> uuid.UUID:
    """Stable UUID so re-seeding updates examples in place instead of duplicating them."""
    return uuid.uuid5(_NAMESPACE, f"{dataset_key}:{source_id}")


def seed_dataset(dataset_key: str, client=None) -> str:
    """Create the dataset if needed and upsert every example. Returns the dataset name."""
    from langsmith import Client

    client = client or Client()
    name = DATASET_NAMES[dataset_key]
    if not client.has_dataset(dataset_name=name):
        client.create_dataset(dataset_name=name, description=f"Offline evaluation set: {dataset_key}")
    examples = [
        {
            "id": example_uuid(dataset_key, row["id"]),
            "inputs": row["inputs"],
            "outputs": row.get("outputs", {}),
            "metadata": {**row.get("metadata", {}), "source_id": row["id"]},
        }
        for row in load_jsonl(dataset_key)
    ]
    client.create_examples(dataset_name=name, examples=examples)
    return name
