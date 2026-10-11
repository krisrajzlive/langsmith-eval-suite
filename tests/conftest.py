import os

import pytest


def pytest_collection_modifyitems(config, items):
    """Unit tests run everywhere; live tests need provider and LangSmith credentials."""
    required = ("OPENAI_API_KEY", "LANGSMITH_API_KEY")
    missing = [k for k in required if not os.environ.get(k)]
    if not missing:
        return
    skip_live = pytest.mark.skip(reason=f"live tests require {', '.join(missing)}")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)
