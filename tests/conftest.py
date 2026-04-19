"""Pytest fixtures shared across the test suite.

Intentionally minimal for the initial scaffolding. Add fixtures here as tests
are introduced (e.g. a tiny synthetic dataset fixture for DataModule tests).
"""

import pytest


@pytest.fixture
def tiny_fake_dataset():
    """Placeholder fixture. Replace with a real synthetic dataset factory when
    test coverage is added."""
    return None
