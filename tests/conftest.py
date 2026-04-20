"""Top-level pytest fixtures.

Provides determinism, path helpers, a session-scoped synthetic prepared-dataset
tree (13-band uint16 GeoTIFFs + Label-Studio-style JSON polygons), and a
cached-weights fixture for the one slow test that loads ResNet-50 ImageNet
weights. Also auto-skips `gpu` tests when CUDA is unavailable.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests._data import build_synthetic_prepared_tree

REPO_ROOT = Path(__file__).resolve().parents[1]


def pytest_configure(config: pytest.Config) -> None:
    os.environ.setdefault("PYTORCH_LIGHTNING_LOG_LEVEL", "ERROR")
    # Lightning writes TB logs to CWD by default; tests redirect via cfg.paths.output_dir.


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Auto-skip `gpu`-marked tests when CUDA is unavailable."""
    import torch

    if torch.cuda.is_available():
        return
    skip_gpu = pytest.mark.skip(reason="CUDA not available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)


@pytest.fixture(autouse=True)
def _deterministic() -> None:
    """Seed all RNGs before every test."""
    from smoke_detection.common.seed import seed_everything

    seed_everything(1234, deterministic=True)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def synthetic_dataset_root(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Read-only session-scoped synthetic prepared-dataset tree."""
    root = tmp_path_factory.mktemp("smokedet_synthetic_prepared")
    build_synthetic_prepared_tree(root)
    prev = os.environ.get("SMOKEDET_DATA_ROOT")
    os.environ["SMOKEDET_DATA_ROOT"] = str(root)
    try:
        yield root
    finally:
        if prev is None:
            os.environ.pop("SMOKEDET_DATA_ROOT", None)
        else:
            os.environ["SMOKEDET_DATA_ROOT"] = prev
        shutil.rmtree(root, ignore_errors=True)


@pytest.fixture
def mutable_dataset_root(tmp_path: Path) -> Path:
    """Function-scoped writable copy of the synthetic prepared tree."""
    root = tmp_path / "mutable_prepared"
    build_synthetic_prepared_tree(root)
    return root


@pytest.fixture
def tiny_classification_batch():
    """2-sample classification batch (pre-transform-compatible)."""
    import torch

    return {
        "idx": torch.tensor([0, 1]),
        "img": torch.randn(2, 4, 90, 90),
        "lbl": torch.tensor([True, False]),
        "imgfile": ["a.tif", "b.tif"],
    }


@pytest.fixture
def tiny_segmentation_batch():
    """2-sample segmentation batch (pre-transform-compatible)."""
    import torch

    mask = torch.zeros(2, 90, 90)
    mask[0, 20:50, 30:70] = 1.0  # one positive, one all-zero
    return {
        "idx": torch.tensor([0, 1]),
        "img": torch.randn(2, 4, 90, 90),
        "fpt": mask,
        "imgfile": ["a.tif", "b.tif"],
    }


@pytest.fixture(scope="session")
def cached_resnet50_weights():
    """Return IMAGENET1K_V1 weights enum; trigger download on first call."""
    from torchvision.models import ResNet50_Weights, resnet50

    _ = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    return ResNet50_Weights.IMAGENET1K_V1
