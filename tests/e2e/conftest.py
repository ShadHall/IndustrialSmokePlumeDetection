"""Fixtures for e2e tests — YAML files on disk for CLI invocation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from smoke_detection.configs.classification import ClassificationConfig
from smoke_detection.configs.segmentation import SegmentationConfig


def _dump_config(cfg, path: Path) -> Path:
    path.write_text(yaml.safe_dump(cfg.model_dump(mode="json")))
    return path


@pytest.fixture
def classification_yaml_tmp(
    sample_classification_config: ClassificationConfig, tmp_path: Path
) -> Path:
    return _dump_config(sample_classification_config, tmp_path / "cls.yaml")


@pytest.fixture
def segmentation_yaml_tmp(sample_segmentation_config: SegmentationConfig, tmp_path: Path) -> Path:
    return _dump_config(sample_segmentation_config, tmp_path / "seg.yaml")
