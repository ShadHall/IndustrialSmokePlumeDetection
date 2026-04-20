"""Tests for classification plotting helpers."""

from __future__ import annotations

from pathlib import Path

from smoke_detection.evaluation.classification_metrics import (
    plot_confusion_matrix,
    plot_roc_curve,
)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _is_png(path: Path) -> bool:
    return path.read_bytes()[:8] == PNG_MAGIC


def test_plot_confusion_matrix_writes_png(tmp_path):
    out = tmp_path / "cm.png"
    plot_confusion_matrix(tp=3, tn=5, fp=1, fn=2, out_path=out)
    assert out.exists() and _is_png(out)


def test_plot_roc_curve_writes_png(tmp_path):
    out = tmp_path / "roc.png"
    plot_roc_curve([0.1, 0.9, 0.4, 0.8], [0, 1, 0, 1], out)
    assert out.exists() and _is_png(out)


def test_plot_roc_curve_handles_single_class(tmp_path):
    """When labels are all one class, sklearn raises ValueError which the
    function catches and reports as auc=nan; we only assert no raise + file exists."""
    out = tmp_path / "roc_single.png"
    plot_roc_curve([0.1, 0.2, 0.3, 0.4], [0, 0, 0, 0], out)
    assert out.exists() and _is_png(out)
