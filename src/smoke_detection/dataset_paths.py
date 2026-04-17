# Paths to the prepared dataset layout (see scripts/prepare_dataset.py).
import os

_CODE_ROOT = os.path.dirname(os.path.abspath(__file__))
# Traverse: smoke_detection/ -> src/ -> repo root -> sibling dataset_prepared/
_WORKSPACE_ROOT = os.path.normpath(os.path.join(_CODE_ROOT, "..", "..", ".."))
DATASET_ROOT = os.environ.get(
    "SMOKEDET_DATA_ROOT",
    os.path.join(_WORKSPACE_ROOT, "dataset_prepared"),
)


def classification_split(name):
    """name: 'train' | 'val' | 'test'"""
    return os.path.join(DATASET_ROOT, "classification", name)


def segmentation_split(name):
    """name: 'train' | 'val' | 'test' -> (images_dir, labels_dir)"""
    base = os.path.join(DATASET_ROOT, "segmentation", name)
    return (
        os.path.join(base, "images"),
        os.path.join(base, "labels"),
    )
