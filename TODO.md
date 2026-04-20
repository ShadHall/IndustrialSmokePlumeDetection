# TODO

Milestone-ordered roadmap for the Industrial Smoke Plume Detection repo.
This repo develops and validates the ResNet-50 classifier and U-Net segmenter.
Trained models are exported as ONNX files, consumed by the flight-software
Python package, and repackaged in Rust for HIL testing.

**Dependency order:** `v0.3` (train + validate) must complete before `v0.4`
(export) can begin. `v0.5` items can run in parallel with `v0.4` but none
block the flight-software handoff.

---

## v0.3.0 — Train & Validate

> Gates all downstream work. No export is meaningful without verified weights.

- [ ] Train classifier (ResNet-50) on the Zenodo dataset
- [ ] Train segmenter (U-Net) on the Zenodo dataset
- [ ] Verify paper-accuracy parity post-Lightning port — classification
      accuracy/AUC and segmentation IoU against Table 1 of Mommert et al. 2020
      (flagged as unverified in CHANGELOG 0.2.0)
- [ ] Add paper-replication experiment configs (`configs/classification/paper.yaml`,
      `configs/segmentation/paper.yaml`) matching the published hyperparameters
- [ ] Port training figure generation to Lightning (callback + end-of-eval
      artifacts) — the Apr-16 design targeted the old argparse scripts and was
      never carried forward
- [ ] Add results notebook (`notebooks/results.ipynb`) — predictions on
      validation images, confusion matrix, IoU and area-ratio distributions

---

## v0.4.0 — Export & Integration Interface

> Produces the versioned ONNX artifacts consumed by the flight software repo.

- [ ] Add ONNX export script (`scripts/export_models.py`) for both classifier
      and segmenter — ONNX is the correct target format for `ort`-based Rust
      inference
- [ ] Add export validation test — assert ONNX output matches PyTorch output
      within `atol=1e-5` for both models; catches silent precision loss at the
      graph boundary
- [ ] Write input/output spec (`docs/INFERENCE_SPEC.md`) — exact preprocessing
      contract for the Rust consumer: channel order, normalization stats,
      expected input shape, classifier threshold, segmenter output thresholding
      logic
- [ ] Add batch-1 inference latency benchmark — CPU baseline before the Rust
      port; real-time gimbal control is latency-sensitive
- [ ] Add GitHub Releases workflow (`.github/workflows/release.yml`) — triggered
      on `v*` tags; exports both models, uploads `.onnx` files as release
      artifacts alongside the wheel

---

## v0.5.0 — Documentation & Polish

> Capstone report support and long-term maintainability. None of these block
> the flight-software integration.

- [ ] Flesh out `docs/MODEL_ARCHITECTURE.md` — currently a stub; needed for the
      capstone report and flight-repo developers
- [ ] Add mypy type checking — deferred during the 0.2.0 cleanup; important
      before the public API is consumed by another repo
- [ ] Add FP16 export option to the export script — minimum quantization step
      for edge deployment; validate that ONNX FP16 output stays within
      acceptable tolerance relative to FP32
- [ ] Add Docker / devcontainer — reproducible environment for capstone graders
      and future collaborators
