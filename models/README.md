# Adding a model

[English](README.md) · [Русский](README.ru.md)

No model is included. A compatible bundle is supplied separately and
selected with `--model DIRECTORY` or `ThreatAnalyzer.from_bundle()`.
The directory may be outside this repository.

The loader accepts `model-bundle-0.1` with `window_scores_v1` or
`window_scores_v2`, and `model-bundle-0.2` with `window_features_v3`.
Taxonomy version, seven-label order, max-logit aggregation, window settings,
file sizes, SHA-256, license and training provenance must match the runtime
contract. Version 3 also requires `heads.npz` and its focus template.
Only self-contained ONNX graphs are supported; external tensor files are
rejected. A raw training checkpoint requires export before installation.

```bash
personal-threats check-model models/approved/bundle --device cpu
python examples/check_installation.py \
  --model models/approved/bundle
```

`check-model` checks graph compatibility and finite output. It does not
certify detection accuracy. Ordinary analysis rejects `test_fixture`
models unless explicit test-only opt-in is requested.

An operating profile uses `quality-local-report-0.1` and references the
exact bundle manifest hash, evaluation scope, confusion counts, thresholds
and score/label/mask arrays. Keep those arrays beside the report at the
paths in its `files` entries. Loading verifies hashes and recomputes the
actual parent-or-types decisions and precision/FPR constraints.
To verify a supplied profile as well, add
`--profile-report models/approved/profile/report.json` to the installation
check after placing the report and its referenced arrays in that directory.

Model code, weights and datasets retain their respective licenses.
