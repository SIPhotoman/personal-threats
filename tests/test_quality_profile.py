# SPDX-License-Identifier: Apache-2.0
"""Check the deployed OR rule without importing any training dependency."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from personal_threats.results import Scores
from personal_threats.review import QualityProfile, ReviewError
from personal_threats.model import LABELS, file_sha256


class QualityProfileTests(unittest.TestCase):
    def test_profile_load_recomputes_arrays_and_rejects_changed_receipts(self):
        import numpy as np

        bundle = SimpleNamespace(
            manifest={
                "purpose": "trained_candidate",
                "training_provenance": {},
            },
            manifest_sha256="b" * 64,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arrays = {
                "scores": np.array([[0.9] * 7, [0.1] * 7]),
                "labels": np.array([[1] * 7, [0] * 7]),
                "known": np.ones((2, 7), dtype=np.uint8),
            }
            files = {}
            for name, array in arrays.items():
                path = root / (name + ".npy")
                np.save(path, array, allow_pickle=False)
                files[name] = {"path": path.name, "sha256": file_sha256(path)}
            counts = {
                "tp": 1,
                "fp": 0,
                "fn": 0,
                "tn": 1,
                "precision": 1.0,
                "recall": 1.0,
                "fpr": 0.0,
                "f1": 1.0,
            }
            report = {
                "schema_version": "quality-local-report-0.1",
                "status": "local_development_evaluated",
                "fixture_mode": False,
                "bundle_manifest_sha256": "b" * 64,
                "development_messages": 2,
                "files": files,
                "profiles": {
                    "balanced": {
                        "all_labels_feasible": True,
                        "presence_rule": "parent_or_types",
                        "thresholds": dict.fromkeys(LABELS, 0.5),
                        "metrics": {
                            **{label: dict(counts) for label in LABELS},
                            "macro_f1": 1.0,
                        },
                    }
                },
            }
            path = root / "report.json"
            path.write_text(json.dumps(report))
            with patch(
                "personal_threats.review.ModelBundle.load", return_value=bundle
            ):
                loaded = QualityProfile.load(path, root / "bundle")
                self.assertTrue(
                    loaded.apply(
                        Scores(0.9, (0.9,) * 6), model_sha256="b" * 64
                    )["threat_present"]
                )
                self.assertFalse(
                    loaded.apply(
                        Scores(0.1, (0.1,) * 6), model_sha256="b" * 64
                    )["threat_present"]
                )
                changed = arrays["scores"].copy()
                changed[1, 0] = 0.9
                np.save(root / "scores.npy", changed, allow_pickle=False)
                with self.assertRaisesRegex(ReviewError, "array changed"):
                    QualityProfile.load(path, root / "bundle")
                report["files"]["scores"]["sha256"] = file_sha256(
                    root / "scores.npy"
                )
                path.write_text(json.dumps(report))
                with self.assertRaisesRegex(ReviewError, "union predictions"):
                    QualityProfile.load(path, root / "bundle")

    def test_profile_cannot_use_legacy_targets_for_a_scoped_model(self):
        bundle = SimpleNamespace(
            manifest={
                "purpose": "trained_candidate",
                "training_provenance": {"evaluation_view_sha256": "a" * 64},
            },
            manifest_sha256="b" * 64,
        )
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            report = {
                "schema_version": "quality-local-report-0.1",
                "status": "local_development_evaluated",
                "fixture_mode": False,
                "bundle_manifest_sha256": "b" * 64,
            }
            with patch(
                "personal_threats.review.ModelBundle.load",
                return_value=bundle,
            ):
                for extra in (
                    {},
                    {
                        "evaluation_view_sha256": "c" * 64,
                        "evaluation_scope": "self_actor_victim_eyewitness",
                    },
                    {
                        "evaluation_view_sha256": "a" * 64,
                        "evaluation_scope": "legacy",
                    },
                ):
                    with self.subTest(extra=extra):
                        path = root / "report.json"
                        path.write_text(json.dumps({**report, **extra}))
                        with self.assertRaisesRegex(
                            ReviewError, "different evaluation scope"
                        ):
                            QualityProfile.load(path, root / "bundle")

    def test_type_can_trigger_presence_when_parent_is_disabled(self):
        profile = QualityProfile(
            "balanced", "a" * 64, (None, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
        )
        result = profile.apply(
            Scores(0.1, (0.8, 0.1, 0.1, 0.1, 0.1, 0.1)), model_sha256="a" * 64
        )
        self.assertTrue(result["threat_present"])
        self.assertEqual(result["types"], ["physical_harm"])
        self.assertFalse(result["event_extraction_available"])
        self.assertFalse(result["production_ready_detector"])

    def test_profile_is_bound_to_model_and_unknown_type_is_explicit(self):
        profile = QualityProfile(
            "balanced", "a" * 64, (0.5, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9)
        )
        scores = Scores(0.8, (0.1,) * 6)
        self.assertTrue(
            profile.apply(scores, model_sha256="a" * 64)["type_uncertain"]
        )
        with self.assertRaises(ReviewError):
            profile.apply(scores, model_sha256="b" * 64)


if __name__ == "__main__":
    unittest.main()
