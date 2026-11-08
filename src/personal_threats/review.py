# SPDX-License-Identifier: Apache-2.0
"""Read-only pilot intake and independent operating-point verification."""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import math
from pathlib import Path
import re
from types import MappingProxyType

from .core import strict_json
from .model import LABELS, ModelBundle, ModelError, file_sha256
from .results import probability


LIMITS = {"strict": 0.001, "balanced": 0.005, "high_recall": 0.02}


class ReviewError(ModelError):
    """A returned run is inconsistent, incomplete or not eligible for use."""


def interior_thresholds(scores, thresholds):
    """Move cutoffs inside score gaps without changing calibration decisions.

    A cutoff at an observed positive score can flip after a tiny numerical
    perturbation. The midpoint between the nearest excluded and included
    scores has the same empirical decisions and a larger margin. Adjacent
    floats may have no interior value; in that case retain the upper value.
    This does not guarantee robustness to changes larger than the gap.
    """
    import numpy as np

    values = np.asarray(scores, dtype=np.float64)
    if (
        values.ndim != 2
        or not len(values)
        or values.shape[1] != len(thresholds)
        or not np.isfinite(values).all()
        or np.any((values < 0) | (values > 1))
    ):
        raise ReviewError("Invalid calibration score matrix")
    result = []
    for head, threshold in enumerate(thresholds):
        if threshold is None:
            result.append(None)
            continue
        threshold = probability(threshold)
        column = values[:, head]
        included = column >= threshold
        if not included.any():
            result.append(threshold)
            continue
        upper = float(column[included].min())
        lower = float(column[~included].max()) if (~included).any() else 0.0
        midpoint = lower + (upper - lower) / 2
        result.append(midpoint if lower < midpoint <= upper else upper)
    return result


def _check(condition, detail):
    if not condition:
        raise ReviewError(detail)


def _read(path, *, maximum=8 * 1024 * 1024):
    with path.open("rb") as stream:
        content = stream.read(maximum + 1)
    _check(len(content) <= maximum, "Pilot JSON exceeds its size limit")
    value = strict_json(content)
    _check(isinstance(value, dict), "Pilot JSON must be an object")
    return value


def _inside(root, name):
    _check(isinstance(name, str) and name, "Invalid pilot artifact path")
    path = root / name
    _check(
        not Path(name).is_absolute()
        and ".." not in Path(name).parts
        and path.resolve().is_relative_to(root.resolve()),
        "Pilot artifact path escapes its directory",
    )
    return path


def _count(value, name):
    _check(type(value) is int and value >= 0, "Invalid count: " + name)
    return value


def verify_receipt(root, required):
    """Verify the committed artifact set, including separate profile files."""
    receipt = _read(root / "receipt.json")
    pins = receipt.get("files")
    _check(
        isinstance(pins, dict) and required <= set(pins),
        "Artifact receipt omits required candidate files",
    )
    for name, digest in pins.items():
        path = _inside(root, name)
        _check(
            isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest),
            "Invalid artifact receipt SHA-256",
        )
        _check(
            file_sha256(path) == digest,
            "Artifact receipt SHA-256 mismatch: " + name,
        )


def _equal(actual, expected, name):
    if expected is None:
        _check(actual is None, "Undefined metric must remain null: " + name)
    else:
        probability(actual)
        _check(
            math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12),
            "Metric differs from its confusion counts: " + name,
        )


def verify_metrics(metrics, count):
    """Recompute metrics; never trust a supplied precision/FPR/F1 flag."""
    _check(
        isinstance(metrics, dict)
        and set(metrics) == set(LABELS) | {"macro_f1"},
        "Metrics do not match the seven-label taxonomy",
    )
    result = {}
    for label in LABELS:
        row = metrics[label]
        _check(
            isinstance(row, dict)
            and set(row)
            == {"tp", "fp", "fn", "tn", "precision", "recall", "fpr", "f1"},
            "Invalid confusion matrix: " + label,
        )
        tp, fp, fn, tn = [_count(row[k], k) for k in ("tp", "fp", "fn", "tn")]
        _check(tp + fp + fn + tn <= count, "Support exceeds development size")
        computed = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "fpr": fp / (fp + tn) if fp + tn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        }
        for metric in ("precision", "recall", "fpr", "f1"):
            _equal(row[metric], computed[metric], label + ":" + metric)
        result[label] = computed
    result["macro_f1"] = sum(result[k]["f1"] for k in LABELS[1:]) / 6
    _equal(metrics["macro_f1"], result["macro_f1"], "macro_f1")
    binary = result[LABELS[0]]
    for label in LABELS[1:]:
        row = result[label]
        _check(
            row["tp"] + row["fp"] <= binary["tp"] + binary["fp"],
            "Type predictions are inconsistent with binary gating",
        )
    return result


def verify_profiles(profiles, count):
    _check(
        isinstance(profiles, dict) and set(profiles) == set(LIMITS),
        "All three operating profiles are required",
    )
    supports = None
    for name, cap in LIMITS.items():
        item = profiles[name]
        _check(isinstance(item, dict), "Invalid operating profile")
        _check(
            item.get("calibration_split") == "development",
            "Threshold selection must use development only",
        )
        _equal(item.get("max_fpr"), cap, "profile FPR limit")
        _equal(item.get("min_precision"), 0.90, "profile precision limit")
        for key in ("thresholds", "operating_points"):
            _check(
                isinstance(item.get(key), dict)
                and set(item[key]) == set(LABELS),
                "Operating points do not match the label taxonomy",
            )
        metrics = verify_metrics(item.get("metrics"), count)
        observed = []
        feasible = []
        for label in LABELS:
            point = item["operating_points"][label]
            threshold = item["thresholds"][label]
            _check(
                isinstance(point, dict)
                and type(point.get("feasible")) is bool,
                "Operating-point feasibility must be boolean",
            )
            row = metrics[label]
            positive, negative = row["tp"] + row["fn"], row["fp"] + row["tn"]
            _check(
                _count(point.get("positives"), "positives") == positive
                and _count(point.get("negatives"), "negatives") == negative,
                "Operating-point support differs from metric support",
            )
            observed.append((positive, negative))
            feasible.append(point["feasible"])
            if point["feasible"]:
                probability(threshold)
                _equal(point.get("threshold"), threshold, "threshold")
                _check(
                    positive > 0 and negative > 0 and row["tp"] > 0,
                    "A feasible point requires nonempty class support",
                )
                _check(
                    point.get("tp") == row["tp"]
                    and point.get("fp") == row["fp"],
                    "Operating-point counts differ from the metric counts",
                )
                for key in ("precision", "recall", "fpr"):
                    _equal(point.get(key), row[key], "operating-point " + key)
                _check(
                    row["precision"] >= 0.90 and row["fpr"] <= cap,
                    "A feasible point violates its absolute requirements",
                )
            else:
                _check(
                    threshold is None and point.get("threshold") is None,
                    "An infeasible point cannot silently select a threshold",
                )
                _check(
                    row["tp"] == 0 and row["fp"] == 0,
                    "An unavailable threshold cannot produce predictions",
                )
        _check(
            type(item.get("all_labels_feasible")) is bool
            and item["all_labels_feasible"] == all(feasible),
            "Global profile feasibility contradicts individual labels",
        )
        _check(
            supports is None or observed == supports,
            "Profile metrics use different class supports",
        )
        supports = observed


def verify_quantization(profiles, quantized, count):
    """Independently check fixed-threshold drift and absolute requirements."""
    _check(
        isinstance(quantized, dict) and set(quantized) == set(LIMITS),
        "Quantization results must include all three profiles",
    )
    for name, cap in LIMITS.items():
        reference = profiles[name]
        item = quantized[name]
        _check(isinstance(item, dict), "Invalid quantization result")
        metrics = verify_metrics(item.get("metrics"), count)
        failures = []
        if not reference["all_labels_feasible"]:
            failures.append("fp32_operating_points_infeasible")
        if (
            reference["metrics"]["macro_f1"] - metrics["macro_f1"]
            > 0.010000001
        ):
            failures.append("macro_f1_drop_exceeds_1pp")
        for label in LABELS:
            left, right = reference["metrics"][label], metrics[label]
            _check(
                left["tp"] + left["fn"] == right["tp"] + right["fn"]
                and left["fp"] + left["tn"] == right["fp"] + right["tn"],
                "FP32 and int8 class supports differ",
            )
            if left["recall"] is None or right["recall"] is None:
                failures.append(label + ":insufficient_support")
            elif left["recall"] - right["recall"] > 0.020000001:
                failures.append(label + ":recall_drop_exceeds_2pp")
            if right["precision"] is None or right["precision"] < 0.90:
                failures.append(label + ":precision_below_0.90")
            if right["fpr"] is None or right["fpr"] > cap:
                failures.append(label + ":fpr_exceeds_profile_limit")
        stated = item.get("failures")
        _check(
            isinstance(stated, list)
            and all(isinstance(s, str) for s in stated)
            and len(stated) == len(set(stated))
            and set(stated) == set(failures),
            "Quantization failure reasons contradict recomputed metrics",
        )
        _check(
            type(item.get("passed")) is bool
            and item["passed"] == (not failures),
            "Quantization acceptance contradicts recomputed metrics",
        )


@dataclass(frozen=True, slots=True)
class OperatingProfile:
    """Verified message/type thresholds bound to one model manifest."""

    name: str
    variant: str
    model_sha256: str
    thresholds: tuple[float, ...]

    def __post_init__(self):
        _check(
            self.name in LIMITS and self.variant in ("fp32", "int8"),
            "Unknown operating profile or model variant",
        )
        _check(
            isinstance(self.model_sha256, str)
            and re.fullmatch(r"[0-9a-f]{64}", self.model_sha256),
            "Invalid profile model SHA-256",
        )
        values = tuple(probability(v) for v in self.thresholds)
        _check(len(values) == len(LABELS), "Expected seven thresholds")
        object.__setattr__(self, "thresholds", values)

    def apply(self, scores, *, model_sha256):
        """Apply an operating point without inventing event extraction."""
        from .results import Scores

        _check(
            isinstance(scores, Scores), "Expected validated seven-head scores"
        )
        _check(
            model_sha256 == self.model_sha256,
            "Operating profile belongs to a different model manifest",
        )
        values = (scores.binary, *scores.types)
        present = values[0] >= self.thresholds[0]
        types = [
            label
            for label, value, threshold in zip(
                LABELS[1:], values[1:], self.thresholds[1:]
            )
            if present and value >= threshold
        ]
        return {
            "threat_present": present,
            "types": types,
            "profile": self.name,
            "variant": self.variant,
            "model_sha256": self.model_sha256,
            "threshold_selection_split": "development",
            "probability_calibration_verified": False,
            "event_extraction_available": False,
            "recipient_extraction_available": False,
            "type_uncertain": present and not types,
            "production_ready_detector": False,
        }


@dataclass(frozen=True, slots=True)
class QualityProfile:
    """Model-bound OR thresholds from the new local development evaluation."""

    name: str
    model_sha256: str
    thresholds: tuple[float | None, ...]

    def __post_init__(self):
        _check(self.name in LIMITS, "Unknown quality profile")
        _check(
            isinstance(self.model_sha256, str)
            and re.fullmatch(r"[a-f0-9]{64}", self.model_sha256),
            "Invalid model SHA-256",
        )
        values = tuple(
            None if value is None else probability(value)
            for value in self.thresholds
        )
        _check(len(values) == 7, "Expected seven quality thresholds")
        object.__setattr__(self, "thresholds", values)

    @classmethod
    def load(cls, report_path, bundle_directory, name="balanced"):
        report = _read(Path(report_path))
        bundle = ModelBundle.load(bundle_directory)
        _check(
            bundle.manifest["purpose"] == "trained_candidate",
            "Test fixtures cannot supply quality operating points",
        )
        _check(
            report.get("schema_version") == "quality-local-report-0.1"
            and report.get("status") == "local_development_evaluated"
            and not report.get("fixture_mode"),
            "Local quality evaluation is incomplete",
        )
        _check(
            bundle.manifest_sha256 == report["bundle_manifest_sha256"],
            "Quality report belongs to another bundle",
        )
        expected_view = bundle.manifest.get("training_provenance", {}).get(
            "evaluation_view_sha256"
        )
        if expected_view is not None:
            _check(
                expected_view == report.get("evaluation_view_sha256")
                and report.get("evaluation_scope")
                == "self_actor_victim_eyewitness",
                "Quality profile uses a different evaluation scope",
            )
        _check(name in LIMITS, "Unknown quality profile")
        profile = report["profiles"][name]
        _check(
            profile["all_labels_feasible"]
            and profile["presence_rule"] == "parent_or_types",
            "Requested joint operating point is infeasible",
        )
        metrics = verify_metrics(
            profile["metrics"], report["development_messages"]
        )
        import numpy as np

        arrays = {}
        for role in ("scores", "labels", "known"):
            entry = report["files"][role]
            path = _inside(Path(report_path).parent, entry["path"])
            _check(
                file_sha256(path) == entry["sha256"],
                "Local quality evaluation array changed",
            )
            arrays[role] = np.load(path, allow_pickle=False)
        shape = (report["development_messages"], 7)
        _check(
            all(a.shape == shape for a in arrays.values()),
            "Local quality arrays have inconsistent shapes",
        )
        _check(
            np.isfinite(arrays["scores"]).all()
            and ((arrays["scores"] >= 0) & (arrays["scores"] <= 1)).all()
            and np.isin(arrays["labels"], [0, 1]).all()
            and np.isin(arrays["known"], [0, 1]).all(),
            "Invalid quality arrays",
        )
        predicted = np.zeros(shape, dtype=bool)
        for head, label in enumerate(LABELS):
            threshold = profile["thresholds"][label]
            if threshold is not None:
                predicted[:, head] = arrays["scores"][:, head] >= probability(
                    threshold
                )
        predicted[:, 0] |= predicted[:, 1:].any(1)
        for head, label in enumerate(LABELS):
            truth = arrays["labels"][:, head].astype(bool)
            mask = arrays["known"][:, head].astype(bool)
            guess = predicted[:, head]
            counts = {
                "tp": int((guess & truth & mask).sum()),
                "fp": int((guess & ~truth & mask).sum()),
                "fn": int((~guess & truth & mask).sum()),
                "tn": int((~guess & ~truth & mask).sum()),
            }
            _check(
                all(metrics[label][k] == v for k, v in counts.items()),
                "Operating point differs from actual union predictions",
            )
        _check(
            all(
                metrics[n]["precision"] is not None
                and metrics[n]["precision"] >= 0.9
                and metrics[n]["fpr"] is not None
                and metrics[n]["fpr"] <= LIMITS[name]
                for n in LABELS
            ),
            "Joint profile failed absolute precision/FPR limits",
        )
        return cls(
            name,
            bundle.manifest_sha256,
            tuple(profile["thresholds"][n] for n in LABELS),
        )

    def apply(self, scores, *, model_sha256):
        from .results import Scores

        _check(isinstance(scores, Scores), "Expected validated scores")
        _check(
            model_sha256 == self.model_sha256,
            "Quality profile belongs to another model",
        )
        values = (scores.binary, *scores.types)
        decisions = [
            threshold is not None and value >= threshold
            for value, threshold in zip(values, self.thresholds)
        ]
        present = any(decisions)
        types = [name for name, flag in zip(LABELS[1:], decisions[1:]) if flag]
        return {
            "threat_present": present,
            "types": types,
            "type_uncertain": present and not types,
            "profile": self.name,
            "model_sha256": self.model_sha256,
            "presence_rule": "parent_or_types",
            "threshold_selection_split": "development",
            "event_extraction_available": False,
            "production_ready_detector": False,
            "probability_calibration_verified": False,
        }


class PilotReview:
    """Validate returned artifacts without reading train or test text."""

    def __init__(
        self,
        directory,
        *,
        expected_dataset_sha256=None,
        allow_test_fixture=False,
        probe=False,
    ):
        self.directory = Path(directory)
        self._report = _read(self.directory / "report.json")
        self._report_sha256 = file_sha256(self.directory / "report.json")
        self._bundles = {}
        self.probes = {}
        self.expected_dataset_verified = False
        report = self.report
        _check(
            report.get("schema_version") == "pilot-run-0.1",
            "Expected a pilot worker report, not a bootstrap report",
        )
        status = report.get("status")
        _check(
            status
            in {
                "complete",
                "failed",
                "budget_exhausted",
                "checkpointed_interruption",
                "exported_pending_evaluation",
            },
            "Unknown pilot completion status",
        )
        self.fixture = report.get("fixture_only")
        _check(type(self.fixture) is bool, "Missing fixture provenance")
        _check(
            not self.fixture or allow_test_fixture,
            "Test fixtures cannot be accepted as trained pilot results",
        )
        _check(
            report.get("test_split_accessed") is False,
            "The pilot must keep the frozen test split untouched",
        )
        self.complete = status == "complete"
        if not self.complete:
            return
        _check(
            report.get("full_development_evaluated") is True,
            "The complete development split must be evaluated",
        )
        count = _count(
            report.get("development_examples"), "development_examples"
        )
        _check(
            count > 0 and (self.fixture or count == 10000),
            "Expected all 10000 development messages",
        )
        state = report.get("training_state", {})
        _check(
            _count(state.get("steps"), "training steps") > 0,
            "A pilot candidate requires optimizer updates",
        )
        cache = report.get("cache", {})
        counts = cache.get("counts", {})
        train = _count(counts.get("train"), "train messages")
        _check(
            train > 0 and counts.get("development") == count,
            "Training cache and evaluation sizes differ",
        )
        _check(
            self.fixture or train == 80000, "Expected all 80000 train messages"
        )
        epoch = _count(state.get("epoch"), "epoch")
        offset = _count(state.get("offset"), "offset")
        seen = _count(state.get("examples_seen"), "examples_seen")
        _check(
            epoch <= 3
            and offset < train
            and seen <= 3 * train
            and seen == epoch * train + offset
            and state["steps"] <= seen,
            "Training cursor contradicts completed examples/epochs",
        )
        fingerprint = report.get("run_fingerprint")
        _check(
            isinstance(fingerprint, str)
            and re.fullmatch(r"[0-9a-f]{64}", fingerprint),
            "Invalid run fingerprint",
        )
        paths = report.get("model_artifacts")
        _check(
            isinstance(paths, dict) and set(paths) == {"fp32", "int8"},
            "Both FP32 and int8 artifacts are required",
        )
        artifacts = self.directory / "artifacts"
        _check(
            artifacts.resolve().is_relative_to(self.directory.resolve()),
            "Artifacts directory escapes the returned run",
        )
        parents = set()
        required = set()
        provenance_by_variant = []
        for variant, name in paths.items():
            path = _inside(artifacts, name)
            bundle = ModelBundle.load(path)
            manifest = bundle.manifest
            _check(
                manifest["purpose"]
                == ("test_fixture" if self.fixture else "trained_candidate"),
                "Report and model disagree on fixture provenance",
            )
            provenance = manifest["training_provenance"]
            _check(isinstance(provenance, dict), "Missing training provenance")
            for key, expected in {
                "run_fingerprint": fingerprint,
                "variant": variant,
                "steps": state["steps"],
                "examples_seen": state["examples_seen"],
                "epochs_completed": state["epoch"],
                "model_id": report["model_id"],
                "model_revision": report["model_revision"],
                "versions": report["versions"],
            }.items():
                _check(
                    provenance.get(key) == expected,
                    "Model provenance differs from the report: " + key,
                )
            digest = provenance.get("dataset_manifest_sha256")
            _check(
                isinstance(digest, str)
                and re.fullmatch(r"[0-9a-f]{64}", digest),
                "Missing dataset manifest SHA-256",
            )
            _check(
                expected_dataset_sha256 is None
                or digest == expected_dataset_sha256,
                "The pilot uses a different dataset manifest",
            )
            checkpoint = provenance.get("checkpoint_model_sha256")
            _check(
                isinstance(checkpoint, str)
                and re.fullmatch(r"[0-9a-f]{64}", checkpoint),
                "Missing checkpoint model SHA-256",
            )
            _check(
                path.parent.name == report.get("candidate_id")
                and path.parent.name
                in (
                    "candidate-" + checkpoint[:16],
                    "candidate-"
                    + checkpoint[:16]
                    + "-"
                    + provenance["run_fingerprint"][:8],
                ),
                "Candidate directory is not bound to the checkpoint weights",
            )
            _check(
                bundle.settings.max_tokens <= 512,
                "Unexpected encoder window length",
            )
            if variant == "int8":
                _check(
                    provenance.get("quantization")
                    == {
                        "algorithm": "dynamic_qint8",
                        "operators": ["MatMul", "Gather"],
                        "per_channel": True,
                        "reduce_range": True,
                    },
                    "Unexpected quantization recipe",
                )
            parents.add(path.parent)
            for filename in (
                "manifest.json",
                *(f["path"] for f in manifest["files"].values()),
            ):
                required.add(
                    str(
                        (path / filename)
                        .resolve()
                        .relative_to(artifacts.resolve())
                    )
                )
            provenance_by_variant.append((digest, checkpoint, bundle.settings))
            self._bundles[variant] = bundle
        _check(
            len(parents) == 1
            and provenance_by_variant[0] == provenance_by_variant[1],
            "FP32 and int8 do not originate from the same checkpoint/data",
        )
        candidate = parents.pop()
        for field, filename in (
            ("profiles", "profiles.json"),
            ("quantization", "quantization.json"),
        ):
            stored = _read(_inside(candidate, filename))
            _check(
                stored == report.get(field),
                "Report and artifact disagree: " + filename,
            )
            required.add(
                str(
                    (candidate / filename)
                    .resolve()
                    .relative_to(artifacts.resolve())
                )
            )
        verify_receipt(artifacts, required)
        verify_profiles(report["profiles"], count)
        verify_quantization(report["profiles"], report["quantization"], count)
        self.expected_dataset_verified = expected_dataset_sha256 is not None
        if probe:
            from .model import OnnxWindowScorer

            for variant, bundle in self.bundles.items():
                scorer = OnnxWindowScorer(bundle.root, device="cpu", threads=2)
                self.probes[variant] = scorer.probe()
                if variant == "int8":
                    import onnx

                    graph = onnx.load(bundle.graph)
                    integer_ops = sum(
                        node.op_type == "MatMulInteger"
                        for node in graph.graph.node
                    )
                    _check(
                        integer_ops > 0,
                        "The int8 graph has no quantized MatMul",
                    )
                    self.probes[variant][
                        "matmul_integer_operations"
                    ] = integer_ops
                    del graph

    @property
    def report(self):
        return deepcopy(self._report)

    @property
    def bundles(self):
        return MappingProxyType(self._bundles)

    def profile(self, name="balanced", *, variant="fp32"):
        _check(
            self.complete,
            "An incomplete pilot has no usable operating profile",
        )
        _check(
            not self.fixture,
            "Fixture operating points cannot authorize inference",
        )
        _check(
            name in LIMITS and variant in self.bundles,
            "Unknown profile or variant",
        )
        item = self.report["profiles"][name]
        _check(
            item["all_labels_feasible"], "Requested FP32 profile is infeasible"
        )
        if variant == "int8":
            _check(
                self.report["quantization"][name]["passed"],
                "Requested int8 profile failed quantization acceptance",
            )
        return OperatingProfile(
            name,
            variant,
            self.bundles[variant].manifest_sha256,
            tuple(item["thresholds"][label] for label in LABELS),
        )

    def summary(self):
        result = {
            "schema_version": "pilot-review-0.1",
            "status": "verified" if self.complete else "incomplete",
            "source_status": self.report["status"],
            "report_sha256": self._report_sha256,
            "expected_dataset_verified": self.expected_dataset_verified,
            "fixture_only": self.fixture,
            "production_ready_detector": False,
            "test_split_accessed": False,
            "threshold_selection_split": "development",
            "metrics_recomputed_from": "reported_confusion_counts",
            "raw_development_predictions_recomputed": False,
            "artifact_receipt_verified": self.complete,
            "probes": self.probes,
        }
        if self.complete:
            result["profiles"] = {
                name: {
                    "fp32_feasible": item["all_labels_feasible"],
                    "fp32_infeasible_labels": [
                        label
                        for label, point in item["operating_points"].items()
                        if not point["feasible"]
                    ],
                    "int8_passed": self.report["quantization"][name]["passed"],
                    "fp32_macro_f1": item["metrics"]["macro_f1"],
                    "int8_macro_f1": self.report["quantization"][name][
                        "metrics"
                    ]["macro_f1"],
                    "int8_failures": self.report["quantization"][name][
                        "failures"
                    ],
                    "fp32_binary": item["metrics"][LABELS[0]],
                    "int8_binary": self.report["quantization"][name][
                        "metrics"
                    ][LABELS[0]],
                }
                for name, item in self.report["profiles"].items()
            }
            result["models"] = {
                name: bundle.manifest_sha256
                for name, bundle in self.bundles.items()
            }
        return result
