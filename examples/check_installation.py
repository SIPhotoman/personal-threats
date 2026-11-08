# SPDX-License-Identifier: Apache-2.0
"""Verify the installed runtime and, optionally, an operator-provided model."""

import argparse
from importlib import metadata
from pathlib import Path
import sqlite3
import sys

from personal_threats import __version__
from personal_threats.core import canonical_json
from personal_threats.model import ModelBundle, ModelError, OnnxWindowScorer
from personal_threats.review import QualityProfile


def verify(*, model=None, profile_report=None, device="auto"):
    if sys.version_info < (3, 14) or not sys.platform.startswith("linux"):
        raise RuntimeError("This distribution targets Linux CPython 3.14+")
    if sqlite3.sqlite_version_info < (3, 37):
        raise RuntimeError("SQLite 3.37+ with STRICT tables is required")
    import onnxruntime as ort

    installed = []
    for package in ("onnxruntime", "onnxruntime-gpu"):
        try:
            metadata.version(package)
        except metadata.PackageNotFoundError:
            continue
        installed.append(package)
    if len(installed) != 1:
        raise RuntimeError(
            "Install exactly one CPU or GPU ONNX Runtime package"
        )
    if profile_report is not None and model is None:
        raise ValueError("A profile report requires an explicit model bundle")
    result = {
        "status": "runtime_available",
        "package_version": __version__,
        "python": sys.version.split()[0],
        "sqlite": sqlite3.sqlite_version,
        "providers": ort.get_available_providers(),
        "dependencies": {
            name: metadata.version(name)
            for name in ("ijson", "numpy", "onnx", installed[0], "tokenizers")
        },
        "model_supplied": model is not None,
        "operating_profile_verified": False,
        "production_ready_detector": False,
    }
    if model is not None:
        bundle = ModelBundle.load(model)
        if bundle.manifest["purpose"] != "trained_candidate":
            raise ModelError("A test fixture is not an installable detector")
        result["model"] = OnnxWindowScorer(model, device=device).probe()
        if profile_report is not None:
            QualityProfile.load(profile_report, model, name="balanced")
            result["operating_profile_verified"] = True
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--profile-report", type=Path)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    args = parser.parse_args()
    print(canonical_json(verify(**vars(args))))


if __name__ == "__main__":
    main()
