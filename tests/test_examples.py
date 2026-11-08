# SPDX-License-Identifier: Apache-2.0
"""Run the documented examples against an installed runtime package."""

from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from personal_threats import HistoryStore, ThreatAnalyzer

from model_fixtures import build_bundle


ROOT = Path(__file__).resolve().parents[1]


def run(arguments, directory, *, check=True):
    return subprocess.run(
        [sys.executable, *map(str, arguments)],
        cwd=directory,
        text=True,
        capture_output=True,
        check=check,
    )


class ExamplesTests(unittest.TestCase):
    def test_bilingual_readme_python_example_and_cli_input(self):
        english = (ROOT / "README.md").read_text(encoding="utf-8")
        russian = (ROOT / "README.ru.md").read_text(encoding="utf-8")
        blocks = [
            re.findall(r"```([^\n]*)\n(.*?)```", text, re.S)
            for text in (english, russian)
        ]
        self.assertEqual(blocks[0], blocks[1])
        with tempfile.TemporaryDirectory() as directory:
            snippets = [
                code for language, code in blocks[0] if language == "python"
            ]
            self.assertTrue(snippets)
            for code in snippets:
                run(["-c", code], directory)
            with HistoryStore(
                Path(directory) / "var/history.sqlite3", read_only=True
            ) as store:
                self.assertEqual(store.statistics()["messages"], 1)
            database = Path(directory) / "cli.sqlite3"
            prefix = ["-m", "personal_threats", "--db", database]
            run([*prefix, "init"], directory)
            for _ in range(2):
                result = run(
                    [
                        *prefix,
                        "import-jsonl",
                        ROOT / "examples/messages.jsonl",
                    ],
                    directory,
                )
                self.assertEqual(json.loads(result.stdout)["records"], 3)
            result = run([*prefix, "stats"], directory)
            counts = json.loads(result.stdout)
            self.assertEqual(counts["messages"], 3)
            self.assertEqual(counts["revisions"], 3)
            self.assertFalse(counts["detector_available"])

    def test_analysis_example_runs_real_onnx_with_explicit_test_only_loader(
        self,
    ):
        spec = importlib.util.spec_from_file_location(
            "analysis_example", ROOT / "examples/analyze.py"
        )
        example = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(example)
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            bundle = build_bundle(work / "fixture")
            source = ROOT / "examples/messages.jsonl"
            rejected = run(
                [
                    ROOT / "examples/analyze.py",
                    source,
                    "--model",
                    bundle,
                    "--db",
                    work / "rejected.sqlite3",
                ],
                directory,
                check=False,
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertFalse(rejected.stdout.strip())
            original_loader = ThreatAnalyzer.from_bundle

            def fixture_loader(*args, **kwargs):
                return original_loader(
                    *args, **kwargs, allow_test_fixture=True
                )

            captured = io.StringIO()
            with (
                patch.object(
                    ThreatAnalyzer, "from_bundle", side_effect=fixture_loader
                ),
                patch.object(example, "configure_logging"),
                patch.object(
                    sys,
                    "argv",
                    [
                        str(ROOT / "examples/analyze.py"),
                        str(source),
                        "--model",
                        str(bundle),
                        "--db",
                        str(work / "scored.sqlite3"),
                    ],
                ),
                redirect_stdout(captured),
            ):
                example.main()
            rows = [
                json.loads(line) for line in captured.getvalue().splitlines()
            ]
            self.assertEqual(len(rows), 4)
            self.assertEqual(rows[-1]["current_assessments"], 3)
            self.assertEqual(rows[-1]["outstanding_jobs"], 0)
            for row in rows[:-1]:
                self.assertIn("prediction", row)
                self.assertEqual(
                    row["prediction"]["model_purpose"], "test_fixture"
                )

    def test_backup_and_installation_examples_as_console_programs(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            source = work / "source.sqlite3"
            target = work / "snapshot.sqlite3"
            with HistoryStore(source):
                pass
            result = run(
                [ROOT / "examples/backup_database.py", source, target],
                directory,
            )
            self.assertEqual(
                json.loads(result.stdout)["status"], "verified_backup"
            )
            with HistoryStore(target, read_only=True) as snapshot:
                self.assertEqual(snapshot.statistics()["messages"], 0)
            result = run([ROOT / "examples/check_installation.py"], directory)
            check = json.loads(result.stdout)
            self.assertEqual(check["status"], "runtime_available")
            self.assertFalse(check["model_supplied"])
            self.assertFalse(check["production_ready_detector"])
            bundle = build_bundle(work / "fixture")
            rejected = run(
                [ROOT / "examples/check_installation.py", "--model", bundle],
                directory,
                check=False,
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertFalse(rejected.stdout.strip())
            for name in ("analyze", "backup_database", "check_installation"):
                self.assertIn(
                    "usage:",
                    run(
                        [ROOT / "examples" / (name + ".py"), "--help"],
                        directory,
                    ).stdout,
                )


if __name__ == "__main__":
    unittest.main()
