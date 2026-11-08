# SPDX-License-Identifier: Apache-2.0
"""Exercise consistent backups and the deployment diagnostics contract."""

import importlib.util
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest

from personal_threats import HistoryStore, Message


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "examples" / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReleaseToolsTests(unittest.TestCase):
    def test_backup_captures_committed_wal_and_never_replaces_an_output(self):
        tool = load_script("backup_database")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / "live.sqlite3", root / "backup.sqlite3"
            with HistoryStore(source) as store:
                store.ingest(Message("d", "1", "sender", "hello"))
                report = tool.backup(source, target)
                self.assertEqual(report["status"], "verified_backup")
                with HistoryStore(target, read_only=True) as snapshot:
                    self.assertEqual(snapshot.statistics()["messages"], 1)
                store.ingest(Message("d", "2", "sender", "world"))
                before = target.read_bytes()
                with self.assertRaises(FileExistsError):
                    tool.backup(source, target)
                self.assertEqual(target.read_bytes(), before)
            with closing(sqlite3.connect(target)) as db:
                self.assertEqual(
                    db.execute("PRAGMA quick_check").fetchone()[0], "ok"
                )

    def test_backup_rejects_a_foreign_database_without_creating_an_output(
        self,
    ):
        tool = load_script("backup_database")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / "other.sqlite3", root / "backup.sqlite3"
            with closing(sqlite3.connect(source)) as db:
                db.execute("CREATE TABLE unrelated(value)")
                db.commit()
            with self.assertRaises(ValueError):
                tool.backup(source, target)
            self.assertFalse(target.exists())

    def test_installation_check_keeps_model_and_runtime_readiness_separate(
        self,
    ):
        tool = load_script("check_installation")
        result = tool.verify(device="cpu")
        self.assertEqual(result["status"], "runtime_available")
        self.assertFalse(result["model_supplied"])
        self.assertFalse(result["production_ready_detector"])
        with self.assertRaises(ValueError):
            tool.verify(profile_report="unbound.json")


if __name__ == "__main__":
    unittest.main()
