# SPDX-License-Identifier: Apache-2.0
"""Create a consistent SQLite backup without replacing an existing file."""

import argparse
from contextlib import closing
import os
from pathlib import Path
import sqlite3

from personal_threats.core import canonical_json
from personal_threats.store import APPLICATION_ID


def backup(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or not source.is_file():
        raise ValueError("Expected an existing source and a different output")
    if not destination.parent.is_dir():
        raise ValueError("Create the backup directory before taking a backup")
    with closing(
        sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)
    ) as origin:
        if (
            origin.execute("PRAGMA application_id").fetchone()[0]
            != APPLICATION_ID
        ):
            raise ValueError("The source is not a personal-threats database")
        descriptor = os.open(
            destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
        os.close(descriptor)
        # A failed backup is preserved for inspection, never silently removed.
        with closing(sqlite3.connect(destination)) as output:
            origin.backup(output, pages=256, sleep=0.05)
            if output.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
                raise RuntimeError("Backup integrity verification failed")
            output.commit()
    return {"status": "verified_backup", "path": str(destination)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(canonical_json(backup(args.source, args.destination)))


if __name__ == "__main__":
    main()
