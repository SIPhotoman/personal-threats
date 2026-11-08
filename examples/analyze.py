# SPDX-License-Identifier: Apache-2.0
"""Consume a JSONL stream through an explicitly supplied model bundle."""

import argparse
from pathlib import Path

from personal_threats import AnalysisSettings, ThreatAnalyzer
from personal_threats.cli import configure_logging
from personal_threats.core import canonical_json
from personal_threats.importers import read_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--profile-report", type=Path)
    parser.add_argument("--db", type=Path, default=Path("var/history.sqlite3"))
    args = parser.parse_args()
    configure_logging("INFO")
    with ThreatAnalyzer.from_bundle(
        args.db,
        args.model,
        report=args.profile_report,
        settings=AnalysisSettings(window_batch_size=8),
    ) as analyzer:
        for result in analyzer.process_messages(read_jsonl(args.input)):
            print(canonical_json(result.to_dict()))
        print(canonical_json(analyzer.statistics()))


if __name__ == "__main__":
    main()
