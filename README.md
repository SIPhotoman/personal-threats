# personal-threats

[English](README.md) | [Русский](README.ru.md)

A Python library for local dialogue processing and personal-threat analysis,
with a CLI for importing messages, managing history and running inference.

**A trained model will be added later.** Message import, SQLite history and
processing queues already work without a model. Threat scoring requires a
compatible model bundle.

## Features

- Import JSONL files and Telegram Desktop JSON exports.
- Store dialogue history, message revisions and deletion markers in SQLite.
- Resume queued work after a restart and recalculate results after edits.
- Process message streams with bounded context and overlapping text windows;
  messages may contain up to 65,536 Unicode code points.
- Integrate ONNX models for CPU or NVIDIA CUDA inference.
- Access results through the Python API or CLI, with JSON output and logs.

## Installation

Linux, CPython 3.14+ and SQLite 3.37+ are required. From the repository root:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps .
```

`requirements.txt` contains the pinned CPU runtime, test and build tools.
For CUDA, use a separate environment with compatible NVIDIA libraries and
install `'.[cuda]'` instead. The minimal library installation is
`python -m pip install .`.

## Python usage

Messages can be stored and inspected before a model is connected:

```python
from personal_threats import HistoryStore, Message

with HistoryStore("var/history.sqlite3") as history:
    history.ingest(
        Message("dialog-demo", "1", "user-a", "Hello. Привет.")
    )
    print(history.statistics())
```

Once a model is available, `ThreatAnalyzer.from_bundle()` loads it, and
`process_message()` or `process_messages()` returns analysis results.
These methods are generators and must be consumed to complete processing.
See [examples/analyze.py](examples/analyze.py) for a complete example.

## CLI

```bash
personal-threats --db var/history.sqlite3 init
personal-threats --db var/history.sqlite3 import-jsonl examples/messages.jsonl
personal-threats --db var/history.sqlite3 stats
```

JSONL records require `dialog_id`, `message_id`, `sender_id` and `text`.
Telegram JSON export folders are imported with `import-telegram`.
Run `personal-threats --help` for available commands.

## Connecting a model

A compatible ONNX bundle can be placed in `models/` or another local
directory, then passed to `ThreatAnalyzer.from_bundle()` or the CLI's
`--model` option. Bundle files are verified against their manifest; models
are loaded from disk without automatic downloads.

The model interface supports a threat-presence score and six categories:
physical harm, coercion, stalking, doxxing, reputational harm and property
harm. Bundle format and operating profiles are described in
[models/README.md](models/README.md).

## Project structure

- `src/personal_threats/` — library and CLI.
- `tests/` — runtime and integration tests.
- `examples/` — usage, backup and installation-check examples.
- `models/` — instructions for adding a model.

## Tests

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

## License

[Apache-2.0](LICENSE). Separately supplied models retain their own licenses.
