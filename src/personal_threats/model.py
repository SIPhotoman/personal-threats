# SPDX-License-Identifier: Apache-2.0
"""Local verified ONNX bundles and lossless token-window score inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import logging
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from typing import Any, Iterator

from .core import (
    MAX_TEXT_CODEPOINTS,
    THREAT_TYPES,
    ContractError,
    Message,
    identifier,
    canonical_json,
    strict_json,
    validate_json,
)
from .results import Fragment, Prediction, Scores, TAXONOMY_VERSION

LOGGER = logging.getLogger("personal_threats")
LABELS = ("threat_present", *THREAT_TYPES)


def task_context_text(task: dict[str, Any]) -> str:
    """Serialize causal context identically in training and inference."""
    names = {
        p["participant_id"]: p.get("display_name")
        for p in task["participants"]
    }
    lines = []
    for message in task["context_messages"]:
        deleted = message.get("is_deleted", False)
        if deleted and task.get("context_policy") == "visible_history":
            continue
        name = names.get(message["sender_id"]) or "unknown"
        marker = " [deleted]" if deleted else ""
        lines.append(f"{name}{marker}: {message['text']}")
    sender = names.get(task["current_message"]["sender_id"]) or "unknown"
    lines.append("[current_sender] " + sender)
    text = "\n".join(lines)
    return text[-MAX_TEXT_CODEPOINTS:]


def quality_context_text(task: dict[str, Any]) -> str:
    """The quality model's second view requires actual visible history."""
    visible = any(
        not (
            message.get("is_deleted", False)
            and task.get("context_policy") == "visible_history"
        )
        for message in task["context_messages"]
    )
    return task_context_text(task) if visible else ""


class ModelError(RuntimeError):
    """A model artifact or inference request cannot be used reliably."""


def _dependency(name: str) -> Any:
    from importlib import import_module

    try:
        if name == "onnxruntime":
            os.environ["ORT_DISABLE_TELEMETRY"] = "1"
        module = import_module(name)
        if name == "onnxruntime":
            module.disable_telemetry_events()
        return module
    except ImportError as error:
        raise ModelError(
            f"Missing model dependency {name!r}; "
            "install personal-threats[cpu] or personal-threats[cuda]"
        ) from error


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class WindowSettings:
    max_tokens: int = 256
    overlap_tokens: int = 32
    context_tokens: int = 64
    pad_token_id: int = 0
    unknown_token_id: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "max_tokens",
            "overlap_tokens",
            "context_tokens",
            "pad_token_id",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ContractError(f"{name} must be a nonnegative integer")
        if not 2 <= self.max_tokens <= 8192:
            raise ContractError("max_tokens must be in [2, 8192]")
        if self.overlap_tokens >= self.max_tokens:
            raise ContractError("Overlap must be smaller than a window")
        if self.context_tokens >= self.max_tokens:
            raise ContractError("Context must leave space for current text")
        if self.unknown_token_id is not None and (
            type(self.unknown_token_id) is not int or self.unknown_token_id < 0
        ):
            raise ContractError("unknown_token_id must be nonnegative or null")


@dataclass(frozen=True, slots=True)
class TokenWindow:
    index: int
    token_start: int
    token_end: int
    original_start: int
    original_end: int
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    token_type_ids: tuple[int, ...]
    offsets: tuple[tuple[int, int] | None, ...]
    context_truncated: bool
    unmapped_content: bool
    unknown_tokens: bool
    normalization_changed: bool
    source_text_sha256: str
    current_token_mask: tuple[int, ...] = ()


def fragment_from_tokens(
    window: TokenWindow,
    start: int,
    end: int,
    message: Message,
) -> Fragment:
    """Map current-sequence tokens to original code points and verify input."""
    if (
        type(start) is not int
        or type(end) is not int
        or not 0 <= start < end <= len(window.offsets)
    ):
        raise ContractError("Invalid token fragment range")
    if (
        hashlib.sha256(message.text.encode()).hexdigest()
        != window.source_text_sha256
    ):
        raise ContractError("Token window belongs to another original text")
    offsets = window.offsets[start:end]
    if any(span is None for span in offsets):
        raise ContractError(
            "Fragment includes context, padding, or special tokens"
        )
    first = min(span[0] for span in offsets)
    last = max(span[1] for span in offsets)
    fragment = Fragment(
        message.message_id, first, last, message.text[first:last]
    )
    fragment.validate(message)
    return fragment


class TokenWindowizer:
    """Immutable tokenizer configuration with complete overflow verification.

    Each instance owns an untruncated reference tokenizer and a windowed
    tokenizer. Calls do not mutate their configuration. Offset comparisons
    verify every original token across all overlapping windows.
    """

    def __init__(self, tokenizer_json: str, settings: WindowSettings) -> None:
        tokenizers = _dependency("tokenizers")
        self.settings = settings
        try:
            self._reference = tokenizers.Tokenizer.from_str(tokenizer_json)
            self._working = tokenizers.Tokenizer.from_str(tokenizer_json)
            for tokenizer in (self._reference, self._working):
                tokenizer.no_truncation()
                tokenizer.no_padding()
            specials = max(
                self._working.num_special_tokens_to_add(False),
                self._working.num_special_tokens_to_add(True),
            )
            capacity = settings.max_tokens - settings.context_tokens - specials
            if capacity <= settings.overlap_tokens:
                raise ContractError(
                    "Context and special tokens leave too little "
                    "space for the requested overlap"
                )
            for token_id in (settings.pad_token_id, settings.unknown_token_id):
                if (
                    token_id is not None
                    and self._working.id_to_token(token_id) is None
                ):
                    raise ContractError(
                        "Special token ID is absent from vocabulary"
                    )
            self._working.enable_truncation(
                max_length=settings.max_tokens,
                stride=settings.overlap_tokens,
                strategy="only_first",
                direction="right",
            )
        except ContractError:
            raise
        except Exception as error:
            raise ModelError(f"Cannot load tokenizer: {error}") from error

    def focus_template(self) -> dict[str, list[int]]:
        """Verify the native single-sequence envelope for focus views."""
        empty = self._reference.encode("").ids
        probe = self._reference.encode("A model проверка")
        content = [
            i for i, flag in enumerate(probe.special_tokens_mask) if not flag
        ]
        if (
            len(empty) != 2
            or not content
            or content[0] != 1
            or content[-1] != len(probe.ids) - 2
            or probe.ids[0] != empty[0]
            or probe.ids[-1] != empty[1]
        ):
            raise ModelError("Unsupported single-sequence tokenizer envelope")
        return {"prefix": [empty[0]], "suffix": [empty[1]]}

    @staticmethod
    def _text(value: str) -> None:
        if not isinstance(value, str) or len(value) > MAX_TEXT_CODEPOINTS:
            raise ContractError("Tokenization input exceeds the message limit")
        validate_json(value)

    def _context(self, text: str) -> tuple[str, bool, bool]:
        """Keep a recent suffix of bounded context, with explicit omission."""
        self._text(text)
        candidate = text
        budget = self.settings.context_tokens
        if not budget:
            return "", bool(text), False
        while candidate:
            encoded = self._reference.encode(
                candidate, add_special_tokens=False
            )
            if len(encoded.ids) <= budget:
                unknown = self.settings.unknown_token_id in encoded.ids
                return candidate, candidate != text, unknown
            offsets = encoded.offsets
            # Byte-level tokens can share a code point. Advance at least one
            # whole character, then re-encode to respect the actual token cap.
            start = max(1, offsets[-budget][0])
            candidate = candidate[start:]
        return "", bool(text), False

    def windows(
        self, text: str, *, context_text: str = ""
    ) -> Iterator[TokenWindow]:
        self._text(text)
        context_text, context_truncated, context_unknown = self._context(
            context_text
        )
        try:
            reference = self._reference.encode(text, add_special_tokens=False)
            encoded = self._working.encode(text, context_text or None)
        except Exception as error:
            raise ModelError(f"Tokenization failed: {error}") from error
        reference_ids = reference.ids
        reference_offsets = reference.offsets
        normalizer = self._reference.normalizer
        normalization_changed = bool(
            normalizer and normalizer.normalize_str(text) != text
        )
        covered = bytearray(len(text))
        for start, end in reference_offsets:
            if not 0 <= start <= end <= len(text):
                raise ModelError(
                    "Tokenizer offsets are not original code points"
                )
            covered[start:end] = b"\x01" * (end - start)
        unmapped = any(
            not covered[index] and not character.isspace()
            for index, character in enumerate(text)
        )
        unknown = (
            context_unknown or self.settings.unknown_token_id in reference_ids
        )
        start_token = 0
        previous_end = 0
        source_sha256 = hashlib.sha256(text.encode()).hexdigest()
        windows = (encoded, *encoded.overflowing)
        for index, item in enumerate(windows):
            if item.overflowing and index:
                raise ModelError("Unexpected nested overflow encodings")
            ids = item.ids
            offsets = item.offsets
            sequence_ids = item.sequence_ids
            positions = [
                position
                for position, sequence in enumerate(sequence_ids)
                if sequence == 0
            ]
            current_ids = [ids[position] for position in positions]
            current_offsets = [offsets[position] for position in positions]
            end_token = start_token + len(current_ids)
            if (
                current_ids != reference_ids[start_token:end_token]
                or current_offsets != reference_offsets[start_token:end_token]
            ):
                raise ModelError(
                    "Token overflow omitted or changed current text"
                )
            if len(ids) > self.settings.max_tokens:
                raise ModelError("Token window exceeds the declared maximum")
            last = index == len(windows) - 1
            if last != (end_token == len(reference_ids)):
                raise ModelError("Token overflow is incomplete or repeated")
            original_start = min(
                (span[0] for span in current_offsets), default=0
            )
            original_end = max(
                (span[1] for span in current_offsets), default=0
            )
            if not index:
                original_start = 0
            elif original_start > previous_end:
                original_start = previous_end
            if last:
                original_end = len(text)
            yield TokenWindow(
                index,
                start_token,
                end_token,
                original_start,
                original_end,
                tuple(ids),
                tuple(item.attention_mask),
                tuple(item.type_ids),
                tuple(
                    offsets[position] if sequence == 0 else None
                    for position, sequence in enumerate(sequence_ids)
                ),
                context_truncated,
                unmapped,
                unknown,
                normalization_changed,
                source_sha256,
                tuple(int(sequence == 0) for sequence in sequence_ids),
            )
            previous_end = max(previous_end, original_end)
            if not last and end_token <= start_token:
                raise ModelError("Tokenizer produced a non-progressing window")
            start_token = end_token - self.settings.overlap_tokens


@dataclass(frozen=True, slots=True)
class ModelBundle:
    root: Path
    _manifest_json: str
    manifest_sha256: str
    graph: Path
    tokenizer: Path
    settings: WindowSettings

    @property
    def manifest(self) -> dict[str, Any]:
        """Return a copy so callers cannot mutate verified provenance."""
        return strict_json(self._manifest_json)

    @classmethod
    def load(cls, directory: str | Path) -> ModelBundle:
        root = Path(directory).resolve()
        path = root / "manifest.json"
        if not path.resolve().is_relative_to(root):
            raise ModelError("Manifest must stay within its model bundle")
        if path.stat().st_size > 1024 * 1024:
            raise ModelError("Model manifest exceeds 1 MiB")
        content = path.read_bytes()
        try:
            text = content.decode("utf-8")
        except UnicodeError as error:
            raise ModelError("Model manifest must be UTF-8") from error
        manifest = strict_json(text)
        expected = {
            "schema_version",
            "taxonomy_version",
            "model_version",
            "purpose",
            "task",
            "labels",
            "aggregation",
            "windowing",
            "files",
            "license",
            "training_provenance",
        }
        feature_task = isinstance(manifest, dict) and (
            manifest.get("task") == "window_features_v3"
        )
        if feature_task:
            expected.add("focus_template")
        if not isinstance(manifest, dict) or set(manifest) != expected:
            raise ModelError("Invalid model manifest fields")
        if (
            manifest["schema_version"]
            != ("model-bundle-0.2" if feature_task else "model-bundle-0.1")
            or manifest["taxonomy_version"] != TAXONOMY_VERSION
            or manifest["task"]
            not in (
                "window_scores_v1",
                "window_scores_v2",
                "window_features_v3",
            )
            or manifest["aggregation"] != "max_logit"
            or manifest["labels"] != list(LABELS)
        ):
            raise ModelError("Unsupported model contract or label order")
        identifier(manifest["model_version"], "model_version")
        if manifest["purpose"] not in ("trained_candidate", "test_fixture"):
            raise ModelError("Unknown model purpose")
        if not isinstance(manifest["license"], str) or not manifest["license"]:
            raise ModelError("Model license expression is required")
        provenance = manifest["training_provenance"]
        if manifest["purpose"] == "trained_candidate" and (
            not isinstance(provenance, dict) or not provenance
        ):
            raise ModelError("Trained candidate requires training provenance")
        windowing = manifest["windowing"]
        fields = set(WindowSettings.__dataclass_fields__)
        if not isinstance(windowing, dict) or set(windowing) != fields:
            raise ModelError("Invalid window settings")
        settings = WindowSettings(**windowing)
        if feature_task:
            template = manifest["focus_template"]
            if (
                not isinstance(template, dict)
                or set(template) != {"prefix", "suffix"}
                or any(
                    not isinstance(v, list)
                    or not 1 <= len(v) <= 8
                    or any(type(i) is not int or i < 0 for i in v)
                    for v in template.values()
                )
            ):
                raise ModelError("Invalid focus sequence template")
        files = manifest["files"]
        roles = {"graph", "tokenizer", "license"}
        if feature_task:
            roles.add("heads")
        if not isinstance(files, dict) or set(files) != roles:
            raise ModelError("Graph, tokenizer and license files are required")
        paths = {}
        limits = {
            "graph": (2 if feature_task else 1) * 1024**3,
            "tokenizer": 64 * 1024**2,
            "license": 1024**2,
            "heads": 16 * 1024**2,
        }
        for role, descriptor in files.items():
            if not isinstance(descriptor, dict) or set(descriptor) != {
                "path",
                "bytes",
                "sha256",
            }:
                raise ModelError("Invalid file descriptor")
            name = descriptor["path"]
            if not isinstance(name, str) or not name or "\\" in name:
                raise ModelError("Invalid artifact path")
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ModelError("Artifact path must stay within its bundle")
            target = (root / name).resolve()
            if not target.is_relative_to(root) or not target.is_file():
                raise ModelError(
                    "Artifact file escapes or is absent from bundle"
                )
            size = descriptor["bytes"]
            digest = descriptor["sha256"]
            if (
                type(size) is not int
                or not 0 < size <= limits[role]
                or size != target.stat().st_size
            ):
                raise ModelError("Artifact size differs or exceeds its limit")
            if not isinstance(digest, str) or not re.fullmatch(
                "[0-9a-f]{64}", digest
            ):
                raise ModelError("Invalid artifact SHA-256")
            if file_sha256(target) != digest:
                raise ModelError("Artifact SHA-256 mismatch")
            paths[role] = target
        if len(set(paths.values())) != len(roles):
            raise ModelError("Artifact files must be distinct")
        return cls(
            root,
            text,
            hashlib.sha256(content).hexdigest(),
            paths["graph"],
            paths["tokenizer"],
            settings,
        )

    def validate_graph(self) -> None:
        """Inspect ONNX without reading external tensor files."""
        onnx = _dependency("onnx")
        try:
            graph = onnx.load_model(self.graph, load_external_data=False)

            def inspect(message: Any) -> None:
                if isinstance(message, onnx.TensorProto):
                    if (
                        message.data_location == onnx.TensorProto.EXTERNAL
                        or message.external_data
                    ):
                        raise ModelError(
                            "External ONNX tensor data is unsupported "
                            "by this bundle version"
                        )
                for field, value in message.ListFields():
                    if field.type == field.TYPE_MESSAGE:
                        if field.is_repeated:
                            for child in value:
                                inspect(child)
                        else:
                            inspect(value)

            inspect(graph)
            onnx.checker.check_model(graph)
        except ModelError:
            raise
        except Exception as error:
            raise ModelError(f"Invalid ONNX graph: {error}") from error


@dataclass(frozen=True, slots=True)
class WindowScore:
    window: TokenWindow
    logits: tuple[float, ...]

    @property
    def scores(self) -> Scores:
        values = tuple(sigmoid(value) for value in self.logits)
        return Scores(values[0], values[1:])


def export_model_bundle(
    directory: str | Path,
    *,
    graph: str | Path,
    tokenizer: str | Path,
    license_file: str | Path,
    license_expression: str,
    model_version: str,
    training_provenance: dict[str, Any] | None,
    settings: WindowSettings,
    purpose: str = "trained_candidate",
    task: str = "window_scores_v1",
    heads: str | Path | None = None,
    focus_template: dict[str, list[int]] | None = None,
) -> ModelBundle:
    """Build, verify and atomically publish a new local artifact directory.

    The source files remain untouched. Cooperating exporters share a process
    lock and preserve existing versions. Published bundles are immutable.
    """
    import fcntl

    destination = Path(directory).absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(str(destination) + ".build.lock", "a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError("Model destination already exists")
        temporary = Path(
            tempfile.mkdtemp(
                prefix="." + destination.name + ".", dir=destination.parent
            )
        )
        try:
            files = {}
            sources = [
                ("graph", graph, "model.onnx"),
                ("tokenizer", tokenizer, "tokenizer.json"),
                ("license", license_file, "LICENSE.txt"),
            ]
            if task == "window_features_v3":
                if heads is None or focus_template is None:
                    raise ModelError(
                        "Feature models require heads and template"
                    )
                sources.append(("heads", heads, "heads.npz"))
            elif heads is not None or focus_template is not None:
                raise ModelError(
                    "Legacy score models do not accept fusion heads"
                )
            for role, source, name in sources:
                target = temporary / name
                shutil.copyfile(source, target)
                with target.open("rb") as stream:
                    os.fsync(stream.fileno())
                files[role] = {
                    "path": name,
                    "bytes": target.stat().st_size,
                    "sha256": file_sha256(target),
                }
            manifest = {
                "schema_version": (
                    "model-bundle-0.2"
                    if heads is not None
                    else "model-bundle-0.1"
                ),
                "taxonomy_version": TAXONOMY_VERSION,
                "model_version": model_version,
                "purpose": purpose,
                "task": task,
                "labels": list(LABELS),
                "aggregation": "max_logit",
                "windowing": asdict(settings),
                "license": license_expression,
                "training_provenance": training_provenance,
                "files": files,
            }
            if heads is not None:
                manifest["focus_template"] = focus_template
            path = temporary / "manifest.json"
            with path.open("w", encoding="utf-8") as stream:
                stream.write(canonical_json(manifest) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            scorer = OnnxWindowScorer(temporary, device="cpu")
            scorer.probe()
            del scorer
            descriptor = os.open(temporary, os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            os.rename(temporary, destination)
            descriptor = os.open(destination.parent, os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            return ModelBundle.load(destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)


def sigmoid(value: float) -> float:
    if not math.isfinite(value):
        raise ModelError("Non-finite model logit")
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1 + exponent)


def focus_sequence(ids, current, template):
    """Keep exact current wordpieces; never retokenize a window substring."""
    if len(ids) != len(current) or any(i not in (0, 1) for i in current):
        raise ModelError("Invalid current token mask for focus view")
    selected = [token for token, keep in zip(ids, current) if keep]
    prefix, suffix = template["prefix"], template["suffix"]
    return (
        prefix + selected + suffix,
        [0] * len(prefix) + [1] * len(selected) + [0] * len(suffix),
    )


class FusionHeads:
    """Small NumPy head; one ONNX session owns the shared encoder weights."""

    def __init__(self, path, np):
        import zipfile

        self.np = np
        names = {
            "classifier.weight",
            "classifier.bias",
            "gate_in.weight",
            "gate_in.bias",
            "gate_out.weight",
            "gate_out.bias",
        }
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            if (
                len(entries) != len(names)
                or {i.filename for i in entries} != {n + ".npy" for n in names}
                or sum(i.file_size for i in entries) > 16 * 1024**2
            ):
                raise ModelError("Invalid or oversized fusion head archive")
        with np.load(path, allow_pickle=False) as source:
            self.weights = {name: source[name] for name in names}
        w = self.weights
        shape = w["classifier.weight"].shape
        if (
            len(shape) != 2
            or shape[0] != len(LABELS)
            or not 1 <= shape[1] <= 2048
        ):
            raise ModelError("Invalid classifier dimensions")
        self.hidden = shape[1]
        gate = w["gate_in.bias"].shape
        if len(gate) != 1 or not 1 <= gate[0] <= 128:
            raise ModelError("Invalid gate dimensions")
        shapes = {
            "classifier.weight": shape,
            "classifier.bias": (7,),
            "gate_in.weight": (gate[0], 4 * self.hidden),
            "gate_in.bias": gate,
            "gate_out.weight": (1, gate[0]),
            "gate_out.bias": (1,),
        }
        if any(
            w[n].dtype != np.float32
            or w[n].shape != shapes[n]
            or not np.isfinite(w[n]).all()
            for n in names
        ):
            raise ModelError("Invalid, nonfinite or non-float32 fusion heads")

    def __call__(self, focus, context, available):
        np, w = self.np, self.weights
        features = np.concatenate(
            [focus, context, np.abs(focus - context), focus * context], axis=1
        )
        gate = np.tanh(features @ w["gate_in.weight"].T + w["gate_in.bias"])
        gate = gate @ w["gate_out.weight"].T + w["gate_out.bias"]
        gate = available[:, None] / (1 + np.exp(-np.clip(gate, -80, 80)))
        pooled = focus + gate * (context - focus)
        logits = pooled @ w["classifier.weight"].T + w["classifier.bias"]
        logits[:, 0] = logits.max(axis=1)
        return logits


class OnnxWindowScorer:
    """Seven-output classifier adapter; event extraction is not synthesized."""

    def __init__(
        self,
        directory: str | Path,
        *,
        device: str = "auto",
        threads: int = 0,
        allow_test_fixture: bool = False,
    ) -> None:
        if device not in ("auto", "cpu", "cuda"):
            raise ContractError("device must be auto, cpu, or cuda")
        if type(threads) is not int or not 0 <= threads <= 256:
            raise ContractError("threads must be an integer in [0, 256]")
        if type(allow_test_fixture) is not bool:
            raise ContractError("allow_test_fixture must be boolean")
        self.bundle = ModelBundle.load(directory)
        self._allow_test_fixture = allow_test_fixture
        self.bundle.validate_graph()
        try:
            tokenizer_json = self.bundle.tokenizer.read_text(encoding="utf-8")
        except UnicodeError as error:
            raise ModelError("Tokenizer JSON must be UTF-8") from error
        self.windowizer = TokenWindowizer(tokenizer_json, self.bundle.settings)
        self.np = _dependency("numpy")
        self.fusion = None
        if self.bundle.manifest["task"] == "window_features_v3":
            manifest = self.bundle.manifest
            if self.windowizer.focus_template() != manifest["focus_template"]:
                raise ModelError("Focus envelope differs from tokenizer")
            self.fusion = FusionHeads(
                self.bundle.root / manifest["files"]["heads"]["path"], self.np
            )
        self.ort = _dependency("onnxruntime")
        available = self.ort.get_available_providers()
        use_cuda = device != "cpu" and "CUDAExecutionProvider" in available
        if device == "cuda" and not use_cuda:
            raise ModelError(
                "CUDA provider is unavailable in this environment"
            )
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if use_cuda
            else ["CPUExecutionProvider"]
        )
        options = self.ort.SessionOptions()
        options.intra_op_num_threads = threads or min(4, os.cpu_count() or 1)
        options.inter_op_num_threads = 1
        options.execution_mode = self.ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = (
            self.ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        options.enable_mem_pattern = False
        options.enable_cpu_mem_arena = False
        try:
            self.session = self.ort.InferenceSession(
                str(self.bundle.graph),
                sess_options=options,
                providers=providers,
            )
        except Exception as error:
            if not use_cuda or device == "cuda":
                raise ModelError(
                    f"Cannot initialize ONNX session: {error}"
                ) from error
            LOGGER.warning(
                "cuda_initialization_failed",
                extra={
                    "fields": {
                        "detail": str(error),
                        "selected_provider": "CPUExecutionProvider",
                    }
                },
            )
            try:
                self.session = self.ort.InferenceSession(
                    str(self.bundle.graph),
                    sess_options=options,
                    providers=["CPUExecutionProvider"],
                )
            except Exception as cpu_error:
                raise ModelError(
                    f"Cannot initialize CPU session: {cpu_error}"
                ) from cpu_error
        if (
            use_cuda
            and "CUDAExecutionProvider" not in self.session.get_providers()
        ):
            if device == "cuda":
                raise ModelError("Requested CUDA session fell back to CPU")
            LOGGER.warning(
                "cuda_unavailable_falling_back",
                extra={
                    "fields": {"selected_provider": "CPUExecutionProvider"}
                },
            )
        self.session.disable_fallback()
        self._signature()

    def _signature(self) -> None:
        inputs = self.session.get_inputs()
        self.input_names = {item.name for item in inputs}
        required = {"input_ids", "attention_mask"}
        if self.bundle.manifest["task"] in (
            "window_scores_v2",
            "window_features_v3",
        ):
            required.add("current_token_mask")
        if not required <= self.input_names or not self.input_names <= (
            required | {"token_type_ids"}
        ):
            raise ModelError("Unsupported ONNX inputs")
        fixed_lengths = set()
        for item in inputs:
            if item.type != "tensor(int64)" or len(item.shape) != 2:
                raise ModelError(
                    "Inputs must be int64 [batch, sequence] tensors"
                )
            batch, length = item.shape
            if isinstance(batch, int) and batch != 1:
                raise ModelError("Fixed ONNX batch size must be one")
            if isinstance(length, int):
                fixed_lengths.add(length)
        if fixed_lengths and fixed_lengths != {
            self.bundle.settings.max_tokens
        }:
            raise ModelError("Graph sequence length differs from the manifest")
        self.fixed_length = next(iter(fixed_lengths), None)
        self.fixed_batch = any(
            isinstance(item.shape[0], int) for item in inputs
        )
        outputs = self.session.get_outputs()
        if self.fusion is not None:
            shapes = {
                "features": (2, self.fusion.hidden),
                "evidence": (3, len(LABELS)),
            }
            if {o.name for o in outputs} != set(shapes):
                raise ModelError(
                    "Feature graph requires features and evidence"
                )
            for output in outputs:
                rank, width = shapes[output.name]
                if (
                    output.type != "tensor(float)"
                    or len(output.shape) != rank
                    or output.shape[-1] != width
                ):
                    raise ModelError("Invalid feature graph output signature")
            return
        if len(outputs) != 1 or outputs[0].name != "logits":
            raise ModelError("Expected one ONNX output named logits")
        output = outputs[0]
        if (
            output.type != "tensor(float)"
            or len(output.shape) != 2
            or output.shape[1] != len(LABELS)
        ):
            raise ModelError("Expected float32 logits [batch, 7]")

    def _batch(self, windows: list[TokenWindow]) -> Iterator[WindowScore]:
        if self.fusion is not None:
            yield from self._fusion_batch(windows)
            return
        length = self.fixed_length or max(
            len(item.input_ids) for item in windows
        )
        shape = (len(windows), length)
        ids = self.np.full(
            shape, self.bundle.settings.pad_token_id, dtype=self.np.int64
        )
        mask = self.np.zeros(shape, dtype=self.np.int64)
        types = self.np.zeros(shape, dtype=self.np.int64)
        current = self.np.zeros(shape, dtype=self.np.int64)
        for index, item in enumerate(windows):
            count = len(item.input_ids)
            ids[index, :count] = item.input_ids
            mask[index, :count] = item.attention_mask
            types[index, :count] = item.token_type_ids
            if "current_token_mask" in self.input_names:
                if len(item.current_token_mask) != count:
                    raise ModelError(
                        "Current token mask is missing or invalid"
                    )
                current[index, :count] = item.current_token_mask
        available = {
            "input_ids": ids,
            "attention_mask": mask,
            "token_type_ids": types,
            "current_token_mask": current,
        }
        try:
            logits = self.session.run(
                ["logits"],
                {name: available[name] for name in self.input_names},
            )[0]
        except Exception as error:
            raise ModelError(f"ONNX inference failed: {error}") from error
        if (
            logits.shape != (len(windows), len(LABELS))
            or logits.dtype != self.np.float32
            or not self.np.isfinite(logits).all()
        ):
            raise ModelError("Invalid, incomplete or non-finite ONNX output")
        for item, row in zip(windows, logits):
            yield WindowScore(item, tuple(float(value) for value in row))

    def _feature_view(self, sequences):
        np = self.np
        length = self.fixed_length or max(len(ids) for ids, _, _ in sequences)
        shape = (len(sequences), length)
        inputs = {
            name: np.zeros(shape, dtype=np.int64) for name in self.input_names
        }
        inputs["input_ids"].fill(self.bundle.settings.pad_token_id)
        for index, (ids, mask, types) in enumerate(sequences):
            count = len(ids)
            if count > length:
                raise ModelError("Focus sequence exceeds graph window")
            inputs["input_ids"][index, :count] = ids
            inputs["attention_mask"][index, :count] = 1
            inputs["current_token_mask"][index, :count] = mask
            if "token_type_ids" in inputs:
                inputs["token_type_ids"][index, :count] = types
        try:
            features, evidence = self.session.run(
                ["features", "evidence"], inputs
            )
        except Exception as error:
            raise ModelError(
                f"ONNX feature inference failed: {error}"
            ) from error
        if (
            features.shape != (shape[0], self.fusion.hidden)
            or evidence.shape != (*shape, len(LABELS))
            or features.dtype != np.float32
            or evidence.dtype != np.float32
            or not np.isfinite(features).all()
            or not np.isfinite(evidence).all()
        ):
            raise ModelError("Invalid or nonfinite feature graph output")
        return features

    def _fusion_batch(self, windows):
        template = self.bundle.manifest["focus_template"]
        focus, context_indices = [], []
        for index, item in enumerate(windows):
            ids, mask = focus_sequence(
                item.input_ids, item.current_token_mask, template
            )
            focus.append((ids, mask, [0] * len(ids)))
            if any(mask) and len(item.input_ids) > len(ids):
                context_indices.append(index)
        features = self._feature_view(focus)
        context = features.copy()
        available = self.np.zeros(len(windows), dtype=self.np.float32)
        if context_indices:
            sequences = [
                (
                    windows[i].input_ids,
                    windows[i].current_token_mask,
                    windows[i].token_type_ids,
                )
                for i in context_indices
            ]
            context[context_indices] = self._feature_view(sequences)
            available[context_indices] = 1
        logits = self.fusion(features, context, available)
        if not self.np.isfinite(logits).all():
            raise ModelError("Fusion produced nonfinite scores")
        for item, row in zip(windows, logits):
            yield WindowScore(item, tuple(float(v) for v in row))

    def window_scores(
        self,
        text: str,
        *,
        context_text: str = "",
        batch_size: int = 8,
    ) -> Iterator[WindowScore]:
        if (
            self.bundle.manifest["purpose"] == "test_fixture"
            and not self._allow_test_fixture
        ):
            raise ModelError(
                "Test fixtures cannot be used as trained detectors"
            )
        if type(batch_size) is not int or not 1 <= batch_size <= 64:
            raise ContractError("batch_size must be an integer in [1, 64]")
        limit = 1 if self.fixed_batch else batch_size
        batch = []
        for window in self.windowizer.windows(text, context_text=context_text):
            batch.append(window)
            if len(batch) == limit:
                yield from self._batch(batch)
                batch.clear()
        if batch:
            yield from self._batch(batch)

    def score(
        self,
        message: Message,
        *,
        context_text: str = "",
        batch_size: int = 8,
    ) -> Prediction:
        maximum = [-math.inf] * len(LABELS)
        count = 0
        context_truncated = unmapped = unknown = normalization_changed = False
        for scored in self.window_scores(
            message.text, context_text=context_text, batch_size=batch_size
        ):
            count += 1
            maximum = [
                max(first, second)
                for first, second in zip(maximum, scored.logits)
            ]
            context_truncated |= scored.window.context_truncated
            unmapped |= scored.window.unmapped_content
            unknown |= scored.window.unknown_tokens
            normalization_changed |= scored.window.normalization_changed
        values = tuple(sigmoid(value) for value in maximum)
        prediction = Prediction(
            message.dialog_id,
            message.message_id,
            message.sender_id,
            self.bundle.manifest["model_version"],
            self.bundle.manifest_sha256,
            Scores(values[0], values[1:]),
            window_count=count,
            context_truncated=context_truncated,
            unmapped_content=unmapped,
            unknown_tokens=unknown,
            model_purpose=self.bundle.manifest["purpose"],
            normalization_changed=normalization_changed,
            context_insufficient=context_truncated,
        )
        prediction.validate(message)
        return prediction

    def runtime_configuration(self, *, batch_size: int = 8) -> dict[str, Any]:
        if type(batch_size) is not int or not 1 <= batch_size <= 64:
            raise ContractError("batch_size must be an integer in [1, 64]")
        return {
            "bundle_sha256": self.bundle.manifest_sha256,
            "providers": self.session.get_providers(),
            "onnxruntime": self.ort.__version__,
            "numpy": self.np.__version__,
            "windowing": asdict(self.bundle.settings),
            "aggregation": "max_logit",
            "batch_size": 1 if self.fixed_batch else batch_size,
            "intra_op_threads": (
                self.session.get_session_options().intra_op_num_threads
            ),
        }

    def probe(self) -> dict[str, Any]:
        """Check shape and finite output without making a threat decision."""
        windows = list(self.windowizer.windows("Проверка model 😀"))
        list(self._batch(windows[:1]))
        return {
            "status": "compatible",
            "model_version": self.bundle.manifest["model_version"],
            "model_purpose": self.bundle.manifest["purpose"],
            "bundle_sha256": self.bundle.manifest_sha256,
            "graph_sha256": self.bundle.manifest["files"]["graph"]["sha256"],
            "task": self.bundle.manifest["task"],
            "labels": list(LABELS),
            **self.runtime_configuration(),
            "calibration_status": "uncalibrated",
            "event_extraction_available": False,
            "production_ready_detector": False,
        }
