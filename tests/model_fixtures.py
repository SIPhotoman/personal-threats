# SPDX-License-Identifier: Apache-2.0
"""Generate untrained ONNX fixtures; never use them as threat detectors."""

from dataclasses import asdict
import json
from pathlib import Path

from personal_threats.model import LABELS, WindowSettings, file_sha256
from personal_threats.results import TAXONOMY_VERSION


def fixture_tokenizer(*, byte_level=False, remove_zero_width=False):
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers
    from tokenizers import processors

    specials = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}
    if byte_level:
        vocabulary = {
            **specials,
            **{
                character: index + 4
                for index, character in enumerate(
                    sorted(pre_tokenizers.ByteLevel.alphabet())
                )
            },
        }
        tokenizer = Tokenizer(models.BPE(vocabulary, [], unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
            add_prefix_space=False
        )
    else:
        alphabet = sorted(
            set("Приветмирhello worldRUEN😀.!?0123456789abcxyz\n\u0301")
        )
        vocabulary = {
            **specials,
            **{
                character: index + 4
                for index, character in enumerate(alphabet)
            },
        }
        tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
    if remove_zero_width:
        tokenizer.normalizer = normalizers.Replace("\u200b", "")
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", 2), ("[SEP]", 3)],
    )
    return tokenizer


def refresh_manifest(directory, manifest=None):
    directory = Path(directory)
    path = directory / "manifest.json"
    if manifest is None:
        manifest = json.loads(path.read_text())
    for descriptor in manifest["files"].values():
        target = directory / descriptor["path"]
        descriptor.update(
            bytes=target.stat().st_size, sha256=file_sha256(target)
        )
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def build_bundle(
    directory,
    *,
    output_width=7,
    fixed_length=False,
    nonfinite=False,
    current_pooling=False,
):
    """Numeric test graph whose active-token sum ignores padding."""
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    settings = WindowSettings(64, 8, 8, 0, 1)
    length = settings.max_tokens if fixed_length else "sequence"
    inputs = [
        helper.make_tensor_value_info(
            name, TensorProto.INT64, ["batch", length]
        )
        for name in ("input_ids", "attention_mask", "token_type_ids")
    ]
    if current_pooling:
        inputs.append(
            helper.make_tensor_value_info(
                "current_token_mask", TensorProto.INT64, ["batch", length]
            )
        )
    weights = np.arange(1, output_width + 1, dtype=np.float32) / 1000
    bias = np.arange(output_width, dtype=np.float32) / 10
    if nonfinite:
        bias[0] = np.nan
    nodes = [
        helper.make_node(
            "Cast", ["input_ids"], ["ids_f"], to=TensorProto.FLOAT
        ),
        helper.make_node(
            "Cast", ["attention_mask"], ["mask_f"], to=TensorProto.FLOAT
        ),
        helper.make_node("Mul", ["ids_f", "mask_f"], ["active"]),
        helper.make_node(
            "ReduceSum", ["active", "axes"], ["total"], keepdims=1
        ),
        helper.make_node("Mul", ["total", "weights"], ["scaled"]),
        helper.make_node("Add", ["scaled", "bias"], ["logits"]),
    ]
    if current_pooling:
        nodes.insert(
            0,
            helper.make_node(
                "Mul", ["attention_mask", "current_token_mask"], ["pool_mask"]
            ),
        )
        nodes[2].input[0] = "pool_mask"
    tensors = [
        numpy_helper.from_array(np.array([1], dtype=np.int64), "axes"),
        numpy_helper.from_array(weights, "weights"),
        numpy_helper.from_array(bias, "bias"),
    ]
    graph = helper.make_graph(
        nodes,
        "untrained_test_fixture",
        inputs,
        [
            helper.make_tensor_value_info(
                "logits", TensorProto.FLOAT, ["batch", output_width]
            )
        ],
        initializer=tensors,
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 18)],
        ir_version=10,
        producer_name="untrained-test-fixture",
    )
    onnx.save_model(model, directory / "model.onnx")
    fixture_tokenizer().save(str(directory / "tokenizer.json"))
    (directory / "LICENSE.txt").write_text(
        "Apache-2.0. Numeric test fixture, not trained weights.\n"
    )
    manifest = {
        "schema_version": "model-bundle-0.1",
        "taxonomy_version": TAXONOMY_VERSION,
        "model_version": "untrained-numeric-fixture-v1",
        "purpose": "test_fixture",
        "task": "window_scores_v2" if current_pooling else "window_scores_v1",
        "labels": list(LABELS),
        "aggregation": "max_logit",
        "windowing": asdict(settings),
        "license": "Apache-2.0",
        "training_provenance": (
            {"pooling": "current_tokens"} if current_pooling else None
        ),
        "files": {
            role: {"path": name, "bytes": 0, "sha256": ""}
            for role, name in (
                ("graph", "model.onnx"),
                ("tokenizer", "tokenizer.json"),
                ("license", "LICENSE.txt"),
            )
        },
    }
    refresh_manifest(directory, manifest)
    return directory
