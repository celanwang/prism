"""Independent goldens and integrity failures for the NumPy reference decoder."""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from prism.sut import DEFAULT_SUT_PATH, SUT, ByteBPE, array_digest, check_sut, load_sut


@pytest.fixture(scope="module")
def sut() -> SUT:
    return load_sut(DEFAULT_SUT_PATH)


@pytest.fixture
def artifacts(tmp_path: Path) -> Path:
    for name in (
        "MANIFEST.json",
        "model-42.npz",
        "tokenizer.json",
        "reference_greedy.json",
        "reference_logits.npz",
    ):
        shutil.copyfile(DEFAULT_SUT_PATH / name, tmp_path / name)
    return tmp_path


def _read_manifest(path: Path) -> dict[str, Any]:
    return json.loads((path / "MANIFEST.json").read_text(encoding="utf-8"))


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    (path / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_array_digest_exact_header_and_contiguous_bytes() -> None:
    # Literal JSON header preserves kaleido's default spaces and dtype.str endianness.
    array = np.array([[1, 2], [3, 4]], dtype="<i2")
    expected = hashlib.sha256(b'["<i2", [2, 2]]\0\x01\0\x02\0\x03\0\x04\0').hexdigest()
    assert array_digest(array) == expected
    assert array_digest(np.asfortranarray(array)) == expected
    assert array_digest(array[:, ::-1]) == array_digest(array[:, ::-1].copy())
    assert array_digest(array.reshape(4)) != expected
    assert array_digest(array.view("<u2")) != expected
    assert array_digest(array.astype(">i2")) != expected
    assert (
        array_digest(array, array)
        == hashlib.sha256((b'["<i2", [2, 2]]\0' + array.tobytes()) * 2).hexdigest()
    )
    # ascontiguousarray turns a scalar into shape (1,), as in kaleido.
    assert array_digest(np.array(1, dtype="<i2")) == array_digest(np.array([1], dtype="<i2"))
    with pytest.raises(TypeError, match="object arrays"):
        array_digest(np.array([object()]))


def test_manifest_provenance_and_artifact_hashes(sut: SUT) -> None:
    manifest = _read_manifest(DEFAULT_SUT_PATH)
    assert manifest["kaleido_artifact_revision"] == "cee0ef8f2e3f64b9d9504da864a088f767dfe02d"
    assert len(manifest["kaleido_head_at_copy"]) == 40
    assert manifest["config"] == vars(sut.config)
    for name, source in manifest["source_files"].items():
        assert Path(source["source_path"]).is_absolute()
        assert len(source["source_sha256"]) == 64
        assert source["kaleido_head_at_copy"] == manifest["kaleido_head_at_copy"]
        assert source["kaleido_artifact_revision"] == manifest["kaleido_artifact_revision"]
        assert source["copied_at"] == manifest["copied_at"]
        if name in ("model-42.npz", "tokenizer.json"):
            assert (
                hashlib.sha256((DEFAULT_SUT_PATH / name).read_bytes()).hexdigest()
                == source["source_sha256"]
            )
    assert len(sut.arrays) == len(manifest["arrays"]) == 28
    for name, array in sut.arrays.items():
        assert array_digest(array) == manifest["arrays"][name]
        assert not array.flags.writeable


@pytest.mark.parametrize("change", ["value", "shape", "dtype", "missing", "extra"])
def test_load_rejects_array_tampering(artifacts: Path, change: str) -> None:
    path = artifacts / "model-42.npz"
    with np.load(path, allow_pickle=False) as archive:
        arrays = dict(archive)
    name = "token_embedding.weight"
    if change == "value":
        arrays[name][0, 0] += 1
    elif change == "shape":
        arrays[name] = arrays[name].reshape(-1)
    elif change == "dtype":
        arrays[name] = arrays[name].view(np.int32)
    elif change == "missing":
        arrays.pop(name)
    else:
        arrays["unexpected"] = np.zeros(1, dtype=np.float32)
    np.savez_compressed(path, **arrays)
    # Update only the container hash to ensure the array check itself rejects corruption.
    manifest = _read_manifest(artifacts)
    manifest["source_files"]["model-42.npz"]["source_sha256"] = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    _write_manifest(artifacts, manifest)
    with pytest.raises(ValueError, match="checkpoint array"):
        load_sut(artifacts)


def test_load_rejects_tokenizer_tampering(artifacts: Path) -> None:
    with (artifacts / "tokenizer.json").open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="file sha256 mismatch: tokenizer.json"):
        load_sut(artifacts)


def test_load_rejects_configuration_mismatch(artifacts: Path) -> None:
    manifest = _read_manifest(artifacts)
    manifest["config"]["layers"] += 1
    _write_manifest(artifacts, manifest)
    with pytest.raises(ValueError, match="configured architecture"):
        load_sut(artifacts)


@pytest.mark.parametrize("name", ["within_window", "window_rollover"])
def test_complete_greedy_sequences_match_independent_fixture(sut: SUT, name: str) -> None:
    fixture = json.loads((DEFAULT_SUT_PATH / "reference_greedy.json").read_text())
    case = fixture["cases"][name]
    original = list(case["prompt_ids"])
    actual = sut.generate(
        case["prompt_ids"], max_new_tokens=case["max_new_tokens"], eos_id=case["eos_id"]
    )
    assert actual == case["token_ids"]
    assert len(actual) == len(original) + case["max_new_tokens"]
    assert case["prompt_ids"] == original


def test_logits_argmax_and_consistency_report() -> None:
    report = check_sut()
    assert report["passed"]
    assert report["greedy_cases_passed"] == 2
    assert report["logits_argmax_positions_passed"] == 32
    assert report["tokenizer_prompts_passed"] == 3
    assert np.isfinite(report["logits_max_abs_error"])
    assert report["logits_max_abs_error"] >= 0  # Record error; deliberately no accuracy threshold.


def test_logit_error_is_not_a_pass_threshold(artifacts: Path) -> None:
    path = artifacts / "reference_logits.npz"
    with np.load(path, allow_pickle=False) as reference:
        tokens = reference["token_ids"]
        logits = reference["logits"] + np.float32(100)
    np.savez_compressed(path, token_ids=tokens, logits=logits)
    manifest = _read_manifest(artifacts)
    manifest["reference_sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    _write_manifest(artifacts, manifest)
    report = check_sut(artifacts)
    assert report["passed"]
    assert report["logits_max_abs_error"] > 99


def test_causal_attention_batching_and_float32(sut: SUT) -> None:
    tokens = np.array([[256, 316, 32, 396], [256, 320, 32, 460]], dtype=np.int64)
    logits = sut.forward(tokens)
    assert logits.shape == (2, 4, sut.config.vocab_size)
    assert logits.dtype == np.float32
    for batch in range(2):
        np.testing.assert_allclose(
            logits[batch], sut.forward(tokens[batch : batch + 1])[0], atol=2e-5, rtol=2e-5
        )
    changed = tokens.copy()
    changed[:, -1] = 1
    np.testing.assert_array_equal(logits[:, :-1], sut.forward(changed)[:, :-1])


@pytest.mark.parametrize(
    "tokens",
    [
        np.array([1], dtype=np.int64),
        np.array([[1]], dtype=np.int32),
        np.empty((1, 0), dtype=np.int64),
        np.empty((0, 1), dtype=np.int64),
        np.array([[-1]], dtype=np.int64),
        np.array([[512]], dtype=np.int64),
        np.zeros((1, 129), dtype=np.int64),
    ],
)
def test_forward_rejects_invalid_inputs(sut: SUT, tokens: np.ndarray) -> None:
    with pytest.raises(ValueError):
        sut.forward(tokens)


def test_generation_eos_and_zero_length(sut: SUT) -> None:
    prompt = [256, 316]
    assert sut.generate(prompt, max_new_tokens=0) == prompt
    following = sut.generate(prompt, max_new_tokens=1)[-1]
    assert sut.generate(prompt, max_new_tokens=5, eos_id=following) == [*prompt, following]
    with pytest.raises(ValueError, match="prompt"):
        sut.generate([], max_new_tokens=1)
    with pytest.raises(ValueError, match="max_new_tokens"):
        sut.generate(prompt, max_new_tokens=-1)
    with pytest.raises(ValueError, match="EOS"):
        sut.generate(prompt, max_new_tokens=1, eos_id=512)


def test_tokenizer_sampling_prompts_and_unicode(sut: SUT) -> None:
    for sample in _read_manifest(DEFAULT_SUT_PATH)["tokenizer_checks"]:
        assert [256] + sut.tokenizer.encode(sample["prompt"]) == sample["prompt_ids"]
    for text in ("", "Hello, 世界！\n\n café", "a   b\tword", "🙂 “Alice”"):
        ids = sut.tokenizer.encode(text, add_eos=True)
        assert ids[-1] == 256
        assert sut.tokenizer.decode(ids) == text
    for invalid in ([True], [-1], [512]):
        with pytest.raises(ValueError, match="token ID"):
            sut.tokenizer.decode(invalid)


def test_tokenizer_rejects_invalid_merges() -> None:
    value = json.loads((DEFAULT_SUT_PATH / "tokenizer.json").read_text())
    value["merges"][0] = [256, 0]
    with pytest.raises(ValueError, match="BPE merge"):
        ByteBPE(value)


def test_check_rejects_wrong_greedy_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(SUT, "generate", lambda self, prompt, **kwargs: prompt)
    with pytest.raises(ValueError, match="greedy token sequence mismatch"):
        check_sut()


def test_check_rejects_wrong_generation_position_argmax(monkeypatch: pytest.MonkeyPatch) -> None:
    forward = SUT.forward

    def corrupted_forward(self: SUT, tokens: np.ndarray) -> np.ndarray:
        logits = forward(self, tokens)
        if tokens.shape[1] == 50:
            logits[0, 17, 0] = np.finfo(np.float32).max
        return logits

    monkeypatch.setattr(SUT, "forward", corrupted_forward)
    with pytest.raises(ValueError, match="generated-position logits argmax mismatch"):
        check_sut()


def test_sut_cli_from_other_directory(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "prism", "sut", "check"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(result.stdout)
    assert report["passed"]
    assert report["greedy_cases_passed"] == 2
    assert report["logits_argmax_positions_passed"] == 32


def test_sut_cli_fails_on_inconsistent_fixture(artifacts: Path) -> None:
    path = artifacts / "reference_greedy.json"
    fixture = json.loads(path.read_text())
    fixture["cases"]["within_window"]["token_ids"][-1] = 0
    path.write_text(json.dumps(fixture))
    manifest = _read_manifest(artifacts)
    manifest["reference_sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    _write_manifest(artifacts, manifest)
    result = subprocess.run(
        [sys.executable, "-m", "prism", "sut", "check", "--path", str(artifacts)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "greedy token sequence mismatch" in result.stderr
    assert not result.stdout
