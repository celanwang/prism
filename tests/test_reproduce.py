"""Provenance, CLI, write boundaries and real process-group lifecycle checks."""

import hashlib
import importlib.util
import json
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from prism import cli
from prism import reproduce as repro
from prism.measure import measure
from prism.record import save_run

ROOT = Path(__file__).resolve().parents[1]


def fixture_inventory(tmp_path):
    (tmp_path / "artifacts").mkdir()
    raw = b"raw\r\n\x00\xff"
    (tmp_path / "artifact").write_bytes(raw)
    inventory = {
        "version": 1,
        "source_commit": "a" * 40,
        "files": [
            {"path": "artifact", "size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        ],
    }
    path = tmp_path / "artifacts/inventory-v1.json"
    path.write_text(json.dumps(inventory))
    return path, inventory


def independent_verifier():
    spec = importlib.util.spec_from_file_location(
        "verify_inventory", ROOT / "scripts/verify_inventory.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.verify


def test_inventory_raw_bytes_tampering_missing_and_independent(tmp_path):
    path, _ = fixture_inventory(tmp_path)
    for verify in (repro.verify_inventory, independent_verifier()):
        assert verify(tmp_path, path)["ok"]
    (tmp_path / "artifact").write_bytes(b"tampered")
    for verify in (repro.verify_inventory, independent_verifier()):
        result = verify(tmp_path, path)
        assert not result["ok"] and "mismatch" in result["errors"][0]
    (tmp_path / "artifact").rename(tmp_path / "absent")
    assert not repro.verify_inventory(tmp_path, path)["ok"]


@pytest.mark.parametrize("name", ["../outside", "/absolute", "a/../outside", "a\\b", "a//b"])
def test_unsafe_inventory_paths(tmp_path, name):
    path, inventory = fixture_inventory(tmp_path)
    inventory["files"][0]["path"] = name
    path.write_text(json.dumps(inventory))
    for verify in (repro.verify_inventory, independent_verifier()):
        with pytest.raises(ValueError):
            verify(tmp_path, path)


def test_inventory_rejects_duplicates_and_symlinks(tmp_path):
    path, inventory = fixture_inventory(tmp_path)
    inventory["files"] *= 2
    path.write_text(json.dumps(inventory))
    with pytest.raises(ValueError, match="duplicate"):
        repro.load_inventory(path)
    path.write_text(json.dumps({**inventory, "files": inventory["files"][:1]}))
    (tmp_path / "artifact").rename(tmp_path / "target")
    (tmp_path / "artifact").symlink_to(tmp_path / "target")
    assert not repro.verify_inventory(tmp_path, path)["ok"]


def test_inventory_hashes_are_frozen_git_blob_bytes():
    inventory = repro.load_inventory(ROOT / "artifacts/inventory-v1.json")
    # Independently batch-read every blob, including binary SUT and nested evidence.
    requests = "".join(entry["git_blob"] + "\n" for entry in inventory["files"])
    raw = subprocess.check_output(["git", "cat-file", "--batch"], input=requests.encode(), cwd=ROOT)
    offset = 0
    for entry in inventory["files"]:
        end = raw.index(b"\n", offset)
        oid, kind, length = raw[offset:end].decode().split()
        size = int(length)
        blob = raw[end + 1 : end + 1 + size]
        assert oid == entry["git_blob"] and kind == "blob"
        assert size == entry["size"]
        assert hashlib.sha256(blob).hexdigest() == entry["sha256"]
        offset = end + 2 + size
    tree = subprocess.check_output(
        [
            "git",
            "ls-tree",
            "-r",
            "--name-only",
            inventory["source_commit"],
            "data/sut",
            "runs",
            "results",
        ],
        cwd=ROOT,
        text=True,
    ).splitlines()
    committed = subprocess.check_output(
        ["git", "ls-tree", "-r", inventory["source_commit"]], cwd=ROOT, text=True
    ).splitlines()
    blob_by_path = {line.split("\t")[1]: line.split("\t")[0].split()[2] for line in committed}
    assert all(blob_by_path[e["path"]] == e["git_blob"] for e in inventory["files"])
    assert set(tree) <= {entry["path"] for entry in inventory["files"]}
    assert not any(".reproduce" in entry["path"] for entry in inventory["files"])


def test_topology_and_bad_graphs():
    recipes = {"c": {"depends": ["a", "b"]}, "b": {"depends": ["a"]}, "a": {"depends": []}}
    assert repro.topological_order(recipes, ["c", "b"]) == ["a", "b", "c"]
    with pytest.raises(ValueError, match="unknown"):
        repro.topological_order(recipes, ["missing"])
    recipes["a"]["depends"] = ["c"]
    with pytest.raises(ValueError, match="cycle"):
        repro.topological_order(recipes, ["c"])


def test_all_anchors_unique_and_sample_covers_every_phase():
    config = repro.load_recipes()
    phases = set()
    for recipe in config["recipes"].values():
        keys = [
            repro.record_key(repro.read_json(ROOT / p / "config.json")) for p in recipe["anchors"]
        ]
        assert len(keys) == len(set(keys))
    for name in config["sample"]:
        phases.update(
            repro.read_json(ROOT / p / "config.json")["phase"]
            for p in config["recipes"][name]["anchors"]
        )
    assert phases == {"P0", "P1", "P2", "P3", "P4"}
    assert len(config["recipes"]["schedsim"]["anchors"]) == 2


def test_dry_run_no_files_no_processes(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("dry-run performed a write/process/environment operation")

    monkeypatch.setattr(Path, "mkdir", forbidden)
    monkeypatch.setattr(Path, "write_text", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(repro, "snapshot", forbidden)
    monkeypatch.setattr("prism.env.collect_env", forbidden)
    config = repro.load_recipes()
    result = repro.reproduce(list(config["recipes"]), dry_run=True, output=tmp_path / "never")
    assert set(result["order"]) == set(config["recipes"])
    assert not (tmp_path / "never").exists()


@pytest.mark.parametrize(
    "argv",
    [
        ["reproduce", "--list"],
        ["reproduce", "--all", "--dry-run"],
        ["reproduce", "--sample", "--dry-run"],
        ["reproduce", "run", "parallel", "--dry-run"],
        ["verify"],
    ],
)
def test_cli_success(monkeypatch, capsys, argv):
    monkeypatch.setattr(sys, "argv", ["prism", *argv])
    cli.main()
    result = json.loads(capsys.readouterr().out)
    assert result.get("ok", True)


@pytest.mark.parametrize(
    "argv",
    [
        ["reproduce"],
        ["reproduce", "run"],
        ["reproduce", "run", "missing", "--dry-run"],
        ["reproduce", "--all", "run", "baseline"],
        ["reproduce", "--sample", "--all"],
        ["reproduce", "--all", "--timeout", "nan"],
    ],
)
def test_cli_invalid(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["prism", *argv])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code != 0


def test_verify_cli_fails_on_tampering(monkeypatch, capsys):
    monkeypatch.setattr(repro, "verify_inventory", lambda: {"ok": False, "errors": ["tampered"]})
    monkeypatch.setattr(sys, "argv", ["prism", "verify"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 1
    assert json.loads(capsys.readouterr().out)["errors"] == ["tampered"]


def test_output_boundary_and_no_overwrite(tmp_path):
    outside = tmp_path / "report.json"
    with pytest.raises(ValueError, match="-o"):
        repro.reproduce(["sut"], output=outside)
    assert not outside.exists()
    target = tmp_path / "existing"
    target.write_text("keep")
    with pytest.raises(FileExistsError):
        repro.write_json_new(target, {})
    assert target.read_text() == "keep"


def make_record(root, slug, config):
    measurement = measure(lambda: None, warmup=0, repeat=2, config=config)
    return save_run(root, slug, config, measurement)


def test_comparison_config_mismatch_env_drift_and_missing(tmp_path, monkeypatch):
    historical = tmp_path / "history"
    historical.mkdir()
    tree = tmp_path / "tree"
    anchor = make_record(historical, "arm", {"task": "T", "phase": "P0", "workload": "x", "v": 1})
    replay = make_record(
        tree / "runs", "arm", {"task": "T", "phase": "P0", "workload": "x", "v": 2}
    )
    recipe = {"anchors": [str(anchor.relative_to(tmp_path))]}
    result = repro.compare_records(tmp_path, tree, recipe, 0.092584)
    assert not result["ok"] and result["records"][0]["status"] == "config-mismatch"
    assert set(result["records"][0]["t4_annotation"]) == {"p50", "p95", "p99"}
    from prism.record import load_run

    def load(path):
        run = load_run(path)
        if path == replay:
            run.measurement.config_hash = load_run(anchor).measurement.config_hash
            run.measurement.env_fingerprint = "different-environment"
        return run

    monkeypatch.setattr("prism.record.load_run", load)
    result = repro.compare_records(tmp_path, tree, recipe, 0.092584)
    assert result["ok"] and result["records"][0]["env_equal"] is False
    assert not repro.compare_records(tmp_path, tmp_path / "empty", recipe, 0.092584)["ok"]


def test_new_derived_models_excluded_and_frozen_predictions_checked(tmp_path):
    config = {"task": "P1-T3", "phase": "P1", "mode": "模拟", "policy": "static"}
    anchor = make_record(tmp_path / "history", "before", config)
    tree = tmp_path / "tree"
    replay = make_record(tree / "runs", "before", config)
    for name in ("summary.json", "requests.json", "events.json", "queue.json"):
        (anchor / name).write_text('{"x": 1}')
        (replay / name).write_text('{"x": 1}')
    derived = make_record(tree / "runs", "after", {**config, "fit": "new"})
    result = repro.compare_records(
        tmp_path, tree, {"anchors": [str(anchor.relative_to(tmp_path))]}, 0.1
    )
    assert result["ok"]
    assert (
        next(r for r in result["records"] if r["run"].endswith(derived.name))["status"]
        == "new-derived"
    )
    (replay / "events.json").write_text('{"x": 2}')
    assert not repro.compare_records(
        tmp_path, tree, {"anchors": [str(anchor.relative_to(tmp_path))]}, 0.1
    )["ok"]


@pytest.mark.parametrize("mode", ["timeout", "cancel", "failure", "success"])
def test_process_group_cleanup_releases_descendant_port(tmp_path, mode):
    ready = tmp_path / "ready.json"
    # The descendant ignores TERM, so killing only the direct child leaks its port.
    server = (
        "import socket,signal,time,json,os;from pathlib import Path;"
        "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
        "s=socket.socket();s.bind(('127.0.0.1',0));s.listen();"
        f"Path({str(ready)!r}).write_text(json.dumps([s.getsockname()[1],os.getpid()]));"
        "time.sleep(60)"
    )
    child = "import subprocess,sys,time;" + f"subprocess.Popen([sys.executable,'-c',{server!r}]);"
    if mode in ("success", "failure"):
        child += f"time.sleep(.7);sys.exit({0 if mode == 'success' else 3})"
    else:
        child += "time.sleep(60)"
    runner = (
        "import sys,signal;from pathlib import Path;"
        f"sys.path.insert(0,{str(ROOT / 'src')!r});"
        "from prism.reproduce import run_process;"
        "signal.signal(signal.SIGTERM,lambda *args:(_ for _ in ()).throw(KeyboardInterrupt()));"
        f"run_process([sys.executable,'-c',{child!r}],Path({str(tmp_path)!r}),"
        f"Path({str(tmp_path / 'process.log')!r}),{3 if mode == 'timeout' else 15})"
    )
    with subprocess.Popen(
        [sys.executable, "-c", runner], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ) as process:
        deadline = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < deadline and process.poll() is None:
            time.sleep(0.02)
        assert ready.exists(), process.communicate(timeout=5)
        port, _ = json.loads(ready.read_text())
        if mode == "cancel":
            process.send_signal(signal.SIGTERM)
        _, stderr = process.communicate(timeout=20)
        assert (process.returncode == 0) == (mode == "success"), stderr.decode()
    deadline = time.monotonic() + 5
    while True:
        try:
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", port))
            break
        except OSError:
            if time.monotonic() > deadline:
                pytest.fail("descendant retained listening port after cleanup")
            time.sleep(0.02)


def test_failed_recipe_persists_report_and_stops_dependencies(tmp_path, monkeypatch):
    config = {
        "version": 1,
        "notes": [],
        "t4": {"resolution_ms": 0.092584},
        "frozen_inputs": [],
        "recipes": {
            "one": {"depends": [], "argv": ["unused"], "anchors": []},
            "two": {"depends": ["one"], "argv": ["unused"], "anchors": []},
        },
    }
    monkeypatch.setattr(repro, "load_recipes", lambda root: config)
    monkeypatch.setattr(repro, "verify_inventory", lambda root: {"ok": True})
    monkeypatch.setattr(repro, "load_inventory", lambda path: {})
    monkeypatch.setattr(repro, "snapshot", lambda root, tree, *args: tree.mkdir())
    calls = []

    def fail(argv, cwd, log, timeout):
        calls.append(cwd.name)
        raise subprocess.TimeoutExpired(argv, timeout)

    monkeypatch.setattr(repro, "run_process", fail)
    output = tmp_path / "runs/.reproduce/output.json"
    result = repro.reproduce(["two"], root=tmp_path, timeout=0.01, output=output)
    assert not result["ok"] and calls == ["one"]
    assert "TimeoutExpired" in result["recipes"][0]["error"]
    assert json.loads(output.read_text()) == result
    assert json.loads((Path(result["session"]) / "report.json").read_text()) == result
    with pytest.raises(FileExistsError):
        repro.reproduce(["two"], root=tmp_path, output=output)
    assert calls == ["one"]
