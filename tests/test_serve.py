"""Independent deployment gates and event-controlled concurrency, without speed targets."""

import http.client
import importlib.util
import json
import random
import socket
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from prism.serve import InferenceService, _Job, stream_tokens
from prism.sut import DEFAULT_SUT_PATH, SUT, load_sut


@pytest.fixture(scope="module")
def sut() -> SUT:
    return load_sut(DEFAULT_SUT_PATH)


def request(
    service: InferenceService, body: Any = None, *, path: str = "/generate", method: str = "POST"
) -> tuple[int, Any]:
    connection = http.client.HTTPConnection("127.0.0.1", service.port, timeout=15)
    try:
        connection.request(method, path, json.dumps(body), {"Content-Type": "application/json"})
        response = connection.getresponse()
        return response.status, json.loads(response.read())
    finally:
        connection.close()


def events(response: http.client.HTTPResponse) -> list[tuple[str, dict[str, Any]]]:
    result = []
    while line := response.readline():
        if line.startswith(b"event: "):
            event = line.decode().strip().split(": ", 1)[1]
            data = json.loads(response.readline().decode().removeprefix("data: "))
            assert response.readline() == b"\n"
            result.append((event, data))
    return result


def streamed(service: InferenceService, body: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    connection = http.client.HTTPConnection("127.0.0.1", service.port, timeout=15)
    try:
        connection.request(
            "POST", "/generate/stream", json.dumps(body), {"Content-Type": "application/json"}
        )
        response = connection.getresponse()
        assert response.status == 200
        assert response.getheader("Content-Type") == "text/event-stream"
        return events(response)
    finally:
        connection.close()


def test_fixture_and_random_deployment_gate(sut: SUT) -> None:
    fixture = json.loads((DEFAULT_SUT_PATH / "reference_greedy.json").read_text())["cases"]
    rng = random.Random(20261019)
    cases = [
        (case["prompt_ids"], case["max_new_tokens"], case["eos_id"], case["token_ids"])
        for case in fixture.values()
    ]
    # Include initial cropping, a rollover and multiple cache rebuilds.
    cases += [
        ([rng.randrange(sut.config.vocab_size) for _ in range(length)], count, None, None)
        for length, count in ((1, 5), (17, 9), (127, 5), (140, 3), (4, 260))
    ]
    with InferenceService(sut, max_new_tokens=300) as service:
        for prompt, count, eos, golden in cases:
            original = list(prompt)
            baseline = sut.generate(prompt, max_new_tokens=count, eos_id=eos, use_cache=True)
            if golden is not None:
                assert baseline == golden
            # Independent local baseline; never derive the oracle from the adapter.
            assert prompt + list(stream_tokens(sut, prompt, max_new_tokens=count, eos_id=eos)) == (
                baseline
            )
            body = {"prompt": prompt, "max_new_tokens": count, "eos_id": eos}
            status, whole = request(service, body)
            stream = streamed(service, body)
            assert status == 200
            assert stream[0][0] == "meta"
            assert stream[0][1]["prompt_ids"] == prompt
            assert stream[-1][0] == "done"
            assert whole["token_ids"] == stream[-1][1]["token_ids"] == baseline
            assert prompt + [d["token_id"] for e, d in stream if e == "token"] == baseline
            assert [d["index"] for e, d in stream if e == "token"] == list(
                range(len(prompt), len(baseline))
            )
            assert prompt == original


def test_text_zero_and_eos(sut: SUT) -> None:
    with InferenceService(sut) as service:
        for text in ("", "Hello, 世界！ café", "Alice"):
            prompt = [sut.tokenizer.eos_id] + sut.tokenizer.encode(text)
            following = sut.generate(prompt, max_new_tokens=1, use_cache=True)[-1]
            for count, eos in ((0, None), (5, following), (5, None)):
                body = {"prompt": text, "max_new_tokens": count, "eos_id": eos}
                expected = sut.generate(prompt, max_new_tokens=count, eos_id=eos, use_cache=True)
                assert request(service, body)[1]["token_ids"] == expected
                assert streamed(service, body)[-1][1]["token_ids"] == expected


def test_first_token_arrives_before_second_forward(
    sut: SUT, monkeypatch: pytest.MonkeyPatch
) -> None:
    import prism.serve as module

    original = module._forward_cached
    second = threading.Event()
    release = threading.Event()
    calls = 0

    def gated(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 2:
            second.set()
            assert release.wait(15), "client did not receive token before second forward"
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "_forward_cached", gated)
    with InferenceService(sut) as service:
        connection = http.client.HTTPConnection("127.0.0.1", service.port, timeout=15)
        try:
            connection.request(
                "POST",
                "/generate/stream",
                json.dumps({"prompt": [256], "max_new_tokens": 2}),
                {"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            assert second.wait(15)
            # Second forward is blocked. A buffered/pseudo-stream cannot pass this read.
            assert response.readline() == b"event: meta\n"
            response.readline()
            response.readline()
            assert response.readline() == b"event: token\n"
            first = json.loads(response.readline().decode().removeprefix("data: "))
            assert first["index"] == 1
            response.readline()
            release.set()
            tail = events(response)
            assert [e for e, _ in tail] == ["token", "done"]
        finally:
            release.set()
            connection.close()


def test_fifo_serialization_and_429(sut: SUT, monkeypatch: pytest.MonkeyPatch) -> None:
    entered = threading.Event()
    release = threading.Event()
    queued = threading.Event()
    order = []
    original = SUT.generate

    def gated(self: SUT, prompt: list[int], **kwargs: Any) -> list[int]:
        order.append(prompt[0])
        if prompt == [1]:
            entered.set()
            assert release.wait(15)
        return original(self, prompt, **kwargs)

    monkeypatch.setattr(SUT, "generate", gated)
    with InferenceService(sut, queue_capacity=1) as service, ThreadPoolExecutor(3) as clients:
        submit = service.submit

        def observed(job: _Job) -> int:
            status = submit(job)
            if job.prompt == [2] and status == 200:
                queued.set()
            return status

        monkeypatch.setattr(service, "submit", observed)
        first = clients.submit(request, service, {"prompt": [1], "max_new_tokens": 2})
        try:
            assert entered.wait(15)
            second = clients.submit(request, service, {"prompt": [2], "max_new_tokens": 2})
            assert queued.wait(15)
            assert request(service, {"prompt": [3], "max_new_tokens": 2})[0] == 429
            assert (
                request(service, {"prompt": [3], "max_new_tokens": 2}, path="/generate/stream")[0]
                == 429
            )
            assert order == [1]
        finally:
            release.set()
        assert first.result()[0] == second.result()[0] == 200
        assert order == [1, 2]
        assert not service.worker.daemon


def test_concurrent_mixed_requests_keep_independent_sequences(sut: SUT) -> None:
    prompts = [[256, 10 + index, 32] for index in range(8)]
    baselines = [sut.generate(prompt, max_new_tokens=4, use_cache=True) for prompt in prompts]
    barrier = threading.Barrier(8)
    with InferenceService(sut, queue_capacity=16) as service:

        def run(index: int) -> list[int]:
            barrier.wait(timeout=15)
            body = {"prompt": prompts[index], "max_new_tokens": 4}
            if index % 2:
                return streamed(service, body)[-1][1]["token_ids"]
            status, data = request(service, body)
            assert status == 200
            return data["token_ids"]

        with ThreadPoolExecutor(8) as clients:
            assert list(clients.map(run, range(8))) == baselines


def test_bounded_http_pool_returns_503(sut: SUT, monkeypatch: pytest.MonkeyPatch) -> None:
    entered = threading.Event()
    release = threading.Event()
    original = SUT.generate

    def gated(self: SUT, prompt: list[int], **kwargs: Any) -> list[int]:
        entered.set()
        assert release.wait(15)
        return original(self, prompt, **kwargs)

    monkeypatch.setattr(SUT, "generate", gated)
    with InferenceService(sut, http_workers=1, http_pending=1) as service:
        with ThreadPoolExecutor(1) as client:
            future = client.submit(request, service, {"prompt": [1], "max_new_tokens": 1})
            assert entered.wait(15)
            pending = socket.create_connection(("127.0.0.1", service.port), timeout=15)
            try:
                # Observe acceptance of the pending socket without timing-based assertions.
                accepted = threading.Event()
                old_submit = service.pool.submit

                def observe(*args: Any, **kwargs: Any) -> Any:
                    result = old_submit(*args, **kwargs)
                    accepted.set()
                    return result

                monkeypatch.setattr(service.pool, "submit", observe)
                pending.sendall(b"GET /healthz HTTP/1.0\r\n\r\n")
                # The TCP connection can already have been accepted before installing observe;
                # use a lock-protected socket inventory as the deterministic acceptance signal.
                with service.lock:
                    already = len(service.sockets) == 2
                if not already:
                    assert accepted.wait(15)
                assert request(service, {"prompt": [2], "max_new_tokens": 1})[0] == 503
                assert service.queue.empty()  # HTTP saturation is distinct from inference-full.
            finally:
                release.set()
                pending.close()
            assert future.result()[0] == 200


@pytest.mark.parametrize(
    "body",
    [
        None,
        [],
        {},
        {"prompt": [], "max_new_tokens": 1},
        {"prompt": [True], "max_new_tokens": 1},
        {"prompt": [-1], "max_new_tokens": 1},
        {"prompt": [512], "max_new_tokens": 1},
        {"prompt": [1], "max_new_tokens": True},
        {"prompt": [1], "max_new_tokens": -1},
        {"prompt": [1], "max_new_tokens": 257},
        {"prompt": [1], "max_new_tokens": 1, "eos_id": False},
        {"prompt": [1], "max_new_tokens": 1, "eos_id": 512},
        {"prompt": [1], "max_new_tokens": 1, "extra": 1},
    ],
)
def test_invalid_schema(sut: SUT, body: Any) -> None:
    with InferenceService(sut) as service:
        status, data = request(service, body)
        assert status == 400 and "error" in data


def test_http_errors_health_and_limits(sut: SUT) -> None:
    with InferenceService(sut, max_body_bytes=80) as service:
        assert request(service, method="GET", path="/healthz")[1] == {"status": "ok"}
        assert request(service, path="/unknown")[0] == 404
        assert request(service, path="/unknown", method="GET")[0] == 404
        assert request(service, method="PUT")[0] == 501
        assert request(service, {"prompt": "x" * 100, "max_new_tokens": 0})[0] == 413
        for body, headers in (
            (b"{", {"Content-Type": "application/json"}),
            (b"\xff", {"Content-Type": "application/json"}),
            (b"{}", {"Content-Type": "text/plain"}),
            (b"{}", {"Transfer-Encoding": "chunked"}),
            (b"{}", {"Content-Length": "-1"}),
        ):
            connection = http.client.HTTPConnection("127.0.0.1", service.port, timeout=15)
            try:
                connection.request("POST", "/generate", body, headers)
                response = connection.getresponse()
                assert response.status == 400
                assert "error" in json.loads(response.read())
            finally:
                connection.close()


def test_worker_failure_is_json_or_sse_and_worker_recovers(
    sut: SUT, monkeypatch: pytest.MonkeyPatch
) -> None:
    import prism.serve as module

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("private diagnostics must not leak")

    with InferenceService(sut) as service:
        with monkeypatch.context() as patch:
            patch.setattr(SUT, "generate", fail)
            assert request(service, {"prompt": [1], "max_new_tokens": 1}) == (
                500,
                {"error": "inference failed"},
            )
            patch.setattr(module, "_forward_cached", fail)
            result = streamed(service, {"prompt": [1], "max_new_tokens": 1})
            assert [e for e, _ in result] == ["meta", "error"]
        assert request(service, {"prompt": [1], "max_new_tokens": 1})[0] == 200


def test_close_cancels_blocked_stream_and_releases_port(sut: SUT) -> None:
    service = InferenceService(sut).start()
    port = service.port
    job = _Job([1], 256, None, True)
    assert service.submit(job) == 200
    service.close()
    service.close()
    assert job.cancelled.is_set()
    assert not service.worker.is_alive() and not service.accept.is_alive()
    assert not service.jobs and not service.sockets
    assert service.submit(_Job([1], 0, None, False)) == 503
    with InferenceService(sut, port=port) as rebound:
        assert request(rebound, method="GET", path="/healthz")[0] == 200
    unstarted = InferenceService(sut, port=port)
    unstarted.close()
    with pytest.raises(RuntimeError):
        unstarted.start()


def test_disconnect_does_not_poison_next_request(sut: SUT) -> None:
    with InferenceService(sut) as service:
        connection = socket.create_connection(("127.0.0.1", service.port), timeout=15)
        body = json.dumps({"prompt": [1], "max_new_tokens": 32}).encode()
        connection.sendall(
            b"POST /generate/stream HTTP/1.0\r\nContent-Type: application/json\r\n"
            + f"Content-Length: {len(body)}\r\n\r\n".encode()
            + body
        )
        connection.close()
        assert request(service, {"prompt": [2], "max_new_tokens": 1})[0] == 200


def test_benchmark_check_artifacts(tmp_path: Path) -> None:
    from prism.record import load_run

    spec = importlib.util.spec_from_file_location(
        "serve_bench", Path(__file__).resolve().parents[1] / "scripts/serve_bench.py"
    )
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    recipe = json.loads(script.CONFIG_PATH.read_text())
    result = script.run_benchmark(recipe, tmp_path, check=True)
    assert result["equivalence"]["passed"]
    assert {r["concurrency"] for r in result["rows"]} == {1, 2, 4, 8}
    assert {r["endpoint"] for r in result["rows"]} == {"whole", "stream"}
    for row in result["rows"]:
        run = load_run(row["path"])
        assert len(run.measurement.samples_ms) == 2
        assert "非实验证据" in run.config["evidence"]
        details = json.loads((run.path / "requests.json").read_text())
        assert len(details) == recipe["check_requests"] * 2
        assert all(d["status"] in (200, 429) for d in details)
        sim = load_run(row["simulation_path"])
        assert sim.config["batch_size"] == 1 and sim.config["batch_timeout_ms"] == 0
        assert sim.config["queue_capacity"] == recipe["queue_capacity"]
        assert sim.config["mode"] == "模拟"
