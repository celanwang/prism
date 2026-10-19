"""Loopback-only HTTP inference with bounded admission and one inference worker.

Run: python -m prism.serve --port 8080 --queue-capacity 4 --http-workers 16.
One accept thread dispatches sockets to a fixed HTTP ThreadPoolExecutor. A semaphore
bounds running plus pending HTTP handlers; exhaustion/closing returns JSON 503.
Handlers validate bodies and submit FIFO jobs to a bounded inference queue (429
when full, excluding the active job). Exactly one worker owns all model calls and
request-local KV caches. HTTP handlers alone write sockets. Streaming uses a bounded
event channel; slow readers backpressure inference, and disconnect/close cancels
work. Socket timeouts bound stalled clients. close() cancels jobs, closes sockets,
drains both pools and releases the port; call it outside the accept thread.
Only stdlib networking/concurrency is used; NumPy remains the existing SUT backend.
"""

from __future__ import annotations

import argparse
import json
import queue
import socket
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Self

import numpy as np

from prism.kvcache import _forward_cached
from prism.sut import DEFAULT_SUT_PATH, SUT, load_sut


def stream_tokens(
    sut: SUT, prompt: list[int], *, max_new_tokens: int, eos_id: int | None = None
) -> Iterator[int]:
    """Yield greedy tokens immediately, rebuilding renumbered windows at rollover.

    Inputs are validated at the HTTP boundary. Independent local generate tests
    constrain this adapter's use of the private cached-forward API.
    """
    tokens = list(prompt)
    cache = None
    for _ in range(max_new_tokens):
        if cache is None or cache[0][0].shape[2] >= sut.config.context_length:
            cache = None
            logits, cache = _forward_cached(
                sut, np.array([tokens[-sut.config.context_length :]], dtype=np.int64)
            )
        else:
            logits, cache = _forward_cached(sut, np.array([[tokens[-1]]], dtype=np.int64), cache)
        following = int(logits[0, -1].argmax())
        tokens.append(following)
        yield following
        if following == eos_id:
            break


@dataclass(eq=False)
class _Job:
    prompt: list[int]
    max_new_tokens: int
    eos_id: int | None
    streaming: bool
    submitted: float = field(default_factory=time.perf_counter)
    events: queue.Queue[tuple[str, dict[str, Any]]] = field(
        default_factory=lambda: queue.Queue(maxsize=8)
    )
    cancelled: threading.Event = field(default_factory=threading.Event)

    def emit(self, event: str, data: dict[str, Any]) -> None:
        while not self.cancelled.is_set():
            try:
                self.events.put((event, data), timeout=0.05)
                return
            except queue.Full:
                pass


class _HTTPServer(HTTPServer):
    allow_reuse_address = True

    def __init__(self, service: InferenceService, port: int) -> None:
        self.service = service
        # Keep the kernel listen backlog from throttling the configured client waves
        # before they reach the explicit HTTP admission bound.
        self.request_queue_size = service.http_capacity
        super().__init__(("127.0.0.1", port), _Handler)

    def process_request(self, request: socket.socket, client_address: Any) -> None:
        service = self.service
        if service.closing.is_set() or not service.slots.acquire(blocking=False):
            body = b'{"error":"HTTP capacity exhausted or service closing"}'
            try:
                request.settimeout(0.2)
                request.sendall(
                    b"HTTP/1.0 503 Service Unavailable\r\nContent-Type: application/json\r\n"
                    + f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode()
                    + body
                )
            except OSError:
                pass
            finally:
                self.shutdown_request(request)
            return
        with service.lock:
            service.sockets.add(request)
        service.pool.submit(self._handle, request, client_address)

    def _handle(self, request: socket.socket, client_address: Any) -> None:
        try:
            request.settimeout(self.service.socket_timeout)
            self.finish_request(request, client_address)
        except OSError:
            pass
        finally:
            self.shutdown_request(request)
            with self.service.lock:
                self.service.sockets.discard(request)
            self.service.slots.release()


class InferenceService:
    """Own an explicit start/close lifecycle; port zero selects an ephemeral port."""

    def __init__(
        self,
        sut: SUT,
        *,
        port: int = 0,
        queue_capacity: int = 4,
        http_workers: int = 16,
        http_pending: int = 0,
        max_body_bytes: int = 65536,
        max_new_tokens: int = 256,
        socket_timeout: float = 10.0,
    ) -> None:
        for name, value, minimum in (
            ("port", port, 0),
            ("queue_capacity", queue_capacity, 1),
            ("http_workers", http_workers, 1),
            ("http_pending", http_pending, 0),
            ("max_body_bytes", max_body_bytes, 1),
            ("max_new_tokens", max_new_tokens, 0),
        ):
            if type(value) is not int or value < minimum:
                raise ValueError(f"invalid {name}")
        if port > 65535 or not 0 < socket_timeout < float("inf"):
            raise ValueError("invalid port or socket_timeout")
        self.sut = sut
        self.max_body_bytes = max_body_bytes
        self.max_new_tokens = max_new_tokens
        self.socket_timeout = socket_timeout
        self.lock = threading.Lock()
        self.closing = threading.Event()
        self.jobs: set[_Job] = set()
        self.sockets: set[socket.socket] = set()
        self.queue: queue.Queue[_Job | None] = queue.Queue(maxsize=queue_capacity)
        self.http_capacity = http_workers + http_pending
        self.slots = threading.BoundedSemaphore(self.http_capacity)
        self.server = _HTTPServer(self, port)
        self.pool = ThreadPoolExecutor(max_workers=http_workers, thread_name_prefix="prism-http")
        self.worker = threading.Thread(target=self._work, name="prism-inference")
        self.accept = threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.05}, name="prism-accept"
        )
        self.started = False

    @property
    def port(self) -> int:
        """Return the bound loopback port."""
        return self.server.server_port

    def start(self) -> Self:
        """Start once, returning this service for context-manager use."""
        if self.started or self.closing.is_set():
            raise RuntimeError("service already started or closed")
        self.started = True
        self.worker.start()
        self.accept.start()
        return self

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(self, *args: Any) -> None:
        self.close()

    def close(self) -> None:
        """Cancel pending/active work and join all owned threads before returning."""
        with self.lock:
            if self.closing.is_set():
                return
            self.closing.set()
            for job in self.jobs:
                job.cancelled.set()
        if self.started:
            self.server.shutdown()
            self.accept.join()
        with self.lock:
            for connection in self.sockets:
                try:
                    connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
        self.pool.shutdown(wait=True)
        if self.started:
            self.queue.put(None)
            self.worker.join()
        self.server.server_close()

    def submit(self, job: _Job) -> int:
        """Atomically admit a job, returning 200, inference-full 429 or closing 503."""
        with self.lock:
            if self.closing.is_set():
                return 503
            try:
                self.queue.put_nowait(job)
            except queue.Full:
                return 429
            self.jobs.add(job)
        return 200

    def _work(self) -> None:
        while (job := self.queue.get()) is not None:
            try:
                if job.cancelled.is_set():
                    continue
                started = time.perf_counter()
                waiting = (started - job.submitted) * 1000
                tokens = list(job.prompt)
                if job.streaming:
                    job.emit("meta", {"prompt_ids": job.prompt, "queue_wait_ms": waiting})
                    for token in stream_tokens(
                        self.sut,
                        job.prompt,
                        max_new_tokens=job.max_new_tokens,
                        eos_id=job.eos_id,
                    ):
                        if job.cancelled.is_set():
                            break
                        tokens.append(token)
                        job.emit("token", {"token_id": token, "index": len(tokens) - 1})
                else:
                    tokens = self.sut.generate(
                        job.prompt,
                        max_new_tokens=job.max_new_tokens,
                        eos_id=job.eos_id,
                        use_cache=True,
                    )
                job.emit(
                    "done",
                    {
                        "token_ids": tokens,
                        "queue_wait_ms": waiting,
                        "inference_ms": (time.perf_counter() - started) * 1000,
                    },
                )
            except Exception:
                job.emit("error", {"error": "inference failed"})
            finally:
                with self.lock:
                    self.jobs.discard(job)


class _Handler(BaseHTTPRequestHandler):
    server: _HTTPServer

    def log_message(self, format: str, *args: Any) -> None:
        """Keep benchmark stdout machine-readable."""

    def _json(self, status: int, data: dict[str, Any]) -> None:
        body = json.dumps(data, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def send_error(self, code: int, message: str | None = None, explain: str | None = None) -> None:
        """Use JSON for parser and unsupported-method errors too."""
        self._json(code, {"error": message or self.responses.get(code, ("error",))[0]})

    def do_GET(self) -> None:
        if self.path != "/healthz":
            self._json(404, {"error": "unknown endpoint"})
        else:
            closing = self.server.service.closing.is_set()
            self._json(503 if closing else 200, {"status": "closing" if closing else "ok"})

    def _parse(self) -> _Job:
        service = self.server.service
        if self.headers.get("Transfer-Encoding") is not None:
            raise ValueError("Transfer-Encoding unsupported")
        lengths = self.headers.get_all("Content-Length", [])
        if len(lengths) != 1 or not lengths[0].isascii() or not lengths[0].isdecimal():
            raise ValueError("one nonnegative Content-Length required")
        length = int(lengths[0])
        if length > service.max_body_bytes:
            raise OverflowError("request body too large")
        if self.headers.get_content_type() != "application/json":
            raise ValueError("Content-Type must be application/json")
        body = self.rfile.read(length)
        if len(body) != length:
            raise ValueError("incomplete body")
        data = json.loads(body)
        if not isinstance(data, dict) or set(data) - {"prompt", "max_new_tokens", "eos_id"}:
            raise ValueError("expected prompt, max_new_tokens and optional eos_id")
        prompt = data.get("prompt")
        if isinstance(prompt, str):
            prompt = [service.sut.tokenizer.eos_id] + service.sut.tokenizer.encode(prompt)
        if (
            not isinstance(prompt, list)
            or not prompt
            or any(type(t) is not int or not 0 <= t < service.sut.config.vocab_size for t in prompt)
        ):
            raise ValueError("prompt must be a string or nonempty valid ID list")
        count = data.get("max_new_tokens")
        if type(count) is not int or not 0 <= count <= service.max_new_tokens:
            raise ValueError("max_new_tokens outside configured output limit")
        eos = data.get("eos_id")
        if eos is not None and (
            type(eos) is not int or not 0 <= eos < service.sut.config.vocab_size
        ):
            raise ValueError("invalid eos_id")
        return _Job(prompt, count, eos, self.path.endswith("/stream"))

    def do_POST(self) -> None:
        if self.path not in ("/generate", "/generate/stream"):
            self._json(404, {"error": "unknown endpoint"})
            return
        try:
            job = self._parse()
        except OverflowError as error:
            self._json(413, {"error": str(error)})
            return
        except (ValueError, UnicodeError, RecursionError) as error:
            self._json(400, {"error": str(error)})
            return
        except TimeoutError:
            self._json(408, {"error": "request body timeout"})
            return
        status = self.server.service.submit(job)
        if status != 200:
            self._json(status, {"error": "inference queue full" if status == 429 else "closing"})
            return
        try:
            if job.streaming:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
            while not job.cancelled.is_set():
                try:
                    event, data = job.events.get(timeout=0.05)
                except queue.Empty:
                    continue
                if job.streaming:
                    self.wfile.write(f"event: {event}\ndata: {json.dumps(data)}\n\n".encode())
                    self.wfile.flush()
                elif event in ("done", "error"):
                    self._json(200 if event == "done" else 500, data)
                if event in ("done", "error"):
                    break
        except OSError:
            pass
        finally:
            job.cancelled.set()


def main() -> None:
    """Serve the checked local model until interrupted."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--queue-capacity", type=int, default=4)
    parser.add_argument("--http-workers", type=int, default=16)
    parser.add_argument("--http-pending", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    with InferenceService(load_sut(DEFAULT_SUT_PATH), **vars(args)) as service:
        print(f"http://127.0.0.1:{service.port}", flush=True)
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
