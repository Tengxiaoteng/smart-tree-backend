#!/usr/bin/env python3
"""
Smart Tree backend stability / soak test.

Runs a mixed CRUD workload against the FastAPI server and reports error rate and latency.

Examples:
  python stability_test.py --base-url http://localhost:8000 --duration 60 --concurrency 10
  python stability_test.py --duration 300 --concurrency 25 --think-ms 20 --json-out stability_report.json
  python stability_test.py --mode inprocess --duration 60 --concurrency 10
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import time
import uuid
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from dataclasses import dataclass, field
import os
from typing import Any

import httpx


def _now() -> float:
    return time.perf_counter()


def _pct(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return float("nan")
    if p <= 0:
        return sorted_values[0]
    if p >= 1:
        return sorted_values[-1]
    k = (len(sorted_values) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_values[int(k)]
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


@dataclass
class EndpointStats:
    total: int = 0
    ok: int = 0
    statuses: Counter[int] = field(default_factory=Counter)
    errors: Counter[str] = field(default_factory=Counter)
    _latencies_ms: list[float] = field(default_factory=list)
    _latency_seen: int = 0

    def record(
        self,
        *,
        ok: bool,
        status_code: int | None,
        latency_ms: float | None,
        error: str | None,
        rng: random.Random,
        latency_sample_limit: int,
    ) -> None:
        self.total += 1
        if ok:
            self.ok += 1
        if status_code is not None:
            self.statuses[int(status_code)] += 1
        if error:
            self.errors[str(error)] += 1
        if latency_ms is None:
            return
        self._latency_seen += 1
        if latency_sample_limit <= 0:
            return
        if len(self._latencies_ms) < latency_sample_limit:
            self._latencies_ms.append(float(latency_ms))
            return
        j = rng.randrange(self._latency_seen)
        if j < latency_sample_limit:
            self._latencies_ms[j] = float(latency_ms)

    def latency_summary(self) -> dict[str, float]:
        if not self._latencies_ms:
            return {}
        values = sorted(self._latencies_ms)
        return {
            "p50_ms": _pct(values, 0.50),
            "p90_ms": _pct(values, 0.90),
            "p99_ms": _pct(values, 0.99),
            "max_ms": values[-1],
        }


@dataclass
class RunStats:
    endpoints: dict[str, EndpointStats] = field(default_factory=lambda: defaultdict(EndpointStats))

    def merge_from(self, other: RunStats) -> None:
        for key, src in other.endpoints.items():
            dst = self.endpoints[key]
            dst.total += src.total
            dst.ok += src.ok
            dst.statuses.update(src.statuses)
            dst.errors.update(src.errors)
            # Merge latency reservoir samples (approx): just concatenate then trim.
            dst._latencies_ms.extend(src._latencies_ms)
            dst._latency_seen += src._latency_seen


@dataclass
class AuthContext:
    username: str
    password: str
    token: str


def _auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


async def _request(
    client: httpx.AsyncClient,
    stats: RunStats,
    *,
    label: str,
    method: str,
    url: str,
    headers: dict[str, str] | None = None,
    json_body: Any | None = None,
    params: dict[str, Any] | None = None,
    rng: random.Random,
    latency_sample_limit: int,
    timeout_s: float,
) -> httpx.Response | None:
    start = _now()
    status_code: int | None = None
    try:
        resp = await client.request(
            method,
            url,
            headers=headers,
            json=json_body,
            params=params,
            timeout=httpx.Timeout(timeout_s),
        )
        status_code = resp.status_code
        latency_ms = (_now() - start) * 1000
        ok = 200 <= resp.status_code < 300
        stats.endpoints[label].record(
            ok=ok,
            status_code=resp.status_code,
            latency_ms=latency_ms,
            error=None if ok else f"http_{resp.status_code}",
            rng=rng,
            latency_sample_limit=latency_sample_limit,
        )
        return resp
    except Exception as exc:  # noqa: BLE001 - this is a load test harness
        latency_ms = (_now() - start) * 1000
        stats.endpoints[label].record(
            ok=False,
            status_code=status_code,
            latency_ms=latency_ms,
            error=exc.__class__.__name__,
            rng=rng,
            latency_sample_limit=latency_sample_limit,
        )
        return None


async def _register_user(
    client: httpx.AsyncClient,
    stats: RunStats,
    *,
    base_rng: random.Random,
    latency_sample_limit: int,
    timeout_s: float,
) -> AuthContext:
    username = f"stability_{uuid.uuid4().hex[:10]}"
    password = "test123456"
    resp = await _request(
        client,
        stats,
        label="POST /api/auth/register",
        method="POST",
        url="/api/auth/register",
        json_body={"username": username, "password": password, "nickname": "stability"},
        rng=base_rng,
        latency_sample_limit=latency_sample_limit,
        timeout_s=timeout_s,
    )
    if resp is None:
        raise RuntimeError("register request failed (no response)")
    if resp.status_code != 200:
        raise RuntimeError(f"register failed: HTTP {resp.status_code}: {resp.text}")
    data = resp.json()
    token = data.get("token")
    if not token:
        raise RuntimeError("register response missing token")
    return AuthContext(username=username, password=password, token=token)


async def _login(
    client: httpx.AsyncClient,
    stats: RunStats,
    auth: AuthContext,
    *,
    base_rng: random.Random,
    latency_sample_limit: int,
    timeout_s: float,
) -> str:
    resp = await _request(
        client,
        stats,
        label="POST /api/auth/login",
        method="POST",
        url="/api/auth/login",
        json_body={"username": auth.username, "password": auth.password},
        rng=base_rng,
        latency_sample_limit=latency_sample_limit,
        timeout_s=timeout_s,
    )
    if resp is None:
        raise RuntimeError("login request failed (no response)")
    if resp.status_code != 200:
        raise RuntimeError(f"login failed: HTTP {resp.status_code}: {resp.text}")
    data = resp.json()
    token = data.get("token")
    if not token:
        raise RuntimeError("login response missing token")
    return token


async def _create_topic(
    client: httpx.AsyncClient,
    stats: RunStats,
    *,
    token: str,
    name: str,
    rng: random.Random,
    latency_sample_limit: int,
    timeout_s: float,
) -> str:
    resp = await _request(
        client,
        stats,
        label="POST /api/topics",
        method="POST",
        url="/api/topics",
        headers=_auth_headers(token),
        json_body={"name": name, "description": "stability test"},
        rng=rng,
        latency_sample_limit=latency_sample_limit,
        timeout_s=timeout_s,
    )
    if resp is None:
        raise RuntimeError("create topic failed (no response)")
    if resp.status_code != 200:
        raise RuntimeError(f"create topic failed: HTTP {resp.status_code}: {resp.text}")
    data = resp.json()
    topic_id = data.get("id")
    if not topic_id:
        raise RuntimeError("create topic response missing id")
    return str(topic_id)


async def _delete_topic_cascade(
    client: httpx.AsyncClient,
    stats: RunStats,
    *,
    token: str,
    topic_id: str,
    rng: random.Random,
    latency_sample_limit: int,
    timeout_s: float,
) -> None:
    await _request(
        client,
        stats,
        label="DELETE /api/topics/{id}/cascade",
        method="DELETE",
        url=f"/api/topics/{topic_id}/cascade",
        headers=_auth_headers(token),
        rng=rng,
        latency_sample_limit=latency_sample_limit,
        timeout_s=timeout_s,
    )


async def _worker(
    worker_id: int,
    client: httpx.AsyncClient,
    auth: AuthContext,
    *,
    deadline: float,
    think_s: float,
    rng_seed: int,
    latency_sample_limit: int,
    timeout_s: float,
) -> tuple[RunStats, str]:
    rng = random.Random(rng_seed + worker_id * 9973)
    stats = RunStats()
    token = auth.token

    # Create per-worker topic to reduce cross-worker contention and name collisions.
    topic_id = await _create_topic(
        client,
        stats,
        token=token,
        name=f"stability_topic_{worker_id}_{uuid.uuid4().hex[:6]}",
        rng=rng,
        latency_sample_limit=latency_sample_limit,
        timeout_s=timeout_s,
    )

    while _now() < deadline:
        # Light read
        await _request(
            client,
            stats,
            label="GET /api/topics",
            method="GET",
            url="/api/topics",
            headers=_auth_headers(token),
            rng=rng,
            latency_sample_limit=latency_sample_limit,
            timeout_s=timeout_s,
        )

        # Create node
        node_name = f"node_{worker_id}_{uuid.uuid4().hex[:8]}"
        resp = await _request(
            client,
            stats,
            label="POST /api/nodes",
            method="POST",
            url="/api/nodes",
            headers=_auth_headers(token),
            json_body={
                "name": node_name,
                "description": "stability node",
                "topicId": topic_id,
                "knowledgeType": "concept",
                "difficulty": "beginner",
            },
            rng=rng,
            latency_sample_limit=latency_sample_limit,
            timeout_s=timeout_s,
        )
        if resp is None:
            if think_s:
                await asyncio.sleep(think_s)
            continue

        if resp.status_code == 401:
            # Token might have expired; re-login and keep going.
            token = await _login(
                client,
                stats,
                auth,
                base_rng=rng,
                latency_sample_limit=latency_sample_limit,
                timeout_s=timeout_s,
            )
            if think_s:
                await asyncio.sleep(think_s)
            continue

        if resp.status_code != 200:
            if think_s:
                await asyncio.sleep(think_s)
            continue

        node = resp.json()
        node_id = node.get("id")
        if not node_id:
            if think_s:
                await asyncio.sleep(think_s)
            continue
        node_id = str(node_id)

        await _request(
            client,
            stats,
            label="GET /api/nodes",
            method="GET",
            url="/api/nodes",
            headers=_auth_headers(token),
            params={"topicId": topic_id},
            rng=rng,
            latency_sample_limit=latency_sample_limit,
            timeout_s=timeout_s,
        )

        await _request(
            client,
            stats,
            label="GET /api/nodes/{id}",
            method="GET",
            url=f"/api/nodes/{node_id}",
            headers=_auth_headers(token),
            rng=rng,
            latency_sample_limit=latency_sample_limit,
            timeout_s=timeout_s,
        )

        await _request(
            client,
            stats,
            label="PATCH /api/nodes/{id}",
            method="PATCH",
            url=f"/api/nodes/{node_id}",
            headers=_auth_headers(token),
            json_body={"description": f"stability node updated {uuid.uuid4().hex[:6]}"},
            rng=rng,
            latency_sample_limit=latency_sample_limit,
            timeout_s=timeout_s,
        )

        await _request(
            client,
            stats,
            label="DELETE /api/nodes/{id}",
            method="DELETE",
            url=f"/api/nodes/{node_id}",
            headers=_auth_headers(token),
            rng=rng,
            latency_sample_limit=latency_sample_limit,
            timeout_s=timeout_s,
        )

        if think_s:
            await asyncio.sleep(think_s)

    return stats, topic_id


def _render_summary(stats: RunStats, *, duration_s: float) -> dict[str, Any]:
    total = sum(s.total for s in stats.endpoints.values())
    ok = sum(s.ok for s in stats.endpoints.values())
    errors_total = total - ok
    endpoint_rows: list[dict[str, Any]] = []
    for key in sorted(stats.endpoints.keys()):
        s = stats.endpoints[key]
        row: dict[str, Any] = {
            "endpoint": key,
            "total": s.total,
            "ok": s.ok,
            "error": s.total - s.ok,
            "ok_rate": (s.ok / s.total) if s.total else 0.0,
            "statuses": dict(s.statuses),
            "errors": dict(s.errors),
        }
        row.update(s.latency_summary())
        endpoint_rows.append(row)

    all_latencies: list[float] = []
    for s in stats.endpoints.values():
        all_latencies.extend(s._latencies_ms)
    all_latencies.sort()

    return {
        "duration_s": duration_s,
        "requests_total": total,
        "requests_ok": ok,
        "requests_error": errors_total,
        "ok_rate": (ok / total) if total else 0.0,
        "rps": (total / duration_s) if duration_s > 0 else 0.0,
        "latency": (
            {
                "p50_ms": _pct(all_latencies, 0.50),
                "p90_ms": _pct(all_latencies, 0.90),
                "p99_ms": _pct(all_latencies, 0.99),
                "max_ms": all_latencies[-1],
            }
            if all_latencies
            else {}
        ),
        "endpoints": endpoint_rows,
    }


def _print_human(summary: dict[str, Any]) -> None:
    print("=" * 72)
    print("Smart Tree API stability test summary")
    print("=" * 72)
    print(
        f"duration={summary['duration_s']:.1f}s  "
        f"requests={summary['requests_total']}  "
        f"ok_rate={summary['ok_rate']*100:.2f}%  "
        f"rps={summary['rps']:.1f}"
    )
    latency = summary.get("latency") or {}
    if latency:
        print(
            "latency_ms "
            f"p50={latency.get('p50_ms', float('nan')):.1f}  "
            f"p90={latency.get('p90_ms', float('nan')):.1f}  "
            f"p99={latency.get('p99_ms', float('nan')):.1f}  "
            f"max={latency.get('max_ms', float('nan')):.1f}"
        )
    print("-" * 72)
    for row in summary["endpoints"]:
        p50 = row.get("p50_ms")
        p99 = row.get("p99_ms")
        p50_s = f"{p50:.1f}" if isinstance(p50, (int, float)) else "-"
        p99_s = f"{p99:.1f}" if isinstance(p99, (int, float)) else "-"
        print(
            f"{row['endpoint']:<28} "
            f"total={row['total']:<6} ok_rate={row['ok_rate']*100:>6.2f}% "
            f"p50={p50_s:>7} p99={p99_s:>7}"
        )
        if row.get("errors"):
            # Show top 2 errors per endpoint.
            top = sorted(row["errors"].items(), key=lambda kv: (-kv[1], kv[0]))[:2]
            if top:
                err_str = ", ".join([f"{k}={v}" for k, v in top])
                print(f"  errors: {err_str}")
    print("=" * 72)


async def _amain() -> int:
    parser = argparse.ArgumentParser(description="Smart Tree backend stability/soak test")
    parser.add_argument(
        "--mode",
        choices=["http", "inprocess"],
        default="http",
        help="http: call a running server; inprocess: run ASGI app without binding a port",
    )
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--duration", type=float, default=60.0, help="Run duration (seconds)")
    parser.add_argument("--concurrency", type=int, default=10, help="Number of concurrent workers")
    parser.add_argument("--timeout", type=float, default=10.0, help="Per-request timeout (seconds)")
    parser.add_argument("--think-ms", type=float, default=0.0, help="Sleep between iterations (ms)")
    parser.add_argument("--username", default="", help="Use an existing user instead of auto-registering")
    parser.add_argument("--password", default="", help="Password for --username")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--quiet-app-stdout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Suppress app's print() output (useful for inprocess mode)",
    )
    parser.add_argument(
        "--max-latency-samples",
        type=int,
        default=20000,
        help="Reservoir sample size for latency percentiles (per endpoint)",
    )
    parser.add_argument("--json-out", default="", help="Write JSON summary to a file path")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    think_s = max(0.0, float(args.think_ms) / 1000.0)
    duration_s = max(0.0, float(args.duration))
    concurrency = max(1, int(args.concurrency))
    latency_sample_limit = max(0, int(args.max_latency_samples))
    timeout_s = max(0.1, float(args.timeout))

    stats = RunStats()

    inprocess_app = None
    if args.mode == "inprocess":
        from main import app as _app  # imported lazily so env loading happens after CLI parsing

        inprocess_app = _app
        await inprocess_app.router.startup()
        transport = httpx.ASGITransport(app=inprocess_app)
        client_ctx: httpx.AsyncClient | None = httpx.AsyncClient(
            transport=transport, base_url="http://inprocess"
        )
    else:
        client_ctx = httpx.AsyncClient(base_url=args.base_url)

    async def _run_workload(client: httpx.AsyncClient) -> None:
        await _request(
            client,
            stats,
            label="GET /health",
            method="GET",
            url="/health",
            rng=rng,
            latency_sample_limit=latency_sample_limit,
            timeout_s=timeout_s,
        )
        await _request(
            client,
            stats,
            label="GET /health/db",
            method="GET",
            url="/health/db",
            rng=rng,
            latency_sample_limit=latency_sample_limit,
            timeout_s=timeout_s,
        )

        if args.username:
            if not args.password:
                raise SystemExit("--password is required when using --username")
            auth = AuthContext(username=args.username, password=args.password, token="")
            auth.token = await _login(
                client,
                stats,
                auth,
                base_rng=rng,
                latency_sample_limit=latency_sample_limit,
                timeout_s=timeout_s,
            )
        else:
            auth = await _register_user(
                client,
                stats,
                base_rng=rng,
                latency_sample_limit=latency_sample_limit,
                timeout_s=timeout_s,
            )

        deadline = _now() + duration_s
        tasks = [
            _worker(
                i,
                client,
                auth,
                deadline=deadline,
                think_s=think_s,
                rng_seed=args.seed,
                latency_sample_limit=latency_sample_limit,
                timeout_s=timeout_s,
            )
            for i in range(concurrency)
        ]
        worker_results = await asyncio.gather(*tasks)

        cleanup_token = await _login(
            client,
            stats,
            auth,
            base_rng=rng,
            latency_sample_limit=latency_sample_limit,
            timeout_s=timeout_s,
        )

        for worker_stat, topic_id in worker_results:
            stats.merge_from(worker_stat)
            await _delete_topic_cascade(
                client,
                stats,
                token=cleanup_token,
                topic_id=topic_id,
                rng=rng,
                latency_sample_limit=latency_sample_limit,
                timeout_s=timeout_s,
            )

    try:
        async with client_ctx as client:
            if args.mode == "inprocess" and args.quiet_app_stdout:
                with open(os.devnull, "w", encoding="utf-8") as devnull:
                    with redirect_stdout(devnull):
                        await _run_workload(client)
            else:
                await _run_workload(client)
    finally:
        if inprocess_app is not None:
            await inprocess_app.router.shutdown()

    summary = _render_summary(stats, duration_s=duration_s)
    _print_human(summary)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"Wrote JSON report: {args.json_out}")

    return 0 if summary["requests_error"] == 0 else 2


def main() -> None:
    raise SystemExit(asyncio.run(_amain()))


if __name__ == "__main__":
    main()
