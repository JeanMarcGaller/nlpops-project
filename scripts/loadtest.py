"""Simple load generator for the classifier service.

Runs concurrent workers for a fixed duration, sends random sample comments to
POST /classify, and prints throughput plus latency stats.

Example:
    uv run python scripts/loadtest.py --url http://localhost:8000 --concurrency 8 --duration 120
"""

from __future__ import annotations

import argparse
import asyncio
import random
import time

import httpx

# Sample comments covering all labels.
COMMENTS = [
    "Vielen Dank fuer diesen wirklich gut recherchierten Artikel.",
    "Das ist doch voelliger Unsinn, niemand glaubt diese Zahlen.",
    "Klar, und morgen fliegen die Schweine. Grossartig.",
    "Hat jemand das Rezept fuer den Kuchen von letzter Woche?",
    "Eine sachliche Anmerkung: die Quelle in Absatz drei fehlt.",
    "Das ist eine bodenlose Frechheit, ich bin empoert!",
    "Die Regierung verheimlicht uns doch die wahren Daten.",
    "Sehr ausgewogene Darstellung, weiter so.",
    "Was hat das mit dem Thema zu tun? Komplett off-topic.",
    "Ironie an: na das wird ja super funktionieren. Ironie aus.",
]


async def worker(
    client: httpx.AsyncClient, url: str, stop_at: float, stats: dict
) -> None:
    while time.perf_counter() < stop_at:
        text = random.choice(COMMENTS)
        started = time.perf_counter()
        try:
            resp = await client.post(f"{url}/classify", json={"comment": text})
            latency = time.perf_counter() - started
            if resp.status_code == 200:
                stats["ok"] += 1
                stats["latencies"].append(latency)
            else:
                stats["errors"] += 1
        except httpx.HTTPError:
            stats["errors"] += 1


async def main() -> None:
    parser = argparse.ArgumentParser(description="Load generator for /classify")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--duration", type=float, default=60.0)
    args = parser.parse_args()

    stats: dict = {"ok": 0, "errors": 0, "latencies": []}
    stop_at = time.perf_counter() + args.duration

    print(
        f"Load test: {args.concurrency} workers, {args.duration:.0f}s, target {args.url}"
    )
    async with httpx.AsyncClient(timeout=30.0) as client:
        await asyncio.gather(
            *(worker(client, args.url, stop_at, stats) for _ in range(args.concurrency))
        )

    lat = sorted(stats["latencies"])
    total = stats["ok"] + stats["errors"]

    print("\n--- Result ---")
    print(f"Total requests : {total}")
    print(f"Successful     : {stats['ok']}")
    print(f"Errors         : {stats['errors']}")
    print(f"Throughput     : {stats['ok'] / args.duration:.1f} req/s")

    if lat:
        print(f"Latency p50    : {lat[len(lat) // 2] * 1000:.0f} ms")
        print(f"Latency p95    : {lat[int(len(lat) * 0.95)] * 1000:.0f} ms")
        print(f"Latency max    : {lat[-1] * 1000:.0f} ms")


if __name__ == "__main__":
    asyncio.run(main())