#!/usr/bin/env python3
"""Async concurrency test for the FastAPI gateway (normal or SSE streaming)."""

import argparse
import asyncio
import json
import statistics
import time
from typing import Any

import httpx


async def read_sse_response(response: httpx.Response) -> int:
    """Consume an SSE response and return the number of generated characters."""
    char_count = 0
    async for line in response.aiter_lines():
        if not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            break
        try:
            payload: dict[str, Any] = json.loads(data)
        except json.JSONDecodeError:
            continue
        char_count += len(payload.get("delta", ""))
        if "error" in payload:
            raise RuntimeError(payload["error"])
    return char_count


async def run_one(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    url: str,
    payload: dict,
    stream: bool,
    index: int,
) -> dict[str, Any]:
    async with semaphore:
        started = time.perf_counter()
        try:
            if stream:
                async with client.stream("POST", url, json=payload) as response:
                    response.raise_for_status()
                    chars = await read_sse_response(response)
                return {
                    "index": index,
                    "ok": True,
                    "seconds": time.perf_counter() - started,
                    "chars": chars,
                }

            response = await client.post(url, json=payload)
            response.raise_for_status()
            body = response.json()
            return {
                "index": index,
                "ok": True,
                "seconds": time.perf_counter() - started,
                "chars": len(body.get("content", "")),
            }
        except Exception as exc:  # Keep one failed request from aborting the whole run.
            return {
                "index": index,
                "ok": False,
                "seconds": time.perf_counter() - started,
                "error": str(exc),
            }


async def main() -> None:
    parser = argparse.ArgumentParser(description="Concurrent load test for the inference API")
    parser.add_argument("--base-url", default="http://localhost:8080", help="FastAPI base URL")
    parser.add_argument("--requests", type=int, default=10, help="Total number of requests")
    parser.add_argument("--concurrency", type=int, default=3, help="Maximum concurrent requests")
    parser.add_argument("--timeout", type=float, default=180.0, help="Per-request timeout in seconds")
    parser.add_argument("--stream", action="store_true", help="Test the SSE streaming endpoint")
    parser.add_argument(
        "--prompt",
        default="请用三句话介绍一下什么是大语言模型推理服务。",
        help="User prompt sent to each request",
    )
    args = parser.parse_args()

    if args.requests < 1 or args.concurrency < 1:
        parser.error("--requests and --concurrency must both be >= 1")

    path = "/api/chat/stream" if args.stream else "/api/chat"
    url = args.base_url.rstrip("/") + path
    payload = {
        "messages": [{"role": "user", "content": args.prompt}],
        "temperature": 0.2,
        "max_tokens": 128,
    }
    semaphore = asyncio.Semaphore(args.concurrency)
    limits = httpx.Limits(
        max_connections=args.concurrency,
        max_keepalive_connections=args.concurrency,
    )
    timeout = httpx.Timeout(args.timeout, connect=10.0)

    print(f"Target:      {url}")
    print(f"Requests:    {args.requests}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Mode:        {'SSE streaming' if args.stream else 'normal'}")

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        start = time.perf_counter()
        results = await asyncio.gather(
            *[
                run_one(client, semaphore, url, payload, args.stream, i + 1)
                for i in range(args.requests)
            ]
        )
        elapsed = time.perf_counter() - start

    successes = [item for item in results if item["ok"]]
    failures = [item for item in results if not item["ok"]]
    latencies = [item["seconds"] for item in successes]
    total_chars = sum(item.get("chars", 0) for item in successes)

    print("\n========== Load Test Summary ==========")
    print(f"Total elapsed: {elapsed:.2f}s")
    print(f"Successful:   {len(successes)}")
    print(f"Failed:       {len(failures)}")
    print(f"Throughput:   {len(successes) / elapsed:.2f} successful req/s")
    print(f"Output chars: {total_chars}")
    if latencies:
        print(f"Avg latency:  {statistics.mean(latencies):.2f}s")
        print(f"Min latency:  {min(latencies):.2f}s")
        print(f"Max latency:  {max(latencies):.2f}s")
    if failures:
        print("\nFailed requests:")
        for item in failures:
            print(f"  #{item['index']}: {item['error']}")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
