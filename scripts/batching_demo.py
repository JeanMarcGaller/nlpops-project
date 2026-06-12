"""Demo showing that MicroBatcher groups requests.

Uses a stub classifier, starts 20 concurrent requests, and prints the real batch
sizes.

Run: uv run python scripts/batching_demo.py
"""

import asyncio
import sys
import time
from pathlib import Path

# Make app imports work from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.batching import MicroBatcher


class StubClassifier:
    """Fake inference with recorded batch sizes."""

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def classify(self, texts: list[str]) -> list[list[dict[str, float]]]:
        self.batch_sizes.append(len(texts))
        time.sleep(0.1)  # lets the next batch collect
        return [[{"label": "stub", "score": 1.0}] for _ in texts]


async def main() -> None:
    stub = StubClassifier()
    batcher = MicroBatcher(stub, max_batch_size=16, max_wait_ms=50)
    await batcher.start()

    # Submit 20 requests concurrently.
    results = await asyncio.gather(
        *(batcher.submit(f"Comment {i}") for i in range(20))
    )

    await batcher.stop()

    print(f"Total requests : {len(results)}")
    print(f"Batch count    : {len(stub.batch_sizes)}")
    print(f"Batch sizes    : {stub.batch_sizes}")

    assert sum(stub.batch_sizes) == 20, "Exactly 20 texts must be processed"
    assert max(stub.batch_sizes) > 1, "At least one batch should have size > 1"

    print("OK: Microbatching groups multiple requests.")


if __name__ == "__main__":
    asyncio.run(main())