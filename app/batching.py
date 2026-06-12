"""Microbatching for classification requests.

Incoming texts are queued and sent to the model in batches. A batch is dispatched
once the max size is reached or max_wait has passed since the first item.

Blocking inference runs in the default thread pool so the event loop stays free.
Only one batch is processed at a time; new requests form the next batch.
"""

from __future__ import annotations

import asyncio
import logging
import time

from app.classifier import ZeroShotClassifier
from app.metrics import (
    BATCH_SIZE,
    BATCH_WAIT_TIME,
    BATCHES_PROCESSED,
    INFERENCE_DURATION,
    QUEUE_DEPTH,
)

logger = logging.getLogger("nlpops.batching")

# Queue item: text plus the future awaited by the caller.
QueueItem = tuple[str, "asyncio.Future[list[dict[str, float]]]"]


class MicroBatcher:
    def __init__(
        self,
        classifier: ZeroShotClassifier,
        max_batch_size: int,
        max_wait_ms: int,
    ) -> None:
        self._classifier = classifier
        self._max_batch_size = max_batch_size
        self._max_wait = max_wait_ms / 1000.0
        self._queue: asyncio.Queue[QueueItem] = asyncio.Queue()
        self._worker: asyncio.Task[None] | None = None

    async def start(self) -> None:
        self._worker = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
            self._worker = None

    async def submit(self, text: str) -> list[dict[str, float]]:
        """Queue one text and wait for its classification."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future[list[dict[str, float]]] = loop.create_future()
        await self._queue.put((text, future))
        QUEUE_DEPTH.inc()
        return await future

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            # Wait for the first item; avoids idle polling.
            batch: list[QueueItem] = [await self._queue.get()]
            wait_started = loop.time()
            deadline = wait_started + self._max_wait

            # Collect more items until size or time limit is reached.
            while len(batch) < self._max_batch_size:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                get_task = asyncio.ensure_future(self._queue.get())
                done, _ = await asyncio.wait({get_task}, timeout=remaining)
                if get_task in done:
                    batch.append(get_task.result())
                else:
                    # Cancel pending get; keep item if it arrived concurrently.
                    get_task.cancel()
                    try:
                        batch.append(await get_task)
                    except asyncio.CancelledError:
                        pass
                    break

            # Time from first item to dispatch.
            BATCH_WAIT_TIME.observe(loop.time() - wait_started)
            await self._process(batch)

    async def _process(self, batch: list[QueueItem]) -> None:
        texts = [text for text, _ in batch]
        futures = [future for _, future in batch]
        logger.info("Classifying batch with %d texts", len(texts))

        # Items are no longer queued.
        QUEUE_DEPTH.dec(len(batch))
        BATCH_SIZE.observe(len(batch))
        BATCHES_PROCESSED.inc()

        loop = asyncio.get_running_loop()
        started = time.perf_counter()
        try:
            results = await loop.run_in_executor(
                None, self._classifier.classify, texts
            )
        except Exception as exc:
            # Propagate failures to all waiting requests.
            for future in futures:
                if not future.done():
                    future.set_exception(exc)
            return
        finally:
            INFERENCE_DURATION.observe(time.perf_counter() - started)

        for future, result in zip(futures, results):
            if not future.done():
                future.set_result(result)