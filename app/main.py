"""FastAPI entry point for the classifier service.

Provides health, metrics, and microbatched zero-shot classification.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.batching import MicroBatcher
from app.classifier import ZeroShotClassifier
from app.config import BATCH_MAX_SIZE, BATCH_MAX_WAIT_MS
from app.metrics import PREDICTIONS, REQUEST_LATENCY, REQUESTS
from app.schemas import ClassifyRequest, ClassifyResponse

# Enable app logs, not just uvicorn logs.
logging.basicConfig(level=logging.INFO)

# Long-lived app objects.
state: dict[str, MicroBatcher] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model once and start the batch worker.
    classifier = ZeroShotClassifier()
    batcher = MicroBatcher(classifier, BATCH_MAX_SIZE, BATCH_MAX_WAIT_MS)
    await batcher.start()
    state["batcher"] = batcher
    yield
    await batcher.stop()
    state.clear()


app = FastAPI(
    title="NLP Operations Classifier",
    description="Zero-shot comment classification with microbatching.",
    version="0.3.0",
    lifespan=lifespan,
)


@app.get("/metrics")
def metrics() -> Response:
    """Expose Prometheus metrics."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check."""
    return {"status": "ok"}


@app.post("/classify", response_model=ClassifyResponse)
async def classify(req: ClassifyRequest) -> ClassifyResponse:
    """Classify one comment via microbatching."""
    REQUESTS.inc()
    with REQUEST_LATENCY.time():
        scored = await state["batcher"].submit(req.comment)

    top = scored[0]
    PREDICTIONS.labels(label=top["label"]).inc()

    return ClassifyResponse(
        comment=req.comment,
        label=top["label"],
        score=top["score"],
    )