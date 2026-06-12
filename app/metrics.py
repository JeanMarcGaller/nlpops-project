"""Prometheus metrics for the classifier service.

Metrics use the default registry and are exposed via /metrics.
"""

from prometheus_client import Counter, Gauge, Histogram

# Incoming classification requests.
REQUESTS = Counter(
    "nlpops_classify_requests_total",
    "Anzahl eingegangener /classify-Requests",
)

# Predicted top labels.
PREDICTIONS = Counter(
    "nlpops_predictions_total",
    "Vorhersagen je Top-Label",
    ["label"],
)

# End-to-end request latency.
REQUEST_LATENCY = Histogram(
    "nlpops_request_latency_seconds",
    "Ende-zu-Ende-Latenz eines /classify-Requests",
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0),
)

# Texts per model batch.
BATCH_SIZE = Histogram(
    "nlpops_batch_size",
    "Anzahl Texte pro Modell-Batch",
    buckets=(1, 2, 4, 8, 16, 32),
)

# Model inference time per batch.
INFERENCE_DURATION = Histogram(
    "nlpops_inference_duration_seconds",
    "Dauer der Modell-Inferenz pro Batch",
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
)

# Processed model batches.
BATCHES_PROCESSED = Counter(
    "nlpops_batches_processed",
    "Anzahl verarbeiteter Modell-Batches",
)

# Batch collection wait time.
BATCH_WAIT_TIME = Histogram(
    "nlpops_batch_wait_time_seconds",
    "Wartezeit beim Sammeln eines Batches (erstes Item bis Dispatch)",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5),
)

# Current batch queue depth.
QUEUE_DEPTH = Gauge(
    "nlpops_queue_depth",
    "Aktuell wartende Requests in der Batch-Queue",
)