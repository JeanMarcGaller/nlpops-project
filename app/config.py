"""Classifier service configuration.

Values can be overridden via environment variables.
"""

import os

# Zero-shot model.
MODEL_NAME = os.getenv("MODEL_NAME", "facebook/bart-large-mnli")

# Microbatch limits.
BATCH_MAX_SIZE = int(os.getenv("BATCH_MAX_SIZE", "16"))
BATCH_MAX_WAIT_MS = int(os.getenv("BATCH_MAX_WAIT_MS", "50"))

# Candidate labels.
CANDIDATE_LABELS = [
    "sachliche Kritik",
    "Zustimmung",
    "Sarkasmus oder Ironie",
    "Off-Topic-Kommentar",
    "Empoerung oder Rant",
    "Desinformation oder Verschwoerung",
]