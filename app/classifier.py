"""Zero-shot classification via Hugging Face Transformers.

The pipeline accepts batches, so microbatching can pass multiple texts without
changing this interface.
"""

from __future__ import annotations

from transformers import pipeline

from app.config import CANDIDATE_LABELS, MODEL_NAME


class ZeroShotClassifier:
    """Wraps the zero-shot classification pipeline."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        labels: list[str] | None = None,
    ) -> None:
        self.labels = labels if labels is not None else CANDIDATE_LABELS

        # Use CPU for Docker deployment.
        self._pipeline = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=-1,
        )

    def classify(self, texts: list[str]) -> list[list[dict[str, float]]]:
        """Classify multiple texts in one call."""
        results = self._pipeline(texts, candidate_labels=self.labels)

        # Normalize single-item pipeline output.
        if isinstance(results, dict):
            results = [results]

        return [
            [
                {"label": label, "score": float(score)}
                for label, score in zip(item["labels"], item["scores"])
            ]
            for item in results
        ]