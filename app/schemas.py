"""Pydantic schemas for the classify API."""

from pydantic import BaseModel, Field


class ClassifyRequest(BaseModel):
    comment: str = Field(..., min_length=1, description="Comment to classify.")


class ClassifyResponse(BaseModel):
    comment: str
    label: str
    score: float