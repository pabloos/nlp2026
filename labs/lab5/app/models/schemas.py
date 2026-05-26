from __future__ import annotations

import re
from datetime import datetime
from typing import List, Literal

from pydantic import BaseModel, field_validator


class PredictionItem(BaseModel):
    id: int
    label: Literal["pos", "neg"]


class SubmitRequest(BaseModel):
    team: str
    predictions: List[PredictionItem]

    @field_validator("team")
    @classmethod
    def validate_team(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("team name cannot be empty")
        if len(v) > 50:
            raise ValueError("team name too long (max 50 chars)")
        if not re.match(r"^[\w\s\-\.]+$", v):
            raise ValueError(
                "team name can only contain letters, numbers, spaces, hyphens and dots"
            )
        return v


class SubmitResponse(BaseModel):
    f1: float
    precision: float
    recall: float
    accuracy: float
    rank: int


class LeaderboardEntry(BaseModel):
    team: str
    f1: float
    precision: float
    timestamp: datetime


class HistoryEntry(BaseModel):
    f1: float
    precision: float
    recall: float
    accuracy: float
    timestamp: datetime


class SubmissionRecord(BaseModel):
    team: str
    f1: float
    precision: float
    recall: float
    accuracy: float
    timestamp: datetime
