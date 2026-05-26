from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from app.models.schemas import SubmissionRecord
from app.repositories.base import SubmissionRepository


class InMemorySubmissionRepository(SubmissionRepository):
    """In-memory implementation — swap for FirestoreSubmissionRepository in production."""

    def __init__(self) -> None:
        self._all: Dict[str, List[SubmissionRecord]] = {}
        self._best: Dict[str, SubmissionRecord] = {}
        self._last_ts: Dict[str, datetime] = {}

    def save_submission(self, record: SubmissionRecord) -> None:
        team = record.team
        self._all.setdefault(team, []).append(record)
        self._last_ts[team] = record.timestamp
        if team not in self._best or record.f1 > self._best[team].f1:
            self._best[team] = record

    def get_best_by_team(self, team: str) -> Optional[SubmissionRecord]:
        return self._best.get(team)

    def get_all_best(self) -> List[SubmissionRecord]:
        return sorted(self._best.values(), key=lambda r: r.f1, reverse=True)

    def get_last_submission_time(self, team: str) -> Optional[datetime]:
        return self._last_ts.get(team)

    def get_team_history(self, team: str) -> List[SubmissionRecord]:
        return list(self._all.get(team, []))
