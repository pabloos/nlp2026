from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional

from app.models.schemas import SubmissionRecord


class SubmissionRepository(ABC):

    @abstractmethod
    def save_submission(self, record: SubmissionRecord) -> None:
        """Persist a new submission."""
        ...

    @abstractmethod
    def get_best_by_team(self, team: str) -> Optional[SubmissionRecord]:
        """Return the best (highest F1) submission for a team."""
        ...

    @abstractmethod
    def get_all_best(self) -> List[SubmissionRecord]:
        """Return the best submission per team, sorted by F1 descending."""
        ...

    @abstractmethod
    def get_last_submission_time(self, team: str) -> Optional[datetime]:
        """Return the timestamp of the most recent submission by a team."""
        ...

    @abstractmethod
    def get_team_history(self, team: str) -> List[SubmissionRecord]:
        """Return all submissions for a team, oldest first."""
        ...
