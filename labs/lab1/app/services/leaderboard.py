from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List

from app.models.schemas import SubmissionRecord, SubmitResponse
from app.repositories.base import SubmissionRepository

RATE_LIMIT_SECONDS = 30


class RateLimitError(Exception):
    pass


class LeaderboardService:

    def __init__(self, repository: SubmissionRepository) -> None:
        self.repo = repository

    def check_rate_limit(self, team: str) -> None:
        last = self.repo.get_last_submission_time(team)
        if last is not None:
            elapsed = (datetime.now(timezone.utc) - last).total_seconds()
            if elapsed < RATE_LIMIT_SECONDS:
                wait = int(RATE_LIMIT_SECONDS - elapsed) + 1
                raise RateLimitError(
                    f"Rate limit: espera {wait}s antes del próximo submit."
                )

    def record_submission(self, team: str, metrics: Dict[str, float]) -> SubmitResponse:
        record = SubmissionRecord(
            team=team,
            f1=metrics["f1"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            accuracy=metrics["accuracy"],
            timestamp=datetime.now(timezone.utc),
        )
        self.repo.save_submission(record)

        all_best = self.repo.get_all_best()
        rank = next(i + 1 for i, r in enumerate(all_best) if r.team == team)

        return SubmitResponse(
            f1=record.f1,
            precision=record.precision,
            recall=record.recall,
            accuracy=record.accuracy,
            rank=rank,
        )

    def get_leaderboard(self) -> List[SubmissionRecord]:
        return self.repo.get_all_best()

    def get_team_history(self, team: str) -> List[SubmissionRecord]:
        return self.repo.get_team_history(team)
