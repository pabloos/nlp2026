from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional

from app.models.schemas import SubmissionRecord, SubmitResponse
from app.repositories.base import SubmissionRepository

RATE_LIMIT_SECONDS = 60  # longer — LLM inference takes time


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
            f1_quality=metrics["f1_quality"],
            f1_delivery=metrics["f1_delivery"],
            timestamp=datetime.now(timezone.utc),
        )
        self.repo.save_submission(record)

        all_best = self.repo.get_all_best()
        rank = next(i + 1 for i, r in enumerate(all_best) if r.team == team)

        return SubmitResponse(
            f1=record.f1,
            f1_quality=record.f1_quality,
            f1_delivery=record.f1_delivery,
            rank=rank,
        )

    def get_leaderboard(self) -> List[SubmissionRecord]:
        return self.repo.get_all_best()

    def get_team_history(self, team: str) -> List[SubmissionRecord]:
        return self.repo.get_team_history(team)
