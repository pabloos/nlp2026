from fastapi import Request

from app.services.leaderboard import LeaderboardService
from app.services.scoring import ScoringService


def get_scoring_service(request: Request) -> ScoringService:
    return request.app.state.scoring_service


def get_leaderboard_service(request: Request) -> LeaderboardService:
    return request.app.state.leaderboard_service
