from fastapi import APIRouter, Depends, HTTPException, status

from app.dependencies import get_leaderboard_service, get_scoring_service
from app.models.schemas import SubmitRequest, SubmitResponse
from app.services.leaderboard import LeaderboardService, RateLimitError
from app.services.scoring import ScoringService

router = APIRouter()


@router.post("/submit", response_model=SubmitResponse)
def submit(
    payload: SubmitRequest,
    scoring: ScoringService = Depends(get_scoring_service),
    leaderboard: LeaderboardService = Depends(get_leaderboard_service),
) -> SubmitResponse:
    try:
        leaderboard.check_rate_limit(payload.team)
    except RateLimitError as e:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(e))

    try:
        metrics = scoring.calculate_metrics(payload.predictions)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))

    return leaderboard.record_submission(payload.team, metrics)
