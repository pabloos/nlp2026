from __future__ import annotations

import html
from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse

from app.dependencies import get_leaderboard_service
from app.models.schemas import HistoryEntry, LeaderboardEntry
from app.services.leaderboard import LeaderboardService

router = APIRouter()


@router.get("/leaderboard", response_model=List[LeaderboardEntry])
def leaderboard_json(
    service: LeaderboardService = Depends(get_leaderboard_service),
) -> List[LeaderboardEntry]:
    records = service.get_leaderboard()
    return [
        LeaderboardEntry(
            team=r.team, f1=r.f1,
            f1_quality=r.f1_quality, f1_delivery=r.f1_delivery,
            timestamp=r.timestamp,
        )
        for r in records
    ]


@router.get("/history/{team}", response_model=List[HistoryEntry])
def team_history(
    team: str,
    service: LeaderboardService = Depends(get_leaderboard_service),
) -> List[HistoryEntry]:
    records = service.get_team_history(team)
    return [
        HistoryEntry(f1=r.f1, f1_quality=r.f1_quality,
                     f1_delivery=r.f1_delivery, timestamp=r.timestamp)
        for r in records
    ]


@router.get("/", response_class=HTMLResponse)
def leaderboard_html(
    service: LeaderboardService = Depends(get_leaderboard_service),
) -> str:
    records = service.get_leaderboard()
    medals = ["🥇", "🥈", "🥉"]

    rows = ""
    for i, r in enumerate(records, 1):
        medal = medals[i - 1] if i <= 3 else f"#{i}"
        ts = r.timestamp.strftime("%H:%M UTC")
        safe_team = html.escape(r.team)
        rows += f"""
        <tr>
          <td class="rank">{medal}</td>
          <td class="team">{safe_team}</td>
          <td class="f1">{r.f1:.4f}</td>
          <td class="fq">{r.f1_quality:.4f}</td>
          <td class="fd">{r.f1_delivery:.4f}</td>
          <td class="ts">{ts}</td>
        </tr>"""

    now = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    empty_row = '<tr><td colspan="6" class="empty">Sin submissions todavía — ¡sé el primero!</td></tr>'

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="refresh" content="10">
  <title>Lab 6 · Leaderboard</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Courier New', monospace;
      background: #0d1117;
      color: #c9d1d9;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 48px 20px;
    }}
    h1 {{ color: #58a6ff; font-size: 1.6rem; margin-bottom: 4px; }}
    .subtitle {{ color: #8b949e; font-size: 0.85rem; margin-bottom: 32px; }}
    table {{ width: 100%; max-width: 860px; border-collapse: collapse; }}
    th {{
      color: #8b949e; font-size: 0.72rem; text-transform: uppercase;
      letter-spacing: .1em; padding: 8px 14px;
      border-bottom: 2px solid #21262d; text-align: left;
    }}
    td {{ padding: 13px 14px; border-bottom: 1px solid #161b22; }}
    tr:hover td {{ background: #161b22; }}
    .rank {{ width: 64px; font-size: 1.15rem; }}
    .f1  {{ font-weight: bold; color: #3fb950; font-size: 1.1rem; }}
    .fq  {{ color: #79c0ff; }}
    .fd  {{ color: #d2a8ff; }}
    .ts  {{ color: #8b949e; font-size: 0.78rem; }}
    .empty {{ color: #484f58; padding: 32px; text-align: center; }}
    .footer {{ color: #484f58; font-size: 0.72rem; margin-top: 24px; }}
  </style>
</head>
<body>
  <h1>NLP Lab 6 · Amazon Aspect Sentiment</h1>
  <p class="subtitle">reseñas Electronics &nbsp;·&nbsp; F1 global = (calidad + entrega) / 2</p>
  <table>
    <thead>
      <tr>
        <th>Rank</th><th>Alumno</th>
        <th>F1 Global</th><th>F1 Calidad</th><th>F1 Entrega</th>
        <th>Último submit</th>
      </tr>
    </thead>
    <tbody>
      {rows if records else empty_row}
    </tbody>
  </table>
  <p class="footer">Auto-refresh cada 10 s &nbsp;·&nbsp; {now}</p>
</body>
</html>"""
