from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from app.repositories.memory import InMemorySubmissionRepository
from app.routes.leaderboard import router as leaderboard_router
from app.routes.submit import router as submit_router
from app.services.leaderboard import LeaderboardService
from app.services.scoring import ScoringService

DATA_DIR    = Path("data")
LABELS_PATH = DATA_DIR / "private_labels.csv"

RNG         = 42
N_TRAIN     = 2000
N_TEST_EACH = 150   # 150 neg + 150 pos = 300 test tweets


def _prepare_data() -> None:
    if LABELS_PATH.exists():
        return

    print("Descargando tweet_eval sentiment...")
    from datasets import load_dataset  # deferred so server starts fast after first run
    ds = load_dataset("tweet_eval", "sentiment")
    df = ds["train"].to_pandas()
    for split in ("validation", "test"):
        df = pd.concat([df, ds[split].to_pandas()], ignore_index=True)

    # tweet_eval labels: 0=negative, 1=neutral, 2=positive — keep only pos/neg
    df = df[df["label"].isin([0, 2])].copy()
    df["label_str"] = df["label"].map({0: "neg", 2: "pos"})
    df = df.sample(frac=1, random_state=RNG).reset_index(drop=True)

    train = df.iloc[:N_TRAIN][["text", "label_str"]].rename(columns={"label_str": "label"})

    remainder = df.iloc[N_TRAIN:]
    pos_rows  = remainder[remainder["label_str"] == "pos"].head(N_TEST_EACH)
    neg_rows  = remainder[remainder["label_str"] == "neg"].head(N_TEST_EACH)
    test_df   = pd.concat([pos_rows, neg_rows]).sample(frac=1, random_state=RNG).reset_index(drop=True)

    public_test     = test_df[["text"]].copy()
    public_test.insert(0, "id", range(len(public_test)))
    private_labels  = pd.DataFrame({"id": range(len(test_df)), "label": test_df["label_str"].values})

    DATA_DIR.mkdir(exist_ok=True)
    train.to_csv(DATA_DIR / "train.csv", index=False)
    public_test.to_csv(DATA_DIR / "public_test.csv", index=False)
    private_labels.to_csv(LABELS_PATH, index=False)
    print(f"Datos listos — train: {len(train)} | test: {len(test_df)}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _prepare_data()
    app.state.scoring_service   = ScoringService(LABELS_PATH)
    app.state.leaderboard_service = LeaderboardService(InMemorySubmissionRepository())
    yield


app = FastAPI(
    title="NLP Lab 5 — Twitter Sentiment Scoring Server",
    description="Mini-Kaggle competitivo para clasificación de sentimiento en tweets con modelos pre-entrenados.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(submit_router)
app.include_router(leaderboard_router)


_ALLOWED_DATA = {"train.csv", "public_test.csv"}

@app.get("/data/{filename}")
def serve_data(filename: str):
    if filename not in _ALLOWED_DATA:
        raise HTTPException(status_code=404, detail="File not found")
    path = DATA_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=503, detail="Data not ready yet")
    return FileResponse(path, media_type="text/csv", filename=filename)
