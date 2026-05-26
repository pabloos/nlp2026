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

RNG          = 42
N_TRAIN      = 1500
N_TEST_EACH  = 50   # 50 pos-quality + 50 neg-quality, each split evenly on delivery → 100 test

# Keywords for delivery sentiment extraction
_POS_DELIVERY = [
    "fast shipping", "quick delivery", "fast delivery", "arrived quickly",
    "shipped quickly", "arrived early", "great shipping", "arrived on time",
    "speedy delivery", "delivered fast", "quick ship", "fast arrival",
    "super fast", "delivered the next day", "came early", "shipped fast",
]
_NEG_DELIVERY = [
    "late delivery", "delayed", "slow shipping", "took forever",
    "never arrived", "damaged", "wrong item", "took too long",
    "shipping delay", "lost in", "shipping was slow", "delivery was slow",
    "weeks to arrive", "took a week", "took 2 weeks", "shipping was terrible",
    "bad shipping", "package was damaged",
]


def _delivery_label(text: str) -> str | None:
    t = text.lower()
    pos = sum(1 for kw in _POS_DELIVERY if kw in t)
    neg = sum(1 for kw in _NEG_DELIVERY if kw in t)
    if pos == neg:
        return None
    return "pos" if pos > neg else "neg"


def _prepare_data() -> None:
    if LABELS_PATH.exists():
        return

    print("Descargando amazon_polarity (streaming)...")
    from datasets import load_dataset

    # amazon_polarity: label 0=neg, 1=pos; fields: title, content
    ds = load_dataset("amazon_polarity", streaming=True, split="train")

    rows: list[dict] = []
    needed = N_TRAIN + N_TEST_EACH * 4  # buffer for balanced sampling
    for review in ds:
        if len(rows) >= needed:
            break
        label   = review.get("label", -1)
        content = (review.get("content") or "").strip()
        title   = (review.get("title") or "").strip()
        if not content or len(content) < 60:
            continue

        quality  = "pos" if label == 1 else "neg"
        delivery = _delivery_label(content)
        if delivery is None:
            continue

        full_text = f"{title}. {content}" if title else content
        rows.append({"text": full_text, "quality": quality, "delivery": delivery})

    df = pd.DataFrame(rows).drop_duplicates(subset="text").sample(frac=1, random_state=RNG).reset_index(drop=True)

    train = df.iloc[:N_TRAIN][["text", "quality", "delivery"]]

    remainder = df.iloc[N_TRAIN:].reset_index(drop=True)

    # Build balanced test: 25 per (quality × delivery) cell = 100 total
    cells = []
    for q in ("pos", "neg"):
        for d in ("pos", "neg"):
            subset = remainder[(remainder["quality"] == q) & (remainder["delivery"] == d)].head(N_TEST_EACH // 2)
            cells.append(subset)
    test_df = pd.concat(cells).sample(frac=1, random_state=RNG).reset_index(drop=True)

    public_test    = test_df[["text"]].copy()
    public_test.insert(0, "id", range(len(public_test)))
    private_labels = pd.DataFrame({
        "id":       range(len(test_df)),
        "quality":  test_df["quality"].values,
        "delivery": test_df["delivery"].values,
    })

    DATA_DIR.mkdir(exist_ok=True)
    train.to_csv(DATA_DIR / "train.csv", index=False)
    public_test.to_csv(DATA_DIR / "public_test.csv", index=False)
    private_labels.to_csv(LABELS_PATH, index=False)
    print(f"Datos listos — train: {len(train)} | test: {len(test_df)}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _prepare_data()
    app.state.scoring_service     = ScoringService(LABELS_PATH)
    app.state.leaderboard_service = LeaderboardService(InMemorySubmissionRepository())
    yield


app = FastAPI(
    title="NLP Lab 6 — Amazon Aspect Sentiment Scoring Server",
    description="Mini-Kaggle competitivo: aspect-based sentiment en reseñas Amazon Electronics con LLM local.",
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
