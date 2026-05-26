from __future__ import annotations

import io
import zipfile
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI

from app.repositories.memory import InMemorySubmissionRepository
from app.routes.leaderboard import router as leaderboard_router
from app.routes.submit import router as submit_router
from app.services.leaderboard import LeaderboardService
from app.services.scoring import ScoringService

DATA_DIR    = Path("data")
LABELS_PATH = DATA_DIR / "private_labels.csv"
IMDB_URL    = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"


def _prepare_data() -> None:
    """Descarga el IMDB dataset y genera los tres CSV si no existen."""
    if LABELS_PATH.exists():
        return

    import tarfile
    DATA_DIR.mkdir(exist_ok=True)
    print("Descargando IMDB dataset (~84 MB)...")
    with urllib.request.urlopen(IMDB_URL, timeout=120) as r:
        tf = tarfile.open(fileobj=io.BytesIO(r.read()), mode="r:gz")
        tf.extractall("/tmp")

    def load_split(split: str) -> pd.DataFrame:
        rows = []
        for label in ("pos", "neg"):
            for f in (Path("/tmp/aclImdb") / split / label).iterdir():
                if f.suffix == ".txt":
                    rows.append({"text": f.read_text(errors="ignore").strip(), "label": label})
        return pd.DataFrame(rows)

    train = load_split("train").sample(frac=1, random_state=42).reset_index(drop=True)
    test_full = load_split("test").sample(frac=1, random_state=42).reset_index(drop=True)
    test = pd.concat([
        test_full[test_full["label"] == "pos"].head(250),
        test_full[test_full["label"] == "neg"].head(250),
    ]).sample(frac=1, random_state=42).reset_index(drop=True)

    public_test    = test[["text"]].copy()
    public_test.insert(0, "id", range(len(public_test)))
    private_labels = pd.DataFrame({"id": range(len(test)), "label": test["label"].values})

    train.to_csv(DATA_DIR / "train.csv", index=False)
    public_test.to_csv(DATA_DIR / "public_test.csv", index=False)
    private_labels.to_csv(LABELS_PATH, index=False)
    print(f"Datos listos — train: {len(train)} | test: {len(test)}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _prepare_data()
    # To switch to Firestore, replace InMemorySubmissionRepository with
    # FirestoreSubmissionRepository (same interface, different backend).
    app.state.scoring_service = ScoringService(LABELS_PATH)
    app.state.leaderboard_service = LeaderboardService(InMemorySubmissionRepository())
    yield


app = FastAPI(
    title="NLP Lab 1 — Spam Detection Scoring Server",
    description="Mini-Kaggle competitivo para clasificación de spam con TF-IDF + LR.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(submit_router)
app.include_router(leaderboard_router)
