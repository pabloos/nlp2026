"""
Run once before building the Docker image:
    cd labs/lab6
    python scripts/generate_data.py
"""
from pathlib import Path
import pandas as pd

DATA_DIR    = Path("data")
LABELS_PATH = DATA_DIR / "private_labels.csv"
RNG         = 42
N_TRAIN     = 1500
N_TEST_EACH = 50

POS_DELIVERY = [
    "fast shipping", "quick delivery", "fast delivery", "arrived quickly",
    "shipped quickly", "arrived early", "great shipping", "arrived on time",
    "speedy delivery", "delivered fast", "quick ship", "fast arrival",
    "super fast", "delivered the next day", "came early", "shipped fast",
]
NEG_DELIVERY = [
    "late delivery", "delayed", "slow shipping", "took forever",
    "never arrived", "damaged", "wrong item", "took too long",
    "shipping delay", "lost in", "shipping was slow", "delivery was slow",
    "weeks to arrive", "took a week", "took 2 weeks", "shipping was terrible",
    "bad shipping", "package was damaged",
]


def delivery_label(text: str):
    t = text.lower()
    pos = sum(1 for kw in POS_DELIVERY if kw in t)
    neg = sum(1 for kw in NEG_DELIVERY if kw in t)
    if pos == neg:
        return None
    return "pos" if pos > neg else "neg"


if LABELS_PATH.exists():
    print("Data already exists — delete data/ to regenerate.")
    raise SystemExit(0)

print("Descargando amazon_polarity (streaming, puede tardar 1-2 min)...")
from datasets import load_dataset

# amazon_polarity: label 0=neg, 1=pos; fields: title, content
ds = load_dataset("amazon_polarity", streaming=True, split="train")

rows = []
needed = N_TRAIN + N_TEST_EACH * 4
for review in ds:
    if len(rows) >= needed:
        break
    label   = review.get("label", -1)
    content = (review.get("content") or "").strip()
    title   = (review.get("title") or "").strip()
    if not content or len(content) < 60:
        continue
    quality = "pos" if label == 1 else "neg"
    dlabel  = delivery_label(content)
    if dlabel is None:
        continue
    full_text = f"{title}. {content}" if title else content
    rows.append({"text": full_text, "quality": quality, "delivery": dlabel})

df = pd.DataFrame(rows).drop_duplicates(subset="text").sample(frac=1, random_state=RNG).reset_index(drop=True)
train = df.iloc[:N_TRAIN][["text", "quality", "delivery"]]

remainder = df.iloc[N_TRAIN:].reset_index(drop=True)
cells = []
for q in ("pos", "neg"):
    for d in ("pos", "neg"):
        subset = remainder[(remainder["quality"] == q) & (remainder["delivery"] == d)].head(N_TEST_EACH // 2)
        cells.append(subset)
test_df = pd.concat(cells).sample(frac=1, random_state=RNG).reset_index(drop=True)

public_test    = test_df[["text"]].copy()
public_test.insert(0, "id", range(len(public_test)))
private_labels = pd.DataFrame({
    "id": range(len(test_df)),
    "quality": test_df["quality"].values,
    "delivery": test_df["delivery"].values,
})

DATA_DIR.mkdir(exist_ok=True)
train.to_csv(DATA_DIR / "train.csv", index=False)
public_test.to_csv(DATA_DIR / "public_test.csv", index=False)
private_labels.to_csv(LABELS_PATH, index=False)

print(f"\nListo!")
print(f"  Train: {len(train)} | Test: {len(test_df)}")
for q in ("pos", "neg"):
    for d in ("pos", "neg"):
        n = ((test_df["quality"] == q) & (test_df["delivery"] == d)).sum()
        print(f"  test quality={q} delivery={d}: {n}")
