"""
Run once before building the Docker image:
    python scripts/generate_data.py
"""
from pathlib import Path
import pandas as pd

DATA_DIR    = Path("data")
LABELS_PATH = DATA_DIR / "private_labels.csv"

RNG         = 42
N_TRAIN     = 2000
N_TEST_EACH = 150

if LABELS_PATH.exists():
    print("Data already exists — delete data/ to regenerate.")
    raise SystemExit(0)

print("Descargando tweet_eval sentiment (puede tardar ~30s)...")
from datasets import load_dataset

ds = load_dataset("tweet_eval", "sentiment", )
df = ds["train"].to_pandas()
for split in ("validation", "test"):
    df = pd.concat([df, ds[split].to_pandas()], ignore_index=True)

df = df[df["label"].isin([0, 2])].copy()
df["label_str"] = df["label"].map({0: "neg", 2: "pos"})
df = df.sample(frac=1, random_state=RNG).reset_index(drop=True)

train = df.iloc[:N_TRAIN][["text", "label_str"]].rename(columns={"label_str": "label"})

remainder = df.iloc[N_TRAIN:]
pos_rows  = remainder[remainder["label_str"] == "pos"].head(N_TEST_EACH)
neg_rows  = remainder[remainder["label_str"] == "neg"].head(N_TEST_EACH)
test_df   = pd.concat([pos_rows, neg_rows]).sample(frac=1, random_state=RNG).reset_index(drop=True)

public_test    = test_df[["text"]].copy()
public_test.insert(0, "id", range(len(public_test)))
private_labels = pd.DataFrame({"id": range(len(test_df)), "label": test_df["label_str"].values})

DATA_DIR.mkdir(exist_ok=True)
train.to_csv(DATA_DIR / "train.csv", index=False)
public_test.to_csv(DATA_DIR / "public_test.csv", index=False)
private_labels.to_csv(LABELS_PATH, index=False)

print(f"Listo — train: {len(train)} | test: {len(test_df)}")
print(f"  pos en test: {(test_df['label_str']=='pos').sum()} | neg: {(test_df['label_str']=='neg').sum()}")
