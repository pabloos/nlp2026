"""Genera train.csv, public_test.csv y private_labels.csv para Lab 1.

Uso con datos sintéticos (por defecto):
    python scripts/generate_data.py

Uso con el dataset real SMS Spam Collection (recomendado para producción):
    1. Descarga SMSSpamCollection desde:
       https://archive.ics.uci.edu/ml/datasets/sms+spam+collection
    2. Ejecuta:
       python scripts/generate_data.py --real path/to/SMSSpamCollection
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

random.seed(42)

# --- Templates sintéticos ---------------------------------------------------

SPAM_MSGS = [
    "WINNER!! As a valued network customer you have been selected to receive a £900 prize reward!",
    "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121",
    "URGENT! Your Mobile number has been awarded a £2,000 Bonus Caller Prize on 9th September",
    "Congratulations ur awarded 500 of CD vouchers or 125gift guaranteed & Free entry 2 100 wkly draw",
    "You have WON a guaranteed £1000 cash or £5000 prize to redeem. Call 09061743810",
    "FreeMsg: Hey there darling it's been 3 weeks now and no word back! Looking for fun?",
    "IMPORTANT - You could be entitled up to £3,160 in compensation from mis-sold PPI on your mortgage",
    "Congrats! 1 year special cinema pass for 2 is yours. Call 09061209465 now! Cost: £3/month",
    "You are awarded a SIM card. Call 07xxxxxxxxx to claim your FREE membership",
    "Last chance to claim your £150 Argos voucher. Text ARGOS to 88600 now!",
    "Call now to claim your prize. Limited time offer expires tonight. Freephone 0800",
    "Win a brand new NOKIA phone every week. Just text WIN to 85023 now!",
    "SIX chances to win CASH! From £100 to £20,000. Text CSH11 to 87575",
    "CASH PRIZE of £1000 awaits you. Don't miss out. SMS PRIZE to 80166 now",
    "You've been selected for a special deal! Claim your £500 Tesco gift card today",
    "URGENT: Your account will be suspended. Verify your details now by calling 0800",
    "YOU HAVE BEEN CHOSEN! Text YES to 78765 to claim your free vacation package",
    "Ringtone club: Get up to 4 FREE ringtones! Text MUSIC to 88599",
    "Double your income! Work from home and earn £2000 a week. Reply INFO",
    "Congratulations! You've won a luxury holiday for two. Call 09058091040 to claim",
    "ALERT: Your loan has been pre-approved for £5000. Call 0800 to confirm",
    "FREE: You have 1 new message. Call 0808 to collect your prize. FREE entry!",
    "Wining number 5600 — claim your prize of £500 before midnight tonight",
    "Txt STOP to opt out. You have been selected to win a cash prize of up to £200",
    "PRIVATE: 08718726270 Call NOW! £0.04 per min BT-national rate. 24hrs 7 days",
]

HAM_MSGS = [
    "Hey, what time are you coming over tonight?",
    "I'll be there in 10 minutes, just leaving now.",
    "Can you pick up some milk on your way home?",
    "Happy birthday! Hope you have a wonderful day.",
    "Are you free for lunch tomorrow?",
    "The meeting has been moved to 3pm.",
    "Just finished the report, will send it over shortly.",
    "Got your message, calling you back in a bit.",
    "Thanks for the help earlier, really appreciated it.",
    "Did you see the game last night? Amazing ending!",
    "Running a bit late, sorry. Be there around 7.",
    "Can you send me the address again? I forgot.",
    "Mom wants to know if you're coming for dinner Sunday.",
    "The package arrived this morning, thanks!",
    "I'll forward you the email when I get to my desk.",
    "Hey, did you get my voicemail earlier?",
    "Sorry I missed your call, was in a meeting.",
    "Do you want me to bring anything to the party?",
    "Just checking in to see how you're doing.",
    "Talk later, heading into class now.",
    "Can we reschedule for next week? I have a conflict.",
    "The kids are asking if we can go to the park.",
    "Great talking to you yesterday, let's catch up soon.",
    "Don't forget we have dinner reservations at 8.",
    "What do you want to do for the weekend?",
    "I got the tickets for Saturday, can't wait!",
    "Feeling much better today, thanks for checking.",
    "The car is in the shop, can you give me a lift?",
    "Just landed, will be at baggage claim in 20 mins.",
    "Please bring the charger, mine is at home.",
    "Will you be at the office tomorrow?",
    "Just saw your email, replying now.",
    "Sounds good, see you then!",
    "Let me know if you need anything else.",
    "Call me when you get a chance.",
    "Hope you feel better soon!",
    "Where did you park? Can't find you.",
    "Traffic is terrible, might be 20 mins late.",
    "Got held up at work. Start without me.",
    "Can you watch the dog this weekend?",
]

# ---------------------------------------------------------------------------


def make_synthetic_dataset(n: int, spam_ratio: float = 0.13) -> pd.DataFrame:
    n_spam = max(1, int(n * spam_ratio))
    n_ham = n - n_spam
    rows = (
        [(random.choice(SPAM_MSGS), "spam") for _ in range(n_spam)]
        + [(random.choice(HAM_MSGS), "ham") for _ in range(n_ham)]
    )
    random.shuffle(rows)
    return pd.DataFrame(rows, columns=["text", "label"])


def split_for_lab(
    df: pd.DataFrame, test_size: int = 200
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df.iloc[test_size:].reset_index(drop=True), df.iloc[:test_size].reset_index(drop=True)


def download_sms_spam() -> pd.DataFrame:
    import io, zipfile, urllib.request
    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    print("Descargando SMS Spam Collection desde UCI...")
    with urllib.request.urlopen(URL, timeout=30) as r:
        zf = zipfile.ZipFile(io.BytesIO(r.read()))
    with zf.open("SMSSpamCollection") as f:
        df = pd.read_csv(f, sep="\t", header=None, names=["label", "text"])
    return df[["text", "label"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera los datasets para Lab 1")
    parser.add_argument(
        "--real",
        default=None,
        help="Ruta al archivo SMSSpamCollection (tab-separated: label<TAB>text)",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Descarga el SMS Spam Collection desde UCI automáticamente",
    )
    parser.add_argument("--out", default="data", help="Directorio de salida")
    parser.add_argument(
        "--test-size", type=int, default=500, help="Nº de muestras en el conjunto de test"
    )
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(exist_ok=True)

    if args.download:
        df = download_sms_spam()
        print(f"Dataset real descargado: {len(df)} filas")
    elif args.real:
        df = pd.read_csv(args.real, sep="\t", header=None, names=["label", "text"])
        df = df[["text", "label"]]
        print(f"Dataset real cargado: {len(df)} filas")
    else:
        df = make_synthetic_dataset(n=1000)
        print(f"Dataset sintético generado: {len(df)} filas")

    train_df, test_df = split_for_lab(df, test_size=args.test_size)

    public_test = test_df[["text"]].copy()
    public_test.insert(0, "id", range(len(public_test)))

    private_labels = pd.DataFrame(
        {"id": range(len(test_df)), "label": test_df["label"].values}
    )

    train_df.to_csv(out / "train.csv", index=False)
    public_test.to_csv(out / "public_test.csv", index=False)
    private_labels.to_csv(out / "private_labels.csv", index=False)

    print(f"  train.csv          → {len(train_df)} filas")
    print(f"  public_test.csv    → {len(public_test)} filas (sin labels)")
    print(f"  private_labels.csv → {len(private_labels)} filas (PRIVADO — no compartir)")
    print(f"\nDistribución train : {dict(train_df['label'].value_counts())}")
    print(f"Distribución test  : {dict(test_df['label'].value_counts())}")


if __name__ == "__main__":
    main()
