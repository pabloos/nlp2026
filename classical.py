"""
classical.py — TF-IDF + Naive Bayes vs Logistic Regression
"""

from pathlib import Path

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression

from dataset import LABEL_ENCODER, TwitterData
from plots import plot_report

# ── configuración ─────────────────────────────────────────────────────────────

SEED       = 42
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── modelos ───────────────────────────────────────────────────────────────────

class NaiveBayesModel:
    checkpoint = OUTPUT_DIR / "naivebayes_model.joblib"

    def load_or_train(self, X_train, y_train):
        if self.checkpoint.exists():
            print("[nb] checkpoint found — loading …")
            self.clf = joblib.load(self.checkpoint)
            return
        print("[nb] training …")
        self.clf = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), sublinear_tf=True)),
            ("nb",    ComplementNB(alpha=0.1)),
        ])
        self.clf.fit(X_train, y_train)
        joblib.dump(self.clf, self.checkpoint)

    def evaluate(self, X_test, y_test):
        preds  = self.clf.predict(X_test)
        scores = self.clf.predict_proba(X_test)[:, 1]
        plot_report(y_test, preds, label_names=LABEL_ENCODER.names,
                    y_score=scores, title="Naive Bayes")

    def inspect(self, n: int = 15):
        features  = self.clf.steps[0][1].get_feature_names_out()
        log_probs = self.clf.steps[-1][1].feature_log_prob_
        ratio     = log_probs[1] - log_probs[0]
        top_pos   = np.argsort(ratio)[-n:][::-1]
        top_neg   = np.argsort(ratio)[:n]
        print(f"\n── Naive Bayes: top features ──")
        print(f"  {'#':>3}  {'positive':<20} {'score':>7}    {'negative':<20} {'score':>7}")
        print(f"  {'─'*60}")
        for rank, (i, j) in enumerate(zip(top_pos, top_neg), 1):
            print(f"  {rank:>3}  {features[i]:<20} {ratio[i]:>+7.3f}    {features[j]:<20} {ratio[j]:>+7.3f}")

    def predict(self, text: str):
        label_id = self.clf.predict([text])[0]
        prob     = self.clf.predict_proba([text])[0][label_id]
        return LABEL_ENCODER.int2str(int(label_id)), prob


class LogisticModel:
    checkpoint = OUTPUT_DIR / "logistic_model.joblib"

    def load_or_train(self, X_train, y_train):
        if self.checkpoint.exists():
            print("[lr] checkpoint found — loading …")
            self.clf = joblib.load(self.checkpoint)
            return
        print("[lr] training …")
        self.clf = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), sublinear_tf=True)),
            ("lr",    LogisticRegression(C=1.0, max_iter=1000, solver="saga", random_state=SEED)),
        ])
        self.clf.fit(X_train, y_train)
        joblib.dump(self.clf, self.checkpoint)

    def evaluate(self, X_test, y_test):
        preds  = self.clf.predict(X_test)
        scores = self.clf.predict_proba(X_test)[:, 1]
        plot_report(y_test, preds, label_names=LABEL_ENCODER.names,
                    y_score=scores, title="Logistic Regression")

    def inspect(self, n: int = 15):
        features = self.clf.steps[0][1].get_feature_names_out()
        coefs    = self.clf.steps[-1][1].coef_[0]
        top_pos  = np.argsort(coefs)[-n:][::-1]
        top_neg  = np.argsort(coefs)[:n]
        print(f"\n── Logistic Regression: top features ──")
        print(f"  {'#':>3}  {'positive':<20} {'coef':>7}    {'negative':<20} {'coef':>7}")
        print(f"  {'─'*60}")
        for rank, (i, j) in enumerate(zip(top_pos, top_neg), 1):
            print(f"  {rank:>3}  {features[i]:<20} {coefs[i]:>+7.3f}    {features[j]:<20} {coefs[j]:>+7.3f}")

    def predict(self, text: str):
        label_id = self.clf.predict([text])[0]
        prob     = self.clf.predict_proba([text])[0][label_id]
        return LABEL_ENCODER.int2str(int(label_id)), prob


# ── ejecución ─────────────────────────────────────────────────────────────────

SAMPLE_TEXTS = [
    "I can't believe how bad this movie was, total waste of time",
]

data = TwitterData()

nb = NaiveBayesModel()
lr = LogisticModel()

nb.load_or_train(data.train_texts, data.train_labels)
lr.load_or_train(data.train_texts, data.train_labels)

nb.evaluate(data.test_texts, data.test_labels)
lr.evaluate(data.test_texts, data.test_labels)
nb.inspect()
lr.inspect()

for text in SAMPLE_TEXTS:
    print(f"\nTweet: {text!r}\n")
    label, prob = lr.predict(text)
    print(f"  Logistic Regression: {label} ({prob:.1%})")
    label, prob = nb.predict(text)
    print(f"  Naive Bayes:         {label} ({prob:.1%})")
