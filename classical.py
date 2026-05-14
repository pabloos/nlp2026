"""
classical.py — TF-IDF + Naive Bayes vs Logistic Regression
"""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression

from dataset import LABEL_ENCODER, TwitterData
from plots import plot_report

SEED = 42

# ── funciones ─────────────────────────────────────────────────────────────────

def evaluate(clf, X_test, y_test, title):
    preds  = clf.predict(X_test)
    scores = clf.predict_proba(X_test)[:, 1]
    plot_report(y_test, preds, label_names=LABEL_ENCODER.names,
                y_score=scores, title=title)

def inspect(clf, scores, title, n=15):
    features = clf.steps[0][1].get_feature_names_out()
    top_pos  = np.argsort(scores)[-n:][::-1]
    top_neg  = np.argsort(scores)[:n]
    print(f"\n── {title}: top features ──")
    print(f"  {'#':>3}  {'positive':<20} {'score':>7}    {'negative':<20} {'score':>7}")
    print(f"  {'─'*60}")
    for rank, (i, j) in enumerate(zip(top_pos, top_neg), 1):
        print(f"  {rank:>3}  {features[i]:<20} {scores[i]:>+7.3f}    {features[j]:<20} {scores[j]:>+7.3f}")

def predict(clf, text):
    label_id = clf.predict([text])[0]
    prob     = clf.predict_proba([text])[0][label_id]
    return LABEL_ENCODER.int2str(int(label_id)), prob

# ── ejecución ─────────────────────────────────────────────────────────────────

SAMPLE_TEXTS = [
    "I can't believe how bad this movie was, total waste of time",
]

data = TwitterData()

nb = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), sublinear_tf=True)),
    ("nb",    ComplementNB(alpha=0.1)),
])
nb.fit(data.train_texts, data.train_labels)

lr = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), sublinear_tf=True)),
    ("lr",    LogisticRegression(C=1.0, max_iter=1000, solver="saga", random_state=SEED)),
])
lr.fit(data.train_texts, data.train_labels)

evaluate(nb, data.test_texts, data.test_labels, "Naive Bayes")
evaluate(lr, data.test_texts, data.test_labels, "Logistic Regression")

nb_scores = nb.steps[-1][1].feature_log_prob_[1] - nb.steps[-1][1].feature_log_prob_[0]
inspect(nb, nb_scores, "Naive Bayes")
inspect(lr, lr.steps[-1][1].coef_[0], "Logistic Regression")

for text in SAMPLE_TEXTS:
    print(f"\nTweet: {text!r}\n")
    label, prob = predict(lr, text)
    print(f"  Logistic Regression: {label} ({prob:.1%})")
    label, prob = predict(nb, text)
    print(f"  Naive Bayes:         {label} ({prob:.1%})")
