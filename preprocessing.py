"""
buscador de tweets

Pipeline:
  texto crudo ─► limpieza (Regex / spaCy) ─► vocabulario ─► TF-IDF ─► coseno
"""

import re
import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from dataset import TwitterData

data   = TwitterData.test_only(test_n=200)
CORPUS = data.test_texts
LABELS = data.test_labels
print(f"{len(CORPUS)} tweets  |  {sum(LABELS)} positivos · {len(LABELS)-sum(LABELS)} negativos\n")

# PREPROCESAMIENTO

_URL_RE    = re.compile(r"http\S+|www\.\S+")
_MENTION   = re.compile(r"@\w+")
_HASHTAG   = re.compile(r"#(\w+)")
_NON_ALPHA = re.compile(r"[^a-z\s]")
_nlp       = spacy.load("en_core_web_sm", disable=["ner", "parser"])
_KEEP_POS  = {"ADJ", "VERB", "ADV", "NOUN"}


def regex_clean(text: str) -> list[str]:
    text = text.lower()
    text = _URL_RE.sub("", text)
    text = _MENTION.sub("", text)
    text = _HASHTAG.sub(r"\1", text)   # #tech → "tech"
    text = _NON_ALPHA.sub(" ", text)
    return [t for t in text.split() if len(t) > 1]

def spacy_clean(text: str) -> list[str]:
    text = _URL_RE.sub("", text)
    text = _MENTION.sub("", text)
    text = _HASHTAG.sub(r"\1", text)
    doc  = _nlp(text.lower())
    return [
        tok.lemma_ for tok in doc
        if tok.pos_ in _KEEP_POS and not tok.is_stop and tok.is_alpha and len(tok) > 1
    ]

def preprocesar(text: str) -> str:
    return " ".join(regex_clean(text))

# # Seleccionar tweets que tengan @ o # para que el cleaning sea visible
# MUESTRA = [i for i, t in enumerate(CORPUS) if re.search(r"[@#]", t)][:5]

# for i in MUESTRA:
#     raw     = CORPUS[i]
#     r_tok   = regex_clean(raw)
#     s_tok   = spacy_clean(raw)
#     print(f"\n  [{i:02d}] {raw!r}")
#     print(f"       Regex → {r_tok}")
#     print(f"       spaCy → {s_tok}")

corpus_limpio = [preprocesar(t) for t in CORPUS]

vectorizer = TfidfVectorizer()
vectorizer.fit(corpus_limpio)

feature_names = vectorizer.get_feature_names_out()

print(f"\n  Vocabulario aprendido: {len(feature_names)} términos únicos")
print(f"  Primeros 15: {feature_names[:15].tolist()}")

tfidf_matrix = vectorizer.transform(corpus_limpio)   # (n_tweets, n_terms)

print("\n\nBÚSQUEDA POR SIMILITUD COSENO")

#  - las queries deben usar vocabulario del corpus
#  - los resultados cambian mucho si se usa spaCy en lugar de RegexCleaner

QUERIES = [
    "tired school exam stress",
    "watching movie tonight",
    "miss friend love",
    "good morning happy",
]

for query in QUERIES:
    q_vec = vectorizer.transform([preprocesar(query)])
    sims  = cosine_similarity(q_vec, tfidf_matrix)[0]
    top3  = np.argsort(sims)[::-1][:3]

    print(f"\n  Query: {query!r}")
    for rank, idx in enumerate(top3, 1):
        #  - similitud 0 = ninguna palabra de la query aparece en ese tweet
        print(f"    {rank}. [{sims[idx]:.3f}]  {CORPUS[idx]}")
