"""
nlp_pipeline.py — POS · NER · SVO · LDA
"""

import webbrowser, pathlib
import spacy
from spacy import displacy
import gensim.corpora as corpora
from gensim.models import LdaModel

nlp = spacy.load("en_core_web_sm")

# ── Corpus ────────────────────────────────────────────────────────────────────
DOCS = [
    "Apple released the iPhone 15 in New York last September and Tim Cook called it the most advanced smartphone ever built.",
    "Google launched a new AI model trained on one trillion parameters that outperforms GPT-4 on every benchmark.",
    "Microsoft acquired a leading robotics company and announced a ten billion dollar investment in artificial intelligence.",
    "Tesla unveiled an electric car that achieves 400 miles on a single charge with a revolutionary new battery.",
    "Meta released a virtual reality headset with improved resolution that allows users to work and collaborate remotely.",
    "Real Madrid won the Champions League for the fifteenth time after defeating Manchester City in London.",
    "Novak Djokovic defeated Rafael Nadal in a dramatic five-set Wimbledon final watched by millions worldwide.",
    "The Olympic Games in Paris attracted four billion television viewers and broke every previous attendance record.",
    "The Boston Celtics won the NBA championship after an extraordinary season led by their young star players.",
    "Max Verstappen broke the Formula 1 record for most wins in a single season driving for Red Bull Racing.",
    "The European Union approved strict new regulations on artificial intelligence that will affect every tech company.",
    "The United Nations Security Council voted on a historic resolution to reduce global carbon emissions by 2035.",
    "The US President signed a landmark trade agreement with the European Commission after two years of negotiations.",
    "NATO leaders met in Brussels to discuss military support and agreed to increase defence spending significantly.",
    "The Spanish Parliament approved a new digital privacy law giving citizens full control over their personal data.",
]

example = nlp(DOCS[0])

print(f"Texto: {example.text}\n")

print("── POS Tagging ───────────────────────────────────────────")
for tok in example:
    if tok.pos_ in ("NOUN", "VERB", "ADJ", "PROPN") and not tok.is_stop:
        print(f"  {tok.text:<22} {tok.pos_:<8} → {tok.lemma_}")

print("\n── NER ───────────────────────────────────────────────────")
for ent in example.ents:
    print(f"  [{ent.label_:<10}] {ent.text}")

print("\n── Dependencias SVO (Subject-Verb-Object) ──────────────────────────────────────")
for tok in example:
    if tok.pos_ == "VERB":
        subjs = [t for t in tok.children if t.dep_ in ("nsubj", "nsubjpass")]
        objs  = [t for t in tok.children if t.dep_ in ("dobj", "attr", "pobj")]
        for s in subjs:
            for o in objs:
                print(f"  ({s.text}) —[{tok.lemma_}]→ ({o.text})")

print("\n── LDA ───────────────────────────────────────────────────")
KEEP   = {"NOUN", "PROPN", "VERB"}
tokens = [[t.lemma_.lower() for t in nlp(doc)
           if t.pos_ in KEEP and not t.is_stop and t.is_alpha and len(t) > 2]
          for doc in DOCS]

dictionary = corpora.Dictionary(tokens)
bow        = [dictionary.doc2bow(t) for t in tokens]
lda        = LdaModel(bow, id2word=dictionary, num_topics=3,
                      random_state=42, passes=40)

# ── Topic naming ──────────────────
TOPIC_NAMES = {0: '???', 1: '???', 2: '???'}

for i, topic in lda.print_topics(num_words=6):
    terms = [t.split("*")[1].strip().strip('"') for t in topic.split("+")]
    print(f"  T{i} '{TOPIC_NAMES[i]}': {', '.join(terms)}")

# ── Visualización ─────────────────────────────────────────────────────────────
for style, fname in [("ent", "output_ner.html"), ("dep", "output_dep.html")]:
    html = displacy.render(example, style=style, page=True)
    path = pathlib.Path(fname)
    path.write_text(html, encoding="utf-8")
    print(f"\n  {fname} guardado")
    webbrowser.open(path.resolve().as_uri())
