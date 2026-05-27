"""
Lab 6 — Amazon Aspect Sentiment con LLM local
==============================================
Tarea  : dada una reseña de Amazon, clasificar el sentimiento para:
           · quality  (calidad del producto)  → pos / neg
           · delivery (envío / entrega)        → pos / neg
Palanca: los prompts. No se entrena nada.
Métrica: F1 global = (F1_quality + F1_delivery) / 2

Uso:
    python lab6_student.py          # evalúa sobre 20 reseñas
    python lab6_student.py --all    # evalúa sobre todo el train (más lento)
"""
import os, argparse, warnings
os.environ['USE_TF'] = '0'
warnings.filterwarnings('ignore')

import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score
from transformers import pipeline
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

# ── Configuración ──────────────────────────────────────────────────────────────
ALUMNO = 'tu_nombre'              # <-- cambia esto
MODEL  = 'google/flan-t5-large'   # puedes probar flan-t5-xl si tienes más RAM

# ── Datos ──────────────────────────────────────────────────────────────────────
_DATA = Path(__file__).parent / 'labs' / 'lab6' / 'data'
train = pd.read_csv(_DATA / 'train.csv')

# ── Modelo ─────────────────────────────────────────────────────────────────────
print(f'Cargando {MODEL}...')
_gen = pipeline('text2text-generation', model=MODEL)
print('Listo.\n')

def call_llm(prompt: str) -> str:
    return _gen(prompt, max_new_tokens=10, do_sample=False)[0]['generated_text'].strip().lower()

def parse_answer(response: str) -> str:
    return 'pos' if 'positive' in response else 'neg'

# ── Evaluación ─────────────────────────────────────────────────────────────────
def evaluar(quality_prompt_fn, delivery_prompt_fn, n: int | None = 20) -> None:
    sample = train if n is None else train.sample(n=n, random_state=42)
    yq_true, yq_pred, yd_true, yd_pred = [], [], [], []
    for i, (_, row) in enumerate(sample.iterrows(), 1):
        q = parse_answer(call_llm(quality_prompt_fn(row['text'])))
        d = parse_answer(call_llm(delivery_prompt_fn(row['text'])))
        yq_true.append(row['quality']);  yq_pred.append(q)
        yd_true.append(row['delivery']); yd_pred.append(d)
        if i % 10 == 0:
            print(f'  {i}/{len(sample)}...')
    f1q = f1_score(yq_true, yq_pred, pos_label='pos')
    f1d = f1_score(yd_true, yd_pred, pos_label='pos')
    print(f'\nAlumno     : {ALUMNO}')
    print(f'F1 calidad : {f1q:.4f}')
    print(f'F1 entrega : {f1d:.4f}')
    print(f'F1 global  : {(f1q + f1d) / 2:.4f}  (sobre {len(sample)} reseñas)')

# ── Few-shot examples (listos para usar en tus prompts) ────────────────────────
_combos = [('pos', 'pos'), ('pos', 'neg'), ('neg', 'pos'), ('neg', 'neg')]
_rows   = [train[(train.quality == q) & (train.delivery == d)].iloc[0]
           for q, d in _combos]

QUALITY_EXAMPLES = '\n\n'.join(
    f"Review: {r.text[:200]}\nAnswer: {'positive' if q == 'pos' else 'negative'}"
    for r, (q, _) in zip(_rows, _combos)
)

DELIVERY_EXAMPLES = '\n\n'.join(
    f"Review: {r.text[:200]}\nAnswer: {'positive' if d == 'pos' else 'negative'}"
    for r, (_, d) in zip(_rows, _combos)
)

# ══════════════════════════════════════════════════════════════════════════════
#  ✏️  EDITA AQUÍ — modifica las dos funciones de prompt
# ══════════════════════════════════════════════════════════════════════════════

def quality_prompt(text: str) -> str:
    return f"""Is the product quality in this review positive or negative?
Review: {text}
Answer:"""


def delivery_prompt(text: str) -> str:
    return f"""Is the shipping or delivery in this review positive or negative? \
If not mentioned, say negative.
Review: {text}
Answer:"""


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', action='store_true',
                        help='Evalúa sobre todo el train en lugar de 20 reseñas')
    args = parser.parse_args()

    evaluar(quality_prompt, delivery_prompt, n=None if args.all else 20)
