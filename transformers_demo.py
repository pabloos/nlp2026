import os, warnings
warnings.filterwarnings("ignore")
os.environ["USE_TF"] = "0"

from transformers import (
    BertTokenizer, BertModel,
    GPT2Tokenizer, GPT2LMHeadModel,
    pipeline,
    logging as hf_logging,
)
hf_logging.set_verbosity_error()

import torch
import torch.nn as nn
import numpy as np

FRASE1 = ["the", "bank", "is", "closed"]
IDX    = FRASE1.index("bank")

def coseno(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# ── Bloque 1: Atención ────────────────────────────────────────────────────────
# Attention(Q,K,V) = softmax( Q·Kᵀ / √d ) · V
# El output de "banco" mezcla información de TODOS los tokens — incluidos los posteriores.
# Con pesos aleatorios no hay semántica, pero la bidireccionalidad es estructural.

torch.manual_seed(42)

attn = nn.MultiheadAttention(embed_dim=8, num_heads=1, batch_first=True)
X    = torch.randn(1, len(FRASE1), 8)   # (batch, seq, dim)
with torch.no_grad():
    _, p = attn(X, X, X)                # Q=K=V=X → self-attention

pesos_bank = p[0, IDX]                                          # fila de "bank" en la matriz de atención
pesos      = {t: round(float(v), 2) for t, v in zip(FRASE1, pesos_bank)}

print("Atención   pesos de 'bank':", pesos)

# ── Bloque 2: Transformer ─────────────────────────────────────────────────────
# Apila atención multicabeza + feed-forward + normalización de capa.
# La arquitectura es idéntica a BERT/GPT — la diferencia es de escala.
print("\nTransformer  1 capa dim=16: 2,224 params  |  BERT: 110 M  |  GPT-3: 175 B")

# ── Bloque 3: Modelos pre-entrenados ─────────────────────────────────────────

# BERT — encoder bidireccional pre-entrenado en texto masivo
print("\nCargando BERT...")
tok_bert = BertTokenizer.from_pretrained("bert-base-uncased")
mdl_bert = BertModel.from_pretrained("bert-base-uncased")
mdl_bert.eval()

inp1  = tok_bert("I went to the bank to withdraw money", return_tensors="pt")
toks1 = tok_bert.convert_ids_to_tokens(inp1["input_ids"][0])
with torch.no_grad():
    emb1 = mdl_bert(**inp1).last_hidden_state[0, toks1.index("bank")].numpy()

inp2  = tok_bert("I sat on the bank of the river", return_tensors="pt")
toks2 = tok_bert.convert_ids_to_tokens(inp2["input_ids"][0])
with torch.no_grad():
    emb2 = mdl_bert(**inp2).last_hidden_state[0, toks2.index("bank")].numpy()

sim_bert = coseno(emb1, emb2)
print(f"BERT     sim = {sim_bert:.3f}")

# GPT-2 — decoder autorregresivo: predice el siguiente token de izquierda a derecha
print("\nCargando GPT-2...")
tok_gpt = GPT2Tokenizer.from_pretrained("gpt2")
mdl_gpt = GPT2LMHeadModel.from_pretrained("gpt2")
mdl_gpt.eval()

for prompt in ["I went to the bank to", "I sat on the bank of the"]:
    ids = tok_gpt.encode(prompt, return_tensors="pt")
    with torch.no_grad():
        out = mdl_gpt.generate(ids, max_new_tokens=8, do_sample=False,
                                pad_token_id=tok_gpt.eos_token_id)
    print(f'"{tok_gpt.decode(out[0], skip_special_tokens=True)}"')

# Pipeline — API de alto nivel: tokeniza, infiere y decodifica en una línea
print()
clf = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
for txt in ["I love sitting on the bank of the river at sunset",
            "The bank charged me hidden fees without any warning"]:
    r = clf(txt)[0]
    print(f"  {r['label']:<8} {r['score']:.2f}  '{txt}'")

