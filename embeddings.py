"""
Introducción a la API de Word Embeddings
"""
import numpy as np
import gensim.downloader as gensim_dl
from plots import plot_embedding_pca

print("Cargando GloVe 100d (Wikipedia + Gigaword)...")
glove = gensim_dl.load("glove-wiki-gigaword-100")
print(f"  {len(glove):,} palabras  ·  {glove.vector_size} dimensiones\n")

print("1. UNA PALABRA = UN VECTOR")
v = glove["game"]
print(f"  glove['game'] → array de {len(v)} números")
print(f"  primeros 8:   {v[:8].round(3)}")
print(f"  norma:        {np.linalg.norm(v):.3f}")

print("2. PALABRAS SIMILARES  (similitud coseno)")
for word in ["gameplay", "graphics", "fps"]:
    vecinos = [w for w, _ in glove.most_similar(word, topn=5)]
    print(f"  '{word}'  →  {vecinos}")
print()

print("3. SIMILITUD COSENO")
pares = [
    ("shooter", "fps",      "sinónimos de género"),
    ("good",    "great",    "sinónimos de valoración"),
    ("amazing", "terrible", "¿antónimos?"),
    ("game",    "economy",  "sin relación"),
]
for w1, w2, nota in pares:
    sim = glove.similarity(w1, w2)
    print(f"  sim('{w1}', '{w2}') = {sim:.3f}   # {nota}")
print()
print("  ⚠  'amazing' y 'terrible' son similares porque aparecen")
print("     en los mismos contextos gramaticales, no por significado.\n")

print("4. ARITMÉTICA DE VECTORES")
analogias = [
    (["king", "woman"], ["man"],    "el clásico"),
    (["paris", "spain"], ["france"], "geografía"),
    (["fps", "story"],  ["action"], "videojuegos"),
]
for pos, neg, nota in analogias:
    res = [w for w, _ in glove.most_similar(positive=pos, negative=neg, topn=3)]
    expr = " + ".join(pos) + " − " + " − ".join(neg)
    print(f"  {expr:<30} ≈  {res}   # {nota}")
print()

print("5. EL QUE NO ENCAJA  (doesnt_match)")
listas = [
    ["fps", "shooter", "rpg", "economics"],
    ["amazing", "great", "fun", "controls"],
]
for lista in listas:
    raro = glove.doesnt_match(lista)
    print(f"  {lista}  →  '{raro}'")
print()

print("6. GEOMETRÍA DEL ESPACIO (PCA 2D)")

grupos  = {
    "género":      ["fps", "shooter", "rpg", "racing", "strategy"],
    "mecánica":    ["gameplay", "controls", "graphics", "story", "multiplayer"],
    "sentimiento": ["amazing", "terrible", "boring", "fun", "broken"],
}
colores = {"género": "#58a6ff", "mecánica": "#3fb950", "sentimiento": "#f85149"}

plot_embedding_pca(grupos, glove, colores)

# print("\n  → palabras del mismo grupo tienden a quedar juntas")
# print("  → el modelo nunca vio etiquetas: aprendió la estructura del lenguaje\n")
# print("═" * 55)
# print("  En el notebook veremos cómo GloVe compara con un modelo")
# print("  entrenado específicamente sobre reseñas de videojuegos.")
# print("═" * 55)
