# =========================
# 1. Exploración básica
# =========================
from datasets import load_dataset

dataset = load_dataset("bdstar/twitter-sentiment-analysis")

print(dataset)
print(dataset["train"][0])


# =========================
# 2. Distribución de clases
# =========================
from collections import Counter

labels = dataset["train"]["label"]

print(Counter(labels))


# =========================
# 3. Longitud de los textos
# =========================
lengths = [len(x.split()) for x in dataset["train"]["text"]]

print("Min length:", min(lengths))
print("Max length:", max(lengths))


# =========================
# 4. Inspección de texto (ruido)
# =========================
sample = dataset["train"]["text"][:20]

for text in sample:
    print(text)


# =========================
# 5. Detección de duplicados
# =========================
texts = dataset["train"]["text"]

duplicates = len(texts) - len(set(texts))

print("Duplicados:", duplicates)


# =========================
# 6. Texto vacío o basura
# =========================
empty_texts = [x for x in texts if len(x.strip()) == 0]

print("Ejemplos vacíos:", empty_texts[:10])


# =========================
# 7. Inspección de etiquetas
# =========================
for i in range(20):
    print(dataset["train"][i]["text"], "→", dataset["train"][i]["label"])


# =========================
# 8. Frecuencia de palabras
# =========================
words = []

for text in dataset["train"]["text"][:1000]:
    words.extend(text.lower().split())

word_counts = Counter(words)

print(word_counts.most_common(20))