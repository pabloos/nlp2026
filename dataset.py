"""
dataset.py — carga y preparación de datasets (compartido entre pipelines)
"""

from datasets import load_dataset, ClassLabel

SEED          = 42
TRAIN_SAMPLES = 40_000
TEST_SAMPLES  = 10_000

# names ordenados por id: names[0]="negative", names[1]="positive"
LABEL_ENCODER = ClassLabel(names=["negative", "positive"])

class TwitterData:
    """Carga y muestrea el dataset de Twitter desde HuggingFace."""

    DATASET_ID = "bdstar/twitter-sentiment-analysis"
    LABEL_MAP  = {"positive": 1, "negative": 0}

    @classmethod
    def raw(cls, split: str = "train"):
        """Devuelve el Dataset de HuggingFace sin procesar.

        Útil cuando se necesita la API completa de HF (`.select()`,
        `.train_test_split()`, `.map()`, `.set_format()`…)
        """
        return load_dataset(cls.DATASET_ID, split=split)

    def __init__(self, train_n: int = TRAIN_SAMPLES, test_n: int = TEST_SAMPLES):
        ds = load_dataset(self.DATASET_ID)
        self.train_texts, self.train_labels = self._sample(ds["train"], train_n, "train")
        self.test_texts,  self.test_labels  = self._sample(ds["test"],  test_n,  "test")

    @classmethod
    def test_only(cls, test_n: int = TEST_SAMPLES):
        obj = cls.__new__(cls)
        ds  = load_dataset(cls.DATASET_ID, split="test")
        obj.test_texts, obj.test_labels = obj._sample(ds, test_n, "test")
        return obj

    def _sample(self, split, n: int, name: str):
        sampled  = split.shuffle(seed=SEED).select(range(min(int(n * 1.2), len(split))))
        filtered = sampled.filter(lambda r: r["label"] in self.LABEL_MAP)
        return filtered["text"][:n], [self.LABEL_MAP[l] for l in filtered["label"][:n]]


AMAZON_DATASET_ID = "McAuley-Lab/Amazon-Reviews-2023"

def amazon_reviews(category: str = "Electronics",
                   n: int = 2_000, min_len: int = 50) -> list[dict]:
    """Carga n reseñas de cualquier categoría Amazon en streaming.

    Devuelve lista de dicts con text, rating y title. Para
    tareas de QA, embeddings o zero-shot donde se necesita el texto
    crudo con metadatos.

    Parameters
    ----------
    category : nombre de la categoría (e.g. "Electronics", "Books")
    n        : número máximo de reseñas a devolver
    min_len  : longitud mínima del texto en caracteres
    """
    ds = load_dataset(
        "json",
        data_files=f"hf://datasets/{AMAZON_DATASET_ID}/raw/review_categories/{category}.jsonl",
        split="train",
        streaming=True,
    )
    filtered = ds.filter(lambda r: r["text"] and len(r["text"]) > min_len)
    return [
        {"text": r["text"], "rating": r.get("rating", 3), "title": r.get("title", "")}
        for r in filtered.take(n)
    ]
