"""
DSPy demo — llama3.2:3b via Ollama
Requisito: ollama serve && pip install dspy-ai
"""
import dspy

lm = dspy.LM("ollama_chat/llama3.2:3b",
             api_base="http://localhost:11434", temperature=0)
dspy.configure(lm=lm)

# ── Firma: describe la tarea, DSPy escribe el prompt ──────────────────────────
class AspectSentiment(dspy.Signature):
    """Classify a smartphone review on quality and delivery."""
    review:   str = dspy.InputField()
    quality:  str = dspy.OutputField(desc="positive or negative")
    delivery: str = dspy.OutputField(desc="positive or negative")

predictor = dspy.Predict(AspectSentiment)

# ── Zero-shot ─────────────────────────────────────────────────────────────────
r = predictor(review="Phone arrived next day, well packaged. "
                      "But battery barely lasts 4 hours and screen has scratches.")
print(f"quality: {r.quality}  |  delivery: {r.delivery}")

# ── Prompt que DSPy generó automáticamente ────────────────────────────────────
print("\n── prompt generado ──")
print(dspy.inspect_history(n=1))

# ── BootstrapFewShot con 4 ejemplos hardcoded (~15 s) ─────────────────────────
examples = [
    dspy.Example(review="Mint condition, great camera, fast shipping.",
                 quality="positive", delivery="positive").with_inputs("review"),
    dspy.Example(review="Cracked screen on arrival, took 2 weeks to arrive.",
                 quality="negative", delivery="negative").with_inputs("review"),
    dspy.Example(review="Battery dead, but arrived next day well packaged.",
                 quality="negative", delivery="positive").with_inputs("review"),
    dspy.Example(review="Works perfectly. Delivered in 24h.",
                 quality="positive", delivery="positive").with_inputs("review"),
]

metric   = lambda ex, pr, **_: int(ex.quality == pr.quality and ex.delivery == pr.delivery)
compiled = dspy.BootstrapFewShot(metric=metric, max_bootstrapped_demos=2)\
              .compile(predictor, trainset=examples)

r2 = compiled(review="Phone came scratched. Delivery was fast though.")
print(f"quality: {r2.quality}  |  delivery: {r2.delivery}")

print("\n── prompt compilado (con demos automáticos) ──")
print(dspy.inspect_history(n=1))
