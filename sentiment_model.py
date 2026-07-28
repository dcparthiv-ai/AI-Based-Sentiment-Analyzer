"""
Sentiment analysis wrapper around a Hugging Face transformers pipeline.

Uses cardiffnlp/twitter-roberta-base-sentiment-latest, which natively
classifies text into three classes: negative / neutral / positive.
This avoids having to bolt a neutral bucket onto a 2-class model.
"""

from transformers import pipeline

_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

_LABEL_MAP = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive",
    # Some model revisions already return human-readable labels
    "negative": "negative",
    "neutral": "neutral",
    "positive": "positive",
}

_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        print(f"[model] Loading {_MODEL_NAME} (first call only, may take a moment)...")
        _pipeline = pipeline("sentiment-analysis", model=_MODEL_NAME)
    return _pipeline


def analyze_text(text: str) -> dict:
    """Analyze a single string. Returns {label, score}."""
    if not text or not text.strip():
        raise ValueError("text must be a non-empty string")

    clf = _get_pipeline()
    raw = clf(text[:512])[0]  # truncate to model's practical limit
    label = _LABEL_MAP.get(raw["label"], raw["label"].lower())
    return {"label": label, "score": round(float(raw["score"]), 4)}


def analyze_batch(texts: list) -> list:
    """Analyze a list of strings in one batched call (much faster than
    calling analyze_text in a loop for large datasets)."""
    texts = [t[:512] for t in texts if t and t.strip()]
    if not texts:
        return []

    clf = _get_pipeline()
    raw_results = clf(texts)
    return [
        {"label": _LABEL_MAP.get(r["label"], r["label"].lower()),
         "score": round(float(r["score"]), 4)}
        for r in raw_results
    ]
