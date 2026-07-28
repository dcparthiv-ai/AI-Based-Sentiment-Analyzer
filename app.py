"""
AI-Based Sentiment Analyzer
Flask + Hugging Face Transformers + MongoDB (with in-memory fallback)

Run with:
    python app.py
Then open http://localhost:5000
"""

import csv
import io

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

from models.sentiment_model import analyze_text, analyze_batch
from db.mongo_client import get_collection, utcnow

app = Flask(__name__)
CORS(app)

collection, DB_BACKEND = get_collection()


@app.route("/")
def index():
    return render_template("index.html", db_backend=DB_BACKEND)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "db_backend": DB_BACKEND})


@app.route("/analyze", methods=["POST"])
def analyze():
    """Analyze a single piece of text and store the result."""
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    source = data.get("source", "manual")

    if not text:
        return jsonify({"error": "Field 'text' is required"}), 400

    try:
        result = analyze_text(text)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    record = {
        "text": text,
        "label": result["label"],
        "score": result["score"],
        "source": source,
        "timestamp": utcnow().isoformat(),
    }
    collection.insert_one(record)

    return jsonify(result)


@app.route("/analyze/batch", methods=["POST"])
def analyze_batch_route():
    """
    Accepts either:
      - JSON: {"texts": ["review 1", "review 2", ...]}
      - multipart/form-data file upload of a CSV with a 'text' column
    """
    texts = []
    source = "batch"

    if "file" in request.files:
        file = request.files["file"]
        source = file.filename
        stream = io.StringIO(file.stream.read().decode("utf-8"))
        reader = csv.DictReader(stream)
        col = "text" if "text" in (reader.fieldnames or []) else reader.fieldnames[0]
        texts = [row[col] for row in reader if row.get(col)]
    else:
        data = request.get_json(silent=True) or {}
        texts = data.get("texts", [])
        source = data.get("source", "batch")

    if not texts:
        return jsonify({"error": "No texts found to analyze"}), 400

    results = analyze_batch(texts)
    records = []
    for text, result in zip(texts, results):
        record = {
            "text": text,
            "label": result["label"],
            "score": result["score"],
            "source": source,
            "timestamp": utcnow().isoformat(),
        }
        records.append(record)

    if records:
        collection.insert_many(records)

    return jsonify({"count": len(records), "results": records})


@app.route("/reviews", methods=["GET"])
def reviews():
    """Return stored reviews, most recent first, optionally filtered by label."""
    label = request.args.get("label")
    limit = int(request.args.get("limit", 100))

    query = {"label": label} if label else {}
    docs = collection.find(query)

    for d in docs:
        d.pop("_id", None)

    docs.sort(key=lambda d: d.get("timestamp", ""), reverse=True)
    return jsonify(docs[:limit])


@app.route("/reports", methods=["GET"])
def reports():
    """Aggregated counts per sentiment label, for charting."""
    agg = list(collection.aggregate(
        [{"$group": {"_id": "$label", "count": {"$sum": 1}}}]
    ))
    counts = {row["_id"]: row["count"] for row in agg}
    for label in ("positive", "neutral", "negative"):
        counts.setdefault(label, 0)

    total = sum(counts.values())
    avg_score = 0
    all_docs = collection.find({})
    if all_docs:
        avg_score = round(sum(d.get("score", 0) for d in all_docs) / len(all_docs), 4)

    return jsonify({
        "counts": counts,
        "total": total,
        "average_confidence": avg_score,
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
