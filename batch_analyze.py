"""
Standalone CLI utility to run sentiment analysis over a CSV file of
reviews without needing the Flask server running.

Usage:
    python batch_analyze.py data/sample_reviews.csv --column text --out results.csv
"""

import argparse
import pandas as pd

from models.sentiment_model import analyze_batch


def main():
    parser = argparse.ArgumentParser(description="Bulk sentiment analysis over a CSV file")
    parser.add_argument("input_csv", help="Path to input CSV file")
    parser.add_argument("--column", default="text", help="Name of the text column (default: text)")
    parser.add_argument("--out", default="results.csv", help="Path to write results CSV")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    if args.column not in df.columns:
        raise SystemExit(f"Column '{args.column}' not found. Available columns: {list(df.columns)}")

    texts = df[args.column].astype(str).tolist()
    print(f"Analyzing {len(texts)} rows...")
    results = analyze_batch(texts)

    df["sentiment_label"] = [r["label"] for r in results]
    df["sentiment_score"] = [r["score"] for r in results]

    df.to_csv(args.out, index=False)
    print(f"Done. Results written to {args.out}")
    print(df["sentiment_label"].value_counts())


if __name__ == "__main__":
    main()
