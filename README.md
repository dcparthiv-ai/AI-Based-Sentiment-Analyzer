# AI-Based Sentiment Analyzer

Analyze social media posts or product reviews and classify them as
**positive / negative / neutral** using a Hugging Face transformer model,
with results stored in MongoDB (or an automatic in-memory fallback) and
visualized on a live dashboard.

## Tech Stack
- **Python** + **Flask** — web server / REST API
- **Hugging Face Transformers** — `cardiffnlp/twitter-roberta-base-sentiment-latest`
- **MongoDB** (via `pymongo`) — persistence, with automatic in-memory fallback if no DB is available
- **Chart.js** — dashboard visualizations (served via Flask + Jinja template)

## Project Structure
```
sentiment-analyzer/
├── app.py                     # Flask app & API routes
├── batch_analyze.py           # CLI tool for offline bulk CSV analysis
├── models/
│   └── sentiment_model.py     # Hugging Face pipeline wrapper
├── db/
│   └── mongo_client.py        # Mongo connection + in-memory fallback
├── templates/
│   └── index.html             # Dashboard UI (Chart.js)
├── data/
│   └── sample_reviews.csv     # Sample data for testing batch upload
├── requirements.txt
└── README.md
```

## Setup

1. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **(Optional) Start MongoDB**
   If you have MongoDB installed locally, just start it — the app will
   detect and use it automatically:
   ```bash
   mongod --dbpath /path/to/data
   ```
   If MongoDB isn't running, the app **automatically falls back** to an
   in-memory store so you can still use every feature during development
   or a demo. You'll just lose data on restart, and you'll see a message
   like:
   ```
   [db] Could not reach MongoDB at mongodb://localhost:27017/ — falling back to in-memory storage.
   ```

4. **Run the app**
   ```bash
   python app.py
   ```
   Then open **http://localhost:5000** in your browser.

   > Note: the first request to `/analyze` will download the Hugging Face
   > model (a few hundred MB), which can take a minute or two. Subsequent
   > requests are fast since the model stays loaded in memory.

## API Reference

| Method | Route            | Description                                              |
|--------|------------------|-----------------------------------------------------------|
| GET    | `/`              | Dashboard UI                                               |
| GET    | `/health`        | Health check + which DB backend is active                |
| POST   | `/analyze`       | Analyze one text. Body: `{"text": "...", "source": "..."}` |
| POST   | `/analyze/batch` | Analyze many. JSON `{"texts": [...]}` or CSV file upload  |
| GET    | `/reviews`       | List stored reviews. Optional `?label=positive&limit=50`  |
| GET    | `/reports`       | Aggregated sentiment counts + average confidence          |

### Example: analyze a single review
```bash
curl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is amazing, best purchase ever!"}'
```
Response:
```json
{"label": "positive", "score": 0.9823}
```

### Example: bulk analyze via CSV upload
```bash
curl -X POST http://localhost:5000/analyze/batch \
  -F "file=@data/sample_reviews.csv"
```

### Example: offline bulk analysis (no server needed)
```bash
python batch_analyze.py data/sample_reviews.csv --column text --out results.csv
```

## How It Works
1. Text comes in via the dashboard, a `curl`/API call, or a CSV upload.
2. `models/sentiment_model.py` runs it through a pretrained RoBERTa model
   fine-tuned on tweets, returning a label (positive/neutral/negative)
   and a confidence score.
3. The result is stored in MongoDB (or the in-memory fallback) with a
   timestamp and source tag.
4. The `/reports` endpoint aggregates counts across all stored reviews;
   the dashboard polls it and renders a doughnut chart, bar chart, and a
   table of recent reviews.

## Possible Extensions
- Pull live posts from Twitter/Reddit APIs for real-time analysis
- Word cloud of frequent terms per sentiment class
- Sentiment trend-over-time line chart
- Flag low-confidence predictions for human review
- Docker Compose file bundling Flask + MongoDB for one-command deploy
