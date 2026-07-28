"""
Database layer for the Sentiment Analyzer.

Tries to connect to a real MongoDB instance. If MongoDB is not reachable
(no server running, wrong URI, etc.), it transparently falls back to an
in-memory store so the app still works end-to-end during development or
demos without requiring MongoDB to be installed.
"""

import itertools
from datetime import datetime, timezone

from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError


class InMemoryCollection:
    """A tiny drop-in replacement for a pymongo Collection.

    Only implements the handful of methods this project needs:
    insert_one, insert_many, find, aggregate (group-by-label-count only).
    """

    def __init__(self):
        self._docs = []
        self._id_counter = itertools.count(1)

    def insert_one(self, doc):
        doc = dict(doc)
        doc["_id"] = next(self._id_counter)
        self._docs.append(doc)
        return doc

    def insert_many(self, docs):
        inserted = []
        for doc in docs:
            inserted.append(self.insert_one(doc))
        return inserted

    def find(self, query=None, *args, **kwargs):
        query = query or {}
        results = self._docs
        for key, value in query.items():
            results = [d for d in results if d.get(key) == value]
        # Return a shallow copy so callers can't mutate internal state
        return [dict(d) for d in results]

    def count_documents(self, query=None):
        return len(self.find(query))

    def aggregate(self, pipeline):
        """Supports the one aggregation this project uses:
        [{"$group": {"_id": "$label", "count": {"$sum": 1}}}]
        """
        counts = {}
        for doc in self._docs:
            label = doc.get("label", "unknown")
            counts[label] = counts.get(label, 0) + 1
        return [{"_id": label, "count": count} for label, count in counts.items()]


def get_collection(mongo_uri="mongodb://localhost:27017/",
                    db_name="sentiment_db",
                    collection_name="reviews",
                    timeout_ms=1500):
    """
    Returns (collection, backend_name) where backend_name is
    'mongodb' or 'in-memory' depending on what's actually available.
    """
    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=timeout_ms)
        client.admin.command("ping")  # forces a real connection attempt
        db = client[db_name]
        print(f"[db] Connected to MongoDB at {mongo_uri}")
        return db[collection_name], "mongodb"
    except ServerSelectionTimeoutError:
        print(f"[db] Could not reach MongoDB at {mongo_uri} — "
              f"falling back to in-memory storage. Data will NOT persist "
              f"across app restarts.")
        return InMemoryCollection(), "in-memory"


def utcnow():
    return datetime.now(timezone.utc)
