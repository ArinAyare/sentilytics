"""Storage layer. MongoRepo for real use; MemoryRepo for tests / zero-setup demo."""
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

LABELS = ["negative", "neutral", "positive"]


def now():
    return datetime.now(timezone.utc)


def new_id():
    return uuid.uuid4().hex


class MemoryRepo:
    kind = "memory"

    def __init__(self):
        self.analyses, self.jobs = [], {}

    def _filter(self, label=None, job_id=None):
        return [d for d in self.analyses if (not label or d["label"] == label) and (not job_id or d.get("job_id") == job_id)]

    def add_analyses(self, docs):
        self.analyses.extend(docs)

    def list_analyses(self, skip=0, limit=20, label=None, job_id=None, newest_first=True):
        items = self._filter(label, job_id)
        key = (lambda d: d["created_at"]) if not job_id else (lambda d: d.get("row", 0))
        items = sorted(items, key=key, reverse=newest_first and not job_id)
        return items[skip:skip + limit]

    def count(self, label=None, job_id=None):
        return len(self._filter(label, job_id))

    def label_counts(self):
        c = Counter(d["label"] for d in self.analyses)
        return {l: c.get(l, 0) for l in LABELS}

    def avg_confidence(self):
        return sum(d["confidence"] for d in self.analyses) / len(self.analyses) if self.analyses else 0.0

    def daily_trend(self, days=30):
        since = now() - timedelta(days=days)
        agg = defaultdict(Counter)
        for d in self.analyses:
            if d["created_at"] >= since:
                agg[d["created_at"].strftime("%Y-%m-%d")][d["label"]] += 1
        return [{"date": k, **{l: agg[k].get(l, 0) for l in LABELS}} for k in sorted(agg)]

    def add_job(self, doc):
        self.jobs[doc["id"]] = doc

    def get_job(self, job_id):
        return self.jobs.get(job_id)

    def list_jobs(self, limit=20):
        return sorted(self.jobs.values(), key=lambda j: j["created_at"], reverse=True)[:limit]


class MongoRepo:
    kind = "mongodb"

    def __init__(self, uri, db_name):
        from pymongo import ASCENDING, DESCENDING, MongoClient
        self._asc, self._desc = ASCENDING, DESCENDING
        self.client = MongoClient(uri, serverSelectionTimeoutMS=4000)
        self.client.admin.command("ping")  # fail fast if unreachable
        db = self.client[db_name]
        self.a, self.j = db["analyses"], db["batch_jobs"]
        self.a.create_index([("created_at", DESCENDING)])
        self.a.create_index([("job_id", ASCENDING), ("row", ASCENDING)])
        self.a.create_index("label")

    @staticmethod
    def _q(label, job_id):
        q = {}
        if label:
            q["label"] = label
        if job_id:
            q["job_id"] = job_id
        return q

    def add_analyses(self, docs):
        if docs:
            self.a.insert_many([dict(d) for d in docs])  # copies: pymongo mutates with _id

    def list_analyses(self, skip=0, limit=20, label=None, job_id=None, newest_first=True):
        sort = [("row", self._asc)] if job_id else [("created_at", self._desc if newest_first else self._asc)]
        return list(self.a.find(self._q(label, job_id), {"_id": 0}).sort(sort).skip(skip).limit(limit))

    def count(self, label=None, job_id=None):
        return self.a.count_documents(self._q(label, job_id))

    def label_counts(self):
        c = {r["_id"]: r["n"] for r in self.a.aggregate([{"$group": {"_id": "$label", "n": {"$sum": 1}}}])}
        return {l: c.get(l, 0) for l in LABELS}

    def avg_confidence(self):
        r = list(self.a.aggregate([{"$group": {"_id": None, "avg": {"$avg": "$confidence"}}}]))
        return float(r[0]["avg"]) if r else 0.0

    def daily_trend(self, days=30):
        since = now() - timedelta(days=days)
        rows = self.a.aggregate([
            {"$match": {"created_at": {"$gte": since}}},
            {"$group": {"_id": {"d": {"$dateToString": {"format": "%Y-%m-%d", "date": "$created_at"}}, "l": "$label"}, "n": {"$sum": 1}}}])
        agg = defaultdict(dict)
        for r in rows:
            agg[r["_id"]["d"]][r["_id"]["l"]] = r["n"]
        return [{"date": k, **{l: agg[k].get(l, 0) for l in LABELS}} for k in sorted(agg)]

    def add_job(self, doc):
        self.j.insert_one(dict(doc))

    def get_job(self, job_id):
        return self.j.find_one({"id": job_id}, {"_id": 0})

    def list_jobs(self, limit=20):
        return list(self.j.find({}, {"_id": 0}).sort("created_at", self._desc).limit(limit))


def build_repo(settings):
    import logging
    if settings.mongo_uri:
        try:
            return MongoRepo(settings.mongo_uri, settings.mongo_db)
        except Exception as e:
            logging.getLogger("sentilytics").warning("MongoDB unavailable (%s) - using in-memory store", e)
    return MemoryRepo()
