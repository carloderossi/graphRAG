import json
from datetime import datetime

class ComparisonLogger:
    def __init__(self, path="comparison_log.jsonl"):
        self.path = path

    def log(self, query, results):
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "query": query,
            "results": results
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")