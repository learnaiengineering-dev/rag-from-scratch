import math
import re
from collections import Counter

from .chunk import Chunk

_TOKEN = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.tf = [Counter(tokenize(c.text)) for c in chunks]
        self.lengths = [sum(t.values()) for t in self.tf]
        self.avg_len = (sum(self.lengths) / len(chunks)) if chunks else 0.0
        df: Counter[str] = Counter()
        for t in self.tf:
            df.update(t.keys())
        n = len(chunks)
        self.idf = {w: math.log(1 + (n - d + 0.5) / (d + 0.5)) for w, d in df.items()}

    def score(self, query_tokens: list[str], i: int) -> float:
        tf, length = self.tf[i], self.lengths[i]
        total = 0.0
        for w in query_tokens:
            f = tf.get(w, 0)
            if f:
                denom = f + self.k1 * (1 - self.b + self.b * length / self.avg_len)
                total += self.idf[w] * f * (self.k1 + 1) / denom
        return total

    def search(self, query: str, k: int = 3) -> list[tuple[Chunk, float]]:
        q = tokenize(query)
        scored = [(self.chunks[i], self.score(q, i)) for i in range(len(self.chunks))]
        scored = [s for s in scored if s[1] > 0]
        return sorted(scored, key=lambda s: s[1], reverse=True)[:k]

