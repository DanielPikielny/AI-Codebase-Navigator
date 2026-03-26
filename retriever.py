from __future__ import annotations
import hashlib
import json
import logging
import os
import pickle
import re
import warnings
from collections import defaultdict
import faiss
import numpy as np

logger = logging.getLogger(__name__)

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False
    logger.warning(
        "rank-bm25 not installed. Hybrid search will fall back to semantic-only. "
        "Run: pip install rank-bm25"
    )

INDEX_CACHE_ROOT = ".index_cache"
_SPLIT_RE = re.compile(r"[^a-zA-Z0-9_]+")

def _tokenise(text: str) -> list:
    """
    Lowercase, split on non-alphanumeric/underscore boundaries.
    Underscore-joined identifiers are kept whole AND split on underscore,
    so 'embed_text' matches both 'embed' and 'text'.
    """
    raw    = _SPLIT_RE.split(text.lower())
    tokens = []
    for tok in raw:
        if len(tok) <= 1:
            continue
        tokens.append(tok)
        if "_" in tok:
            tokens.extend(p for p in tok.split("_") if len(p) > 1)
    return tokens

def _rrf(rankings: list, k: int = 60) -> list:
    """
    Combine ranked lists of chunk indices using RRF.
    Returns [(chunk_idx, rrf_score)] sorted descending.
    k=60 is the standard constant from the original RRF paper.
    """
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            scores[idx] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

class CodeRetriever:
    def __init__(self, dimension: int = 1536):
        self.dimension = dimension
        self.index     = faiss.IndexFlatL2(dimension)
        self.metadata  = []
        self._bm25     = None

    def add(self, embeddings: list, chunks: list) -> None:
        vectors = np.array(embeddings, dtype="float32")
        self.index.add(vectors)
        self.metadata.extend(chunks)
        self._bm25 = None
        if HAS_BM25:
            corpus     = [_tokenise(c["text"]) for c in self.metadata]
            self._bm25 = BM25Okapi(corpus)

    def _ensure_bm25(self) -> bool:
        """
        Ensure BM25 index exists. Returns True if usable, False otherwise.
        FIX-⑩: returns False (with a warning) instead of raising ImportError.
        """
        if not HAS_BM25:
            warnings.warn(
                "rank-bm25 is not installed; keyword search is unavailable. "
                "Hybrid search is using semantic-only results. "
                "Run: pip install rank-bm25",
                RuntimeWarning,
                stacklevel=3,
            )
            return False
        if self._bm25 is None:
            if not self.metadata:
                return False
            corpus     = [_tokenise(c["text"]) for c in self.metadata]
            self._bm25 = BM25Okapi(corpus)
        return True

    def semantic_search(self, query_embedding: list, k: int = 10) -> list:
        """Return [(chunk_idx, l2_distance)] from FAISS."""
        if not self.metadata:
            return []
        q = np.array([query_embedding], dtype="float32")
        dists, idxs = self.index.search(q, min(k, len(self.metadata)))
        return [(int(i), float(d)) for d, i in zip(dists[0], idxs[0]) if i >= 0]

    def keyword_search(self, query: str, k: int = 10) -> list:
        """
        Return [(chunk_idx, bm25_score)] sorted descending.
        Returns [] if BM25 is unavailable (FIX-⑩).
        """
        if not self._ensure_bm25():
            return []
        tokens = _tokenise(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(idx, float(sc)) for idx, sc in ranked[:k] if sc > 0]

    def hybrid_search(
        self,
        query:           str,
        query_embedding: list,
        k:       int = 5,
        fetch_k: int = 20,
    ) -> list:
        """
        Fuse semantic and keyword rankings with RRF, return top-k enriched chunks.
        Falls back to semantic-only (with a warning) if BM25 is unavailable.
        """
        sem_hits = self.semantic_search(query_embedding, k=fetch_k)
        kw_hits  = self.keyword_search(query, k=fetch_k)
        sem_ranking = [idx for idx, _ in sem_hits]
        kw_ranking  = [idx for idx, _ in kw_hits]
        if kw_ranking:
            fused = _rrf([sem_ranking, kw_ranking])[:k]
        else:
            fused = [(idx, 1.0 / (60 + rank)) for rank, idx in enumerate(sem_ranking[:k])]

        sem_dist = {idx: dist for idx, dist in sem_hits}
        kw_score = {idx: sc   for idx, sc   in kw_hits}

        results = []
        for idx, rrf_score in fused:
            if 0 <= idx < len(self.metadata):
                chunk                  = dict(self.metadata[idx])
                dist                   = sem_dist.get(idx, 4.0)
                chunk["score"]         = float(dist)
                chunk["similarity"]    = max(0.0, 1.0 - dist / 4.0)
                chunk["bm25_score"]    = kw_score.get(idx, 0.0)
                chunk["rrf_score"]     = rrf_score
                signals = []
                if idx in sem_dist: signals.append("semantic")
                if idx in kw_score: signals.append("keyword")
                chunk["match_signals"] = signals
                results.append(chunk)
        return results

    def search(self, query_embedding: list, k: int = 5) -> list:
        """Semantic-only search — kept for backward compatibility."""
        return [
            {
                **dict(self.metadata[idx]),
                "score":         dist,
                "similarity":    max(0.0, 1.0 - dist / 4.0),
                "match_signals": ["semantic"],
            }
            for idx, dist in self.semantic_search(query_embedding, k=k)
            if 0 <= idx < len(self.metadata)
        ]

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "metadata.pkl"), "wb") as f:
            pickle.dump({"metadata": self.metadata, "dimension": self.dimension}, f)

    @classmethod
    def load(cls, path: str) -> "CodeRetriever":
        with open(os.path.join(path, "metadata.pkl"), "rb") as f:
            payload = pickle.load(f)
        dimension    = payload.get("dimension", 1536)
        obj          = cls(dimension=dimension)
        obj.index    = faiss.read_index(os.path.join(path, "index.faiss"))
        obj.metadata = payload["metadata"]
        if HAS_BM25 and obj.metadata:
            obj._bm25 = BM25Okapi([_tokenise(c["text"]) for c in obj.metadata])
        return obj

    @staticmethod
    def _repo_fingerprint(repo_path: str, extensions: set) -> str:
        """
        FIX-④: fingerprint now includes CHUNK_SIZE and EMBED_MODEL so that
        changing chunking config or switching embedding models forces a rebuild.
        """
        from embeddings import CHUNK_SIZE, EMBED_MODEL

        skip = {
            "__pycache__", "node_modules", ".venv", "venv",
            "dist", "build", ".next", "target",
        }
        sig = [
            f"chunk_size:{CHUNK_SIZE}",
            f"embed_model:{EMBED_MODEL}",
        ]
        for root, dirs, files in os.walk(repo_path):
            dirs[:] = sorted(
                d for d in dirs if d not in skip and not d.startswith(".")
            )
            for fname in sorted(files):
                if os.path.splitext(fname)[1].lower() in extensions:
                    p = os.path.join(root, fname)
                    try:
                        s = os.stat(p)
                        sig.append(f"{p}:{s.st_mtime:.3f}:{s.st_size}")
                    except OSError:
                        pass
        return hashlib.md5(json.dumps(sig).encode()).hexdigest()

    @classmethod
    def build_or_load(
        cls,
        repo_path:        str,
        create_chunks_fn,
        embed_chunks_fn,
        extensions:       set,
        progress_fn=None,
    ) -> "CodeRetriever":
        def _log(msg: str):
            if progress_fn:
                progress_fn(msg)

        fp        = cls._repo_fingerprint(repo_path, extensions)
        cache_dir = os.path.join(INDEX_CACHE_ROOT, fp)

        if os.path.isfile(os.path.join(cache_dir, "index.faiss")):
            _log("Loading cached index…")
            return cls.load(cache_dir)

        _log("Loading and chunking source files…")
        chunks = create_chunks_fn(repo_path)

        if not chunks:
            raise ValueError(
                f"No supported source files found in '{repo_path}'. "
                "Check the path and ensure it contains .py / .js / .ts / .java / .cpp files."
            )

        _log(f"Embedding {len(chunks)} chunks…")
        embeddings = embed_chunks_fn(chunks)

        obj = cls()
        obj.add(embeddings, chunks)

        _log("Saving index to disk…")
        obj.save(cache_dir)

        _log("Index ready.")
        return obj