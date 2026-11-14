"""Retrieval implementations for Day 7 connecting Day 6 pgvector & embeddings."""
from __future__ import annotations

import json
import math
import os
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import psycopg
import torch
from dotenv import dotenv_values
from sentence_transformers import SentenceTransformer

from baseline import TfidfIndex, tokenize

DEFAULT_DAY6_ENV = Path(
    r"C:\hoctap\AI\VinAIthucchien\LAB\Lab6\K5-L3A-DAY06-NgoDucThinh-2A202603548-DataIngestion\starter\.env"
)
_ALLOWED_FILTER_KEYS = {"chapter", "source_id", "document_id", "year", "document_type"}
_CHAPTER_RE = re.compile(r"\bchương\s+(iv|iii|ii|i)\b", re.IGNORECASE)
_APPENDIX_RE = re.compile(r"\bphụ\s+lục\b", re.IGNORECASE)
_LEGAL_REF_RE = re.compile(
    r"(?:điều\s+\d+|khoản\s+\d+|mẫu\s+số\s+\d+|\d+\s*(?:giờ|ngày|tháng|năm|tuổi)|13/2023/nđ-cp)",
    re.IGNORECASE,
)
_VI_STOPWORDS = {
    "là", "gì", "của", "và", "các", "những", "trong", "cho", "theo", "về",
    "được", "có", "không", "khi", "nào", "bao", "nhiêu", "lâu", "từ", "đến",
    "tới", "tại", "với", "hoặc", "hay", "như", "thế", "nếu", "thì", "mà",
    "này", "đó", "họ", "ai", "việc", "phải", "cần", "trên", "dưới", "sau",
    "trước", "một", "hai", "ba", "bên", "đối", "với", "trường", "hợp",
}


def resolve_database_url(database_url: str | None = None) -> str:
    """Resolve DATABASE_URL from argument, Day 7 env/.env, or Day 6 starter/.env."""
    if database_url:
        return database_url
    url = os.getenv("DATABASE_URL")
    if url:
        return url
    starter_dir = Path(__file__).resolve().parent
    candidates = [
        starter_dir / ".env",
        starter_dir.parent / ".env",
        starter_dir.parent.parent / ".env",
        DEFAULT_DAY6_ENV,
    ]
    for path in candidates:
        if not path.is_file():
            continue
        vals = dotenv_values(path)
        candidate = (vals.get("DATABASE_URL") or "").strip()
        if candidate:
            os.environ["DATABASE_URL"] = candidate
            return candidate
    if not url:
        raise RuntimeError(
            "Thiếu DATABASE_URL. Hãy đặt biến môi trường DATABASE_URL hoặc tạo .env."
        )
    return url


def infer_filters_from_query(query: str) -> dict[str, Any] | None:
    """Reliably infer chapter filter only when explicitly named in the user query."""
    norm = unicodedata.normalize("NFC", query)
    m_chap = _CHAPTER_RE.search(norm)
    if m_chap:
        return {"chapter": f"Chương {m_chap.group(1).upper()}"}
    if _APPENDIX_RE.search(norm):
        return {"chapter": "Phụ lục"}
    return None


def matches_filters(chunk: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    """Check whether a Day 7 chunk dict satisfies all metadata filter clauses."""
    if not filters:
        return True
    meta = chunk.get("metadata") or {}
    for key, expected in filters.items():
        if key not in _ALLOWED_FILTER_KEYS:
            raise ValueError(f"Unsupported filter key: {key}")
        if key in ("source_id", "document_id"):
            actual = chunk.get("source_id")
        else:
            actual = meta.get(key, chunk.get(key))
        if actual != expected:
            return False
    return True


def _normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text).casefold()).strip()


def _min_max_norm(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return {k: (1.0 if hi > 0 else 0.0) for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


class Day6QueryEmbedder:
    """Query embedder matching Day 6 google/embeddinggemma-2 (256d, SearchQuery prompt)."""

    def __init__(
        self,
        model_name: str = "google/embeddinggemma-2",
        truncate_dim: int = 256,
    ) -> None:
        self.model_name = model_name
        self.truncate_dim = truncate_dim
        try:
            self._model = SentenceTransformer(
                model_name,
                truncate_dim=truncate_dim,
                model_kwargs={"dtype": torch.float32},
                local_files_only=True,
            )
        except Exception:
            self._model = SentenceTransformer(
                model_name,
                truncate_dim=truncate_dim,
                model_kwargs={"dtype": torch.float32},
            )
        dim = self._model.get_embedding_dimension()
        if dim is None:
            raise RuntimeError(f"Could not determine embedding dimensions for {model_name}")
        self.dimensions = int(dim)

    def embed_query(self, query: str) -> list[float]:
        vec = self._model.encode(
            query,
            prompt_name="SearchQuery",
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vec.tolist()


class TfidfRetriever:
    """Lexical TF-IDF retriever wrapping baseline.TfidfIndex with optional metadata filter."""

    strategy = "tfidf"

    def __init__(self, chunks: list[dict[str, Any]], infer_filter: bool = False) -> None:
        self.chunks = chunks
        self.infer_filter = infer_filter
        self._full_index = TfidfIndex(chunks)

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        active_filters = filters if filters is not None else (
            infer_filters_from_query(query) if self.infer_filter else None
        )
        if not active_filters:
            rows = self._full_index.search(query, top_k=top_k)
            return [{**r, "rank": i + 1} for i, r in enumerate(rows)]

        filtered = [c for c in self.chunks if matches_filters(c, active_filters)]
        if not filtered:
            return []
        q_vec = self._full_index._vector(Counter(tokenize(query)))
        scored: list[dict[str, Any]] = []
        chunk_idx_by_id = {c["id"]: i for i, c in enumerate(self.chunks)}
        for c in filtered:
            vec = self._full_index.vectors[chunk_idx_by_id[c["id"]]]
            score = sum(v * vec.get(t, 0.0) for t, v in q_vec.items())
            if score > 0:
                scored.append({**c, "score": score, "strategy": self.strategy})
        scored.sort(key=lambda r: (-r["score"], r["id"]))
        return [{**r, "rank": i + 1} for i, r in enumerate(scored[:top_k])]


class PgVectorDenseRetriever:
    """Dense retriever querying Day 6 PostgreSQL + pgvector table `rag_chunks`."""

    strategy = "dense_pgvector"

    def __init__(
        self,
        database_url: str | None = None,
        embedder: Day6QueryEmbedder | None = None,
        pg_strategy: str = "C",
        infer_filter: bool = True,
    ) -> None:
        self.database_url = resolve_database_url(database_url)
        self.embedder = embedder or Day6QueryEmbedder()
        self.pg_strategy = pg_strategy
        self.infer_filter = infer_filter
        self._query_cache: dict[str, list[float]] = {}

    def _embed(self, query: str) -> list[float]:
        if query not in self._query_cache:
            self._query_cache[query] = self.embedder.embed_query(query)
        return self._query_cache[query]

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        active_filters = filters if filters is not None else (
            infer_filters_from_query(query) if self.infer_filter else None
        )
        q_vec = self._embed(query)
        if len(q_vec) != self.embedder.dimensions:
            raise ValueError(
                f"Query vector dim {len(q_vec)} != {self.embedder.dimensions}"
            )
        q_lit = "[" + ",".join(f"{float(v):.8f}" for v in q_vec) + "]"

        where_clauses = ["strategy = %s"]
        params: list[Any] = [self.pg_strategy]
        if active_filters:
            for key, val in active_filters.items():
                if key not in _ALLOWED_FILTER_KEYS:
                    raise ValueError(f"Unsupported metadata filter key: {key}")
                if key == "chapter":
                    where_clauses.append("chapter = %s")
                    params.append(str(val))
                elif key in ("source_id", "document_id"):
                    where_clauses.append("document_id = %s")
                    params.append(str(val))
                else:
                    where_clauses.append("metadata @> %s::jsonb")
                    params.append(json.dumps({key: val}, ensure_ascii=False))

        sql = f"""
            SELECT
                chunk_id,
                document_id,
                source,
                page,
                chapter,
                text,
                metadata,
                1 - (embedding <=> %s::vector) AS score
            FROM rag_chunks
            WHERE {" AND ".join(where_clauses)}
            ORDER BY embedding <=> %s::vector ASC, chunk_id ASC
            LIMIT %s
        """
        full_params = [q_lit] + params + [q_lit, top_k]
        with psycopg.connect(self.database_url, connect_timeout=10) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, full_params)
                rows = cur.fetchall()

        results: list[dict[str, Any]] = []
        for idx, row in enumerate(rows, start=1):
            meta = row[6] if isinstance(row[6], dict) else json.loads(row[6] or "{}")
            if row[4] and "chapter" not in meta:
                meta["chapter"] = row[4]
            results.append(
                {
                    "id": row[0],
                    "source_id": row[1],
                    "source": row[2],
                    "page": int(row[3]),
                    "text": row[5],
                    "metadata": meta,
                    "score": round(float(row[7]), 6),
                    "rank": idx,
                    "strategy": self.strategy,
                }
            )
        return results


class BM25Retriever:
    """Okapi BM25 lexical retriever supporting metadata filters."""

    strategy = "bm25"

    def __init__(
        self,
        chunks: list[dict[str, Any]],
        k1: float = 1.5,
        b: float = 0.75,
        infer_filter: bool = True,
    ) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.infer_filter = infer_filter
        self.tokenized = [tokenize(c["text"]) for c in chunks]
        self.doc_lens = [len(t) for t in self.tokenized]
        self.avgdl = sum(self.doc_lens) / len(chunks) if chunks else 1.0
        self.df: Counter[str] = Counter()
        self.tfs: list[Counter[str]] = []
        for toks in self.tokenized:
            tf = Counter(toks)
            self.tfs.append(tf)
            for term in tf:
                self.df[term] += 1
        self.n_docs = len(chunks)

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        active_filters = filters if filters is not None else (
            infer_filters_from_query(query) if self.infer_filter else None
        )
        q_tokens = tokenize(query)
        scored: list[dict[str, Any]] = []
        for idx, chunk in enumerate(self.chunks):
            if not matches_filters(chunk, active_filters):
                continue
            dl = self.doc_lens[idx]
            tf_map = self.tfs[idx]
            score = 0.0
            for term in q_tokens:
                if term not in tf_map:
                    continue
                df = self.df[term]
                idf = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)
                tf = tf_map[term]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / self.avgdl))
                score += idf * (tf * (self.k1 + 1.0)) / denom
            if score > 0:
                scored.append(
                    {
                        **chunk,
                        "score": round(score, 6),
                        "strategy": self.strategy,
                    }
                )
        scored.sort(key=lambda r: (-r["score"], r["id"]))
        return [{**r, "rank": i + 1} for i, r in enumerate(scored[:top_k])]


class HybridRRFRetriever:
    """Hybrid Dense (pgvector) + Lexical (BM25) using Reciprocal Rank Fusion (k=60)."""

    strategy = "hybrid_rrf"

    def __init__(
        self,
        dense: PgVectorDenseRetriever,
        bm25: BM25Retriever,
        candidate_n: int = 20,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        bm25_weight: float = 1.0,
        infer_filter: bool = True,
    ) -> None:
        self.dense = dense
        self.bm25 = bm25
        self.candidate_n = candidate_n
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self.infer_filter = infer_filter

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        active_filters = filters if filters is not None else (
            infer_filters_from_query(query) if self.infer_filter else None
        )
        dense_hits = self.dense.search(query, filters=active_filters, top_k=self.candidate_n)
        bm25_hits = self.bm25.search(query, filters=active_filters, top_k=self.candidate_n)

        by_id: dict[str, dict[str, Any]] = {}
        rrf_scores: dict[str, float] = {}
        dense_meta: dict[str, tuple[int, float]] = {}
        bm25_meta: dict[str, tuple[int, float]] = {}

        for rank, item in enumerate(dense_hits, start=1):
            cid = item["id"]
            by_id.setdefault(cid, item)
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + self.dense_weight / (self.rrf_k + rank)
            dense_meta[cid] = (rank, float(item["score"]))

        for rank, item in enumerate(bm25_hits, start=1):
            cid = item["id"]
            by_id.setdefault(cid, item)
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + self.bm25_weight / (self.rrf_k + rank)
            bm25_meta[cid] = (rank, float(item["score"]))

        ranked_ids = sorted(rrf_scores.keys(), key=lambda cid: (-rrf_scores[cid], cid))
        results: list[dict[str, Any]] = []
        for idx, cid in enumerate(ranked_ids[:top_k], start=1):
            base = dict(by_id[cid])
            d_rank, d_score = dense_meta.get(cid, (None, 0.0))
            b_rank, b_score = bm25_meta.get(cid, (None, 0.0))
            base.update(
                {
                    "score": round(rrf_scores[cid], 6),
                    "rank": idx,
                    "strategy": self.strategy,
                    "dense_rank": d_rank,
                    "dense_score": d_score,
                    "bm25_rank": b_rank,
                    "bm25_score": b_score,
                }
            )
            results.append(base)
        return results


def _phrase_and_legal_match_score(query: str, chunk_text: str) -> float:
    """Compute exact legal-term, numeric-duration, and salient bigram match ratio in [0, 1]."""
    q_norm = _normalize_for_match(query)
    t_norm = _normalize_for_match(chunk_text)

    legal_refs = _LEGAL_REF_RE.findall(q_norm)
    q_tokens = [tok for tok in tokenize(query) if tok not in _VI_STOPWORDS]
    bigrams = [f"{q_tokens[i]} {q_tokens[i + 1]}" for i in range(len(q_tokens) - 1)]

    signals: list[float] = []
    if legal_refs:
        ref_hits = sum(1 for ref in legal_refs if ref in t_norm)
        signals.append(ref_hits / len(legal_refs))
    if bigrams:
        bg_hits = sum(1 for bg in bigrams if bg in t_norm)
        signals.append(bg_hits / len(bigrams))

    if not signals:
        return 0.0
    if legal_refs and bigrams:
        return 0.6 * signals[0] + 0.4 * signals[1]
    return signals[0]


class HybridRerankRetriever:
    """Two-stage Hybrid (Dense + BM25 + TF-IDF) with Min-Max + legal-phrase reranking."""

    strategy = "hybrid_rerank"

    def __init__(
        self,
        dense: PgVectorDenseRetriever,
        bm25: BM25Retriever,
        tfidf: TfidfRetriever,
        candidate_n: int = 20,
        w_dense: float = 0.45,
        w_bm25: float = 0.25,
        w_tfidf: float = 0.10,
        w_exact: float = 0.20,
        infer_filter: bool = True,
    ) -> None:
        self.dense = dense
        self.bm25 = bm25
        self.tfidf = tfidf
        self.candidate_n = candidate_n
        self.w_dense = w_dense
        self.w_bm25 = w_bm25
        self.w_tfidf = w_tfidf
        self.w_exact = w_exact
        self.infer_filter = infer_filter

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        active_filters = filters if filters is not None else (
            infer_filters_from_query(query) if self.infer_filter else None
        )
        dense_hits = self.dense.search(query, filters=active_filters, top_k=self.candidate_n)
        bm25_hits = self.bm25.search(query, filters=active_filters, top_k=self.candidate_n)
        tfidf_hits = self.tfidf.search(query, filters=active_filters, top_k=self.candidate_n)

        by_id: dict[str, dict[str, Any]] = {}
        raw_dense: dict[str, float] = {}
        raw_bm25: dict[str, float] = {}
        raw_tfidf: dict[str, float] = {}

        for h in dense_hits:
            by_id.setdefault(h["id"], h)
            raw_dense[h["id"]] = float(h["score"])
        for h in bm25_hits:
            by_id.setdefault(h["id"], h)
            raw_bm25[h["id"]] = float(h["score"])
        for h in tfidf_hits:
            by_id.setdefault(h["id"], h)
            raw_tfidf[h["id"]] = float(h["score"])

        if not by_id:
            return []

        all_ids = list(by_id.keys())
        norm_dense = _min_max_norm({cid: raw_dense.get(cid, 0.0) for cid in all_ids})
        norm_bm25 = _min_max_norm({cid: raw_bm25.get(cid, 0.0) for cid in all_ids})
        norm_tfidf = _min_max_norm({cid: raw_tfidf.get(cid, 0.0) for cid in all_ids})
        exact_scores = {
            cid: _phrase_and_legal_match_score(query, by_id[cid]["text"])
            for cid in all_ids
        }

        combined: dict[str, float] = {}
        for cid in all_ids:
            combined[cid] = (
                self.w_dense * norm_dense[cid]
                + self.w_bm25 * norm_bm25[cid]
                + self.w_tfidf * norm_tfidf[cid]
                + self.w_exact * exact_scores[cid]
            )

        ranked_ids = sorted(all_ids, key=lambda cid: (-combined[cid], cid))
        results: list[dict[str, Any]] = []
        for idx, cid in enumerate(ranked_ids[:top_k], start=1):
            base = dict(by_id[cid])
            base.update(
                {
                    "score": round(combined[cid], 6),
                    "rank": idx,
                    "strategy": self.strategy,
                    "dense_score": round(raw_dense.get(cid, 0.0), 6),
                    "bm25_score": round(raw_bm25.get(cid, 0.0), 6),
                    "tfidf_score": round(raw_tfidf.get(cid, 0.0), 6),
                    "exact_match_score": round(exact_scores[cid], 6),
                }
            )
            results.append(base)
        return results
