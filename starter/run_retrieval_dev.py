"""Run and compare Day 7 retrieval methods and improvements on the dev split only."""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from baseline import hit, pack_context, validate
from retrievers import (
    BM25Retriever,
    Day6QueryEmbedder,
    HybridRerankRetriever,
    HybridRRFRetriever,
    PgVectorDenseRetriever,
    TfidfRetriever,
    infer_filters_from_query,
)


def evaluate_config(
    name: str,
    retriever: Any,
    questions: list[dict[str, Any]],
    *,
    top_k: int = 3,
    budget: int = 800,
    use_question_filter: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    traces: list[dict[str, Any]] = []
    for q in questions:
        user_filter = q.get("filter") if use_question_filter else None
        inferred_filter = (
            None
            if user_filter is not None
            else (infer_filters_from_query(q["question"]) if getattr(retriever, "infer_filter", False) else None)
        )
        effective_filter = user_filter if user_filter is not None else inferred_filter

        start = time.perf_counter()
        results = retriever.search(q["question"], filters=user_filter, top_k=top_k)
        # Keep top-3 slice for comparable Hit@1 / Hit@3 reporting even if top_k > 3.
        top3 = results[:3]
        context, ids = pack_context(results, budget=budget)
        elapsed_ms = (time.perf_counter() - start) * 1000

        marked = [{**c, "hit": hit(c, q["gold"])} for c in results]
        marked_top3 = marked[:3]
        traces.append(
            {
                "config": name,
                "strategy": retriever.strategy,
                "query_id": q["id"],
                "split": q["split"],
                "question": q["question"],
                "top_k": top_k,
                "budget_chars": budget,
                "user_filter": user_filter,
                "inferred_filter": inferred_filter,
                "effective_filter": effective_filter,
                "gold": q["gold"],
                "results": marked,
                "context": context,
                "context_ids": ids,
                "context_chars": len(context),
                "elapsed_ms": round(elapsed_ms, 3),
                "hit1": bool(marked_top3 and marked_top3[0]["hit"]),
                "hit3": any(c["hit"] for c in marked_top3),
                "context_hit": any(c["hit"] and c["id"] in ids for c in marked),
            }
        )

    n = len(traces)
    hit1_n = sum(1 for t in traces if t["hit1"])
    hit3_n = sum(1 for t in traces if t["hit3"])
    ctx_n = sum(1 for t in traces if t["context_hit"])
    summary = {
        "config": name,
        "strategy": retriever.strategy,
        "split": questions[0]["split"] if questions else "unknown",
        "top_k": top_k,
        "budget_chars": budget,
        "n": n,
        "hit1": hit1_n,
        "hit1_rate": round(hit1_n / n, 4) if n else 0.0,
        "hit3": hit3_n,
        "hit3_rate": round(hit3_n / n, 4) if n else 0.0,
        "context_hit": ctx_n,
        "context_hit_rate": round(ctx_n / n, 4) if n else 0.0,
        "avg_context_chars": round(statistics.mean(t["context_chars"] for t in traces), 1),
        "median_ms": round(statistics.median(t["elapsed_ms"] for t in traces), 3),
        "total_ms": round(sum(t["elapsed_ms"] for t in traces), 3),
        "service_cost_usd": 0.0,
    }
    return summary, traces


def main() -> None:
    parser = argparse.ArgumentParser(description="Chạy thực nghiệm retrieval trên split dev")
    parser.add_argument("--chunks", type=Path, default=Path("starter/data/converted/chunks.json"))
    parser.add_argument("--questions", type=Path, default=Path("starter/data/converted/questions.json"))
    parser.add_argument("--output-root", type=Path, default=Path("starter/results"))
    args = parser.parse_args()

    chunks = json.loads(args.chunks.read_text(encoding="utf-8"))
    all_questions = json.loads(args.questions.read_text(encoding="utf-8"))
    # Evaluate strictly on dev; never inspect held-out test questions/gold here.
    dev_questions = [q for q in all_questions if q.get("split") == "dev"]
    validate(chunks, dev_questions)

    t0 = time.perf_counter()
    embedder = Day6QueryEmbedder()
    model_load_sec = round(time.perf_counter() - t0, 3)

    tfidf = TfidfRetriever(chunks, infer_filter=False)
    dense = PgVectorDenseRetriever(embedder=embedder, pg_strategy="C", infer_filter=True)
    bm25 = BM25Retriever(chunks, infer_filter=True)
    hybrid_rrf = HybridRRFRetriever(dense=dense, bm25=bm25, candidate_n=20, rrf_k=60)
    hybrid_rerank = HybridRerankRetriever(
        dense=dense, bm25=bm25, tfidf=tfidf, candidate_n=20
    )

    configs = [
        ("dev-tfidf", tfidf, 3, 800, False),
        ("dev-dense", dense, 3, 800, True),
        ("dev-bm25", bm25, 3, 800, True),
        ("dev-hybrid-rrf", hybrid_rrf, 3, 800, True),
        ("dev-improve1-budget1800", hybrid_rrf, 3, 1800, True),
        ("dev-improve2-rerank-budget2200", hybrid_rerank, 3, 2200, True),
    ]

    all_summaries: list[dict[str, Any]] = []
    dev_details: dict[str, list[dict[str, Any]]] = {}

    for cfg_name, retriever, top_k, budget, use_flt in configs:
        dense._query_cache.clear()
        summary, traces = evaluate_config(
            cfg_name,
            retriever,
            dev_questions,
            top_k=top_k,
            budget=budget,
            use_question_filter=use_flt,
        )
        summary["model_load_seconds"] = model_load_sec if retriever is not tfidf and retriever is not bm25 else 0.0
        out_dir = args.output_root / cfg_name
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "traces.json").write_text(
            json.dumps(traces, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (out_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        all_summaries.append(summary)
        dev_details[cfg_name] = traces

    (args.output_root / "dev_comparison.json").write_text(
        json.dumps(
            {"model_load_seconds": model_load_sec, "configs": all_summaries},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Model load time (google/embeddinggemma-2 256d): {model_load_sec}s")
    print("\n=== DEV RETRIEVAL COMPARISON (N=15) ===")
    header = (
        f"{'Config':<32} | {'Strategy':<14} | {'Budget':>6} | "
        f"{'Hit@1':>9} | {'Hit@3':>9} | {'CtxHit':>9} | {'Med(ms)':>8}"
    )
    print(header)
    print("-" * len(header))
    for s in all_summaries:
        n = s["n"]
        print(
            f"{s['config']:<32} | {s['strategy']:<14} | {s['budget_chars']:>6} | "
            f"{s['hit1']:>2}/{n} ({s['hit1_rate']*100:>4.1f}%) | "
            f"{s['hit3']:>2}/{n} ({s['hit3_rate']*100:>4.1f}%) | "
            f"{s['context_hit']:>2}/{n} ({s['context_hit_rate']*100:>4.1f}%) | "
            f"{s['median_ms']:>8.2f}"
        )

    print("\n=== DEV MISSES / CONTEXT DROPS BY CONFIG ===")
    for cfg_name, traces in dev_details.items():
        misses = [
            (
                t["query_id"],
                f"hit1={t['hit1']}",
                f"hit3={t['hit3']}",
                f"ctx={t['context_hit']}",
                f"top3_ids={[r['id'] for r in t['results'][:3]]}",
                f"ctx_ids={t['context_ids']}",
            )
            for t in traces
            if not (t["hit1"] and t["hit3"] and t["context_hit"])
        ]
        print(f"- {cfg_name}: {len(misses)} imperfect questions")
        for m in misses:
            print("   ", " | ".join(str(x) for x in m))


if __name__ == "__main__":
    main()
