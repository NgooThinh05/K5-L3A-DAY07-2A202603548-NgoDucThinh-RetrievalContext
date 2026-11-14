"""Independent test-set evaluation for frozen baseline and selected Day 7 configs."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from baseline import validate
from llm_controller import RealLLMController
from retrievers import (
    BM25Retriever,
    Day6QueryEmbedder,
    HybridRerankRetriever,
    PgVectorDenseRetriever,
    TfidfRetriever,
)
from run_agent_benchmark import run_cases_for_config
from run_retrieval_dev import evaluate_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Đánh giá độc lập trên split test cho cấu hình baseline và selected đã chốt"
    )
    parser.add_argument("--chunks", type=Path, default=Path("starter/data/converted/chunks.json"))
    parser.add_argument("--questions", type=Path, default=Path("starter/data/converted/questions.json"))
    parser.add_argument("--cases", type=Path, default=Path("starter/data/converted/cases.json"))
    parser.add_argument("--output-root", type=Path, default=Path("starter/results"))
    parser.add_argument("--max-searches", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--skip-agent", action="store_true")
    args = parser.parse_args()

    chunks = json.loads(args.chunks.read_text(encoding="utf-8"))
    all_questions = json.loads(args.questions.read_text(encoding="utf-8"))
    test_questions = [q for q in all_questions if q.get("split") == "test"]
    validate(chunks, test_questions)

    t0 = time.perf_counter()
    embedder = Day6QueryEmbedder()
    model_load_sec = round(time.perf_counter() - t0, 3)

    tfidf = TfidfRetriever(chunks, infer_filter=False)
    dense = PgVectorDenseRetriever(embedder=embedder, pg_strategy="C", infer_filter=True)
    bm25 = BM25Retriever(chunks, infer_filter=True)
    hybrid_rerank = HybridRerankRetriever(
        dense=dense, bm25=bm25, tfidf=tfidf, candidate_n=20, infer_filter=True
    )

    retrieval_configs = [
        ("test-tfidf", tfidf, args.top_k, 800, False),
        ("test-selected-rerank-budget2200", hybrid_rerank, args.top_k, 2200, True),
    ]

    summaries: list[dict[str, Any]] = []
    for cfg_name, retriever, top_k, budget, use_flt in retrieval_configs:
        summary, traces = evaluate_config(
            cfg_name,
            retriever,
            test_questions,
            top_k=top_k,
            budget=budget,
            use_question_filter=use_flt,
        )
        summary["model_load_seconds"] = 0.0 if retriever is tfidf else model_load_sec
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
        summaries.append(summary)

    (args.output_root / "test_comparison.json").write_text(
        json.dumps(
            {"model_load_seconds": model_load_sec, "configs": summaries},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"=== TEST RETRIEVAL & CONTEXT (N={len(test_questions)}) ===")
    for s in summaries:
        n = s["n"]
        print(
            f"{s['config']:<32} | Hit@1={s['hit1']}/{n} ({s['hit1_rate']*100:.1f}%) | "
            f"Hit@3={s['hit3']}/{n} ({s['hit3_rate']*100:.1f}%) | "
            f"CtxHit={s['context_hit']}/{n} ({s['context_hit_rate']*100:.1f}%) | "
            f"Med={s['median_ms']}ms"
        )

    if not args.skip_agent:
        all_cases = json.loads(args.cases.read_text(encoding="utf-8"))
        test_cases = [c for c in all_cases if c.get("split") == "test"]
        controller = RealLLMController(max_searches=args.max_searches)
        print(
            f"\n=== TEST AGENT BENCHMARK (N={len(test_cases)}, "
            f"provider={controller.provider}, model={controller.model}) ==="
        )
        for cfg in ("baseline", "selected"):
            agent_summary = run_cases_for_config(
                cfg,
                test_cases,
                chunks,
                controller,
                embedder=embedder if cfg == "selected" else None,
                top_k=args.top_k,
                max_searches=args.max_searches,
                out_dir=args.output_root / f"agent-{cfg}-test",
            )
            print(json.dumps(agent_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
