"""Run the Day 7 Agentic RAG loop on dev or test cases and save full traces."""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from agentic import run_agent
from llm_controller import RealLLMController
from retrievers import (
    BM25Retriever,
    Day6QueryEmbedder,
    HybridRerankRetriever,
    PgVectorDenseRetriever,
    TfidfRetriever,
)


def _build_retriever(
    cfg_name: str,
    chunks: list[dict[str, Any]],
    embedder: Day6QueryEmbedder | None,
) -> tuple[Any, int, bool]:
    """Return (retriever, budget_chars, use_case_filter)."""
    if cfg_name == "baseline":
        return TfidfRetriever(chunks, infer_filter=False), 800, False
    if cfg_name == "selected":
        if embedder is None:
            embedder = Day6QueryEmbedder()
        dense = PgVectorDenseRetriever(embedder=embedder, pg_strategy="C", infer_filter=True)
        bm25 = BM25Retriever(chunks, infer_filter=True)
        tfidf = TfidfRetriever(chunks, infer_filter=False)
        retriever = HybridRerankRetriever(
            dense=dense, bm25=bm25, tfidf=tfidf, candidate_n=20, infer_filter=True
        )
        return retriever, 2200, True
    raise ValueError(f"Unknown config: {cfg_name}")


def run_cases_for_config(
    cfg_name: str,
    cases: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    controller: RealLLMController,
    *,
    embedder: Day6QueryEmbedder | None,
    top_k: int,
    max_searches: int,
    out_dir: Path,
) -> dict[str, Any]:
    retriever, budget, use_case_filter = _build_retriever(cfg_name, chunks, embedder)
    traces: list[dict[str, Any]] = []

    for idx, case in enumerate(cases, start=1):
        controller.reset_logs()
        filters = case.get("filter") if use_case_filter else None
        t0 = time.perf_counter()
        try:
            res = run_agent(
                case["question"],
                controller,
                retriever,
                filters=filters,
                max_searches=max_searches,
                top_k=top_k,
                budget=budget,
            )
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
            search_events = [e for e in res["events"] if e.get("event") == "search"]
            final_ctx = search_events[-1]["context"] if search_events else ""
            final_ctx_ids = search_events[-1]["context_ids"] if search_events else []
            llm_calls = list(controller.call_logs)
            in_tok = sum(c["prompt_tokens"] for c in llm_calls)
            out_tok = sum(c["completion_tokens"] for c in llm_calls)
            cost_usd = round(sum(c["cost_usd"] for c in llm_calls), 8)

            trace = {
                "query_id": case["id"],
                "split": case["split"],
                "config": cfg_name,
                "strategy": retriever.strategy,
                "provider": controller.provider,
                "model": controller.model,
                "question": case["question"],
                "answerable": case["answerable"],
                "status": res["status"],
                "answer": res["answer"],
                "citations": res["citations"],
                "num_searches": len(search_events),
                "final_context_ids": final_ctx_ids,
                "final_context": final_ctx,
                "events": res["events"],
                "llm_calls": llm_calls,
                "prompt_tokens": in_tok,
                "completion_tokens": out_tok,
                "cost_usd": cost_usd,
                "elapsed_ms": elapsed_ms,
                "error": None,
            }
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
            llm_calls = list(controller.call_logs)
            trace = {
                "query_id": case["id"],
                "split": case["split"],
                "config": cfg_name,
                "strategy": retriever.strategy,
                "provider": controller.provider,
                "model": controller.model,
                "question": case["question"],
                "answerable": case["answerable"],
                "status": "error",
                "answer": "",
                "citations": [],
                "num_searches": 0,
                "final_context_ids": [],
                "final_context": "",
                "events": [],
                "llm_calls": llm_calls,
                "prompt_tokens": sum(c["prompt_tokens"] for c in llm_calls),
                "completion_tokens": sum(c["completion_tokens"] for c in llm_calls),
                "cost_usd": round(sum(c["cost_usd"] for c in llm_calls), 8),
                "elapsed_ms": elapsed_ms,
                "error": f"{type(exc).__name__}: {exc}",
            }

        traces.append(trace)
        print(
            f"[{cfg_name} {idx}/{len(cases)}] {case['id']} -> status={trace['status']} "
            f"searches={trace['num_searches']} cits={trace['citations']} ({trace['elapsed_ms']}ms)"
        )

    status_counts = Counter(t["status"] for t in traces)
    search_dist = Counter(t["num_searches"] for t in traces if t["status"] != "error")
    summary = {
        "config": cfg_name,
        "strategy": retriever.strategy,
        "provider": controller.provider,
        "model": controller.model,
        "top_k": top_k,
        "budget_chars": budget,
        "max_searches": max_searches,
        "n_cases": len(traces),
        "status_counts": dict(status_counts),
        "search_calls_distribution": {str(k): v for k, v in sorted(search_dist.items())},
        "total_prompt_tokens": sum(t["prompt_tokens"] for t in traces),
        "total_completion_tokens": sum(t["completion_tokens"] for t in traces),
        "total_cost_usd": round(sum(t["cost_usd"] for t in traces), 6),
        "total_elapsed_ms": round(sum(t["elapsed_ms"] for t in traces), 2),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "traces.json").write_text(
        json.dumps(traces, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Chạy Agentic RAG thật trên dev hoặc test")
    parser.add_argument("--cases", type=Path, default=Path("starter/data/converted/cases.json"))
    parser.add_argument("--chunks", type=Path, default=Path("starter/data/converted/chunks.json"))
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--config", choices=("baseline", "selected", "both"), default="selected")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-searches", type=int, default=3)
    parser.add_argument("--provider", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--smoke-test", action="store_true", help="Chỉ chạy 1 câu đầu để kiểm tra kết nối")
    parser.add_argument("--output-root", type=Path, default=Path("starter/results"))
    args = parser.parse_args()

    chunks = json.loads(args.chunks.read_text(encoding="utf-8"))
    all_cases = json.loads(args.cases.read_text(encoding="utf-8"))
    selected_cases = [c for c in all_cases if c.get("split") == args.split]
    if args.smoke_test:
        selected_cases = selected_cases[:1]

    controller = RealLLMController(
        provider=args.provider,
        model=args.model,
        max_searches=args.max_searches,
    )
    print(f"LLM Connected: provider={controller.provider}, model={controller.model}")

    configs = ["baseline", "selected"] if args.config == "both" else [args.config]
    embedder = Day6QueryEmbedder() if "selected" in configs else None

    for cfg in configs:
        folder = (
            f"agent-smoke-{cfg}-{args.split}"
            if args.smoke_test
            else f"agent-{cfg}-{args.split}"
        )
        summary = run_cases_for_config(
            cfg,
            selected_cases,
            chunks,
            controller,
            embedder=embedder,
            top_k=args.top_k,
            max_searches=args.max_searches,
            out_dir=args.output_root / folder,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
