from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.loaders import load_multihop_dataset  # noqa: E402
from retriever import (  # noqa: E402
    ABLATION_PRESETS,
    VERIFICATION_ABLATION_PRESETS,
    RESOLUTION_ADAPTER_SCHEMA,
    CURIOSITY_RESPONSE_SCHEMA_V32,
    RESOLUTION_REWRITE_SCHEMA,
    NormalRAG,
    VALID_RETRIEVER_MODES,
    adapt_answer_to_resolution,
    exact_match_score,
    f1_score,
    generate_curiosity_questions,
    rewrite_question_with_resolutions,
    run_iterative_resolution,
)
from scripts.retrieval_helpers import (  # noqa: E402
    DATASET_DEFAULTS,
    build_retriever,
    gold_supporting_facts,
    gold_supporting_titles,
    normalize_title,
    summarize_support_recall,
    support_metric,
)
from scripts.model_client import (  # noqa: E402
    combine_llm_metrics,
    llm_metrics_delta,
    llm_metrics_snapshot,
    make_llm_client,
    select_model_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the minimal iterative question-resolution pipeline."
    )
    parser.add_argument("--dataset", choices=tuple(DATASET_DEFAULTS), default="hotpotqa")
    parser.set_defaults(ablation="qsr_rag", verification_ablation="full")
    parser.add_argument("--data", default=None)
    parser.add_argument("--index-dir", default=None)
    parser.add_argument("--num-questions", type=int, default=50)
    parser.add_argument("--selection", choices=("random", "first"), default="random")
    parser.add_argument("--seed", type=int, default=42)
    parser.set_defaults(retriever_mode="dense_only")
    parser.add_argument(
        "--question-ids-from",
        default=None,
        help=(
            "Details JSONL whose example_id order is reused exactly. Recommended "
            "for paired Full vs Dense-only evaluation."
        ),
    )
    parser.add_argument(
        "--allow-question-id-prefix-expansion",
        action="store_true",
        help=(
            "Allow a resumed checkpoint selected from an ordered prefix of a "
            "larger --question-ids-from manifest. Every other signature field "
            "must still match exactly."
        ),
    )
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from an existing details JSONL file (default: enabled).",
    )
    parser.add_argument(
        "--retry-network-failures",
        action="store_true",
        help=(
            "When resuming, remove previously saved rows whose model calls "
            "failed because of HTTP, SSL, timeout, or connection errors, then "
            "run those examples again."
        ),
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--details-output", default=None)
    parser.set_defaults(scope_index_to_example=False)
    parser.add_argument("--supporting-doc-top-k", type=int, default=20)
    parser.add_argument("--supporting-sentence-top-k", type=int, default=20)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--generator-model-config", default="gpt4o_mini")
    parser.add_argument("--answer-model-config", default="gpt4o_mini")
    parser.add_argument("--adapter-model-config", default="gpt4o_mini")
    parser.add_argument("--rewriter-model-config", default="gpt4o_mini")
    parser.add_argument(
        "--content-filter-fallback-model-config",
        default=None,
        help=(
            "Optional model config used only when the primary provider rejects "
            "a stage call with an explicit inappropriate-content/DataInspection "
            "error. Fallback usage is recorded in runtime metrics and summaries."
        ),
    )
    parser.add_argument("--embedding-model", default="models/bge-m3")
    parser.add_argument("--reranker-model", default="models/bge-reranker-v2-m3")
    parser.add_argument("--embedding-device", default="auto")
    parser.add_argument("--reranker-device", default="auto")
    parser.set_defaults(bm25_top_k=100, dense_top_k=100)
    parser.add_argument("--dense-only-top-k", type=int, default=200)
    parser.set_defaults(sparse_top_k=100, exact_top_k=30)
    parser.add_argument("--candidate-top-k", type=int, default=200)
    parser.add_argument("--document-top-k", type=int, default=60)
    parser.add_argument("--sentence-top-k", type=int, default=20)
    parser.add_argument("--sentence-window-radius", type=int, default=1)
    parser.add_argument("--sentence-max-per-document", type=int, default=3)
    parser.add_argument("--sentence-max-per-title", type=int, default=3)
    parser.add_argument("--sentence-preselect-per-doc", type=int, default=6)
    parser.add_argument("--sentence-global-candidate-limit", type=int, default=240)
    parser.add_argument("--sentence-candidate-limit", type=int, default=0)
    parser.add_argument("--sentence-document-top-k", type=int, default=40)
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--reranker-batch-size", type=int, default=4)
    parser.add_argument("--embedding-max-length", type=int, default=512)
    parser.add_argument("--reranker-max-length", type=int, default=512)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.verification_experiment_requested = False
    if args.num_questions <= 0:
        raise SystemExit("--num-questions must be positive.")
    if args.retriever_mode == "dense_only" and args.ablation != "qsr_rag":
        raise SystemExit(
            "Retriever ablation must keep the full QSR-RAG controller: use "
            "--ablation qsr_rag with --retriever-mode dense_only."
        )
    if args.verification_experiment_requested and args.ablation != "qsr_rag":
        raise SystemExit(
            "Verification ablation must keep the full QSR-RAG state updater: "
            "use --ablation qsr_rag."
        )
    if (
        args.retriever_mode == "dense_only"
        and args.dense_only_top_k != args.candidate_top_k
    ):
        raise SystemExit(
            "Dense-only must use the same candidate budget as Hybrid: "
            "--dense-only-top-k must equal --candidate-top-k (recommended 200)."
        )
    defaults = DATASET_DEFAULTS[args.dataset]
    args.data = args.data or defaults["data"]
    args.index_dir = args.index_dir or defaults["index_dir"]
    stem = f"{args.dataset}_qsr_rag_seed{args.seed}"
    output_dir = Path("outputs/main") / args.dataset
    summary_path = Path(args.output or output_dir / f"{stem}_summary.json")
    details_path = Path(
        args.details_output or output_dir / f"{stem}_details.jsonl"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    details_path.parent.mkdir(parents=True, exist_ok=True)
    paired_question_ids = load_question_ids(
        args.question_ids_from,
        target_count=args.num_questions,
    )
    args.question_ids_digest = question_ids_digest(paired_question_ids)
    signature = build_run_signature(args)
    rows = load_resume_rows(
        details_path,
        expected_signature=signature,
        resume=args.resume,
        compatible_question_ids=(
            paired_question_ids
            if args.allow_question_id_prefix_expansion
            else None
        ),
    )
    checkpoint_changed = False
    trimmed_checkpoint_rows = 0
    if args.resume and len(rows) > args.num_questions:
        trimmed_checkpoint_rows = len(rows) - args.num_questions
        rows = rows[: args.num_questions]
        checkpoint_changed = True
        print(
            "Trimmed "
            f"{trimmed_checkpoint_rows} checkpoint row(s) beyond the requested "
            f"target of {args.num_questions}.",
            flush=True,
        )
    retried_network_failures = 0
    if args.resume and args.retry_network_failures:
        retained_rows = [
            row for row in rows if not is_retryable_network_failure(row)
        ]
        retried_network_failures = len(rows) - len(retained_rows)
        if retried_network_failures:
            rows = retained_rows
            checkpoint_changed = True
            print(
                "Removed "
                f"{retried_network_failures} saved network-failure row(s); "
                "those examples will be run again.",
                flush=True,
            )
    if checkpoint_changed:
        for checkpoint_index, row in enumerate(rows, start=1):
            row["index"] = checkpoint_index
        write_jsonl_atomic(details_path, rows)
    all_examples = load_multihop_dataset(args.data, dataset=args.dataset)
    examples = select_examples_for_run(
        all_examples,
        target_count=args.num_questions,
        selection=args.selection,
        seed=args.seed,
        completed_ids=[str(row.get("example_id") or "") for row in rows],
        required_ids=paired_question_ids,
    )
    dataset_index_by_id = {
        str(example.example_id): index
        for index, example in enumerate(all_examples)
    }
    if rows and refresh_retrieval_recall(
        rows,
        examples,
        document_top_k=args.supporting_doc_top_k,
        sentence_top_k=args.supporting_sentence_top_k,
    ):
        write_jsonl_atomic(details_path, rows)
        print(
            "Updated checkpoint retrieval recall using the union of each "
            "Normal RAG call's Top-K results.",
            flush=True,
        )
    completed_ids = {str(row.get("example_id") or "") for row in rows}
    pending = [
        (index, example)
        for index, example in enumerate(examples, start=1)
        if str(example.example_id) not in completed_ids
    ]
    resumed_count = len(rows)
    if resumed_count:
        print(
            f"Resuming {args.dataset}: completed={resumed_count} "
            f"remaining={len(pending)} details={details_path.resolve()}",
            flush=True,
        )
    if not pending:
        summary = summarize(rows, args)
        summary.update(
            {
                "resume_enabled": args.resume,
                "resumed_questions": resumed_count,
                "target_questions": args.num_questions,
                "completed_questions": len(rows),
                "remaining_questions": 0,
                "retried_network_failures": retried_network_failures,
                "trimmed_checkpoint_rows": trimmed_checkpoint_rows,
                "details_output": str(details_path.resolve()),
            }
        )
        write_json_atomic(summary_path, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print("All selected questions were already completed.")
        return 0

    retriever = build_retriever(args)
    generator_client = _model_client(args, args.generator_model_config, "question_generator", CURIOSITY_RESPONSE_SCHEMA_V32)
    answer_client = _model_client(args, args.answer_model_config, "normal_rag_reader", None)
    adapter_client = _model_client(args, args.adapter_model_config, "resolution_adapter", RESOLUTION_ADAPTER_SCHEMA)
    updater_client = _model_client(
        args,
        args.rewriter_model_config,
        "resolution_rewriter",
        RESOLUTION_REWRITE_SCHEMA,
    )
    clients = [generator_client, answer_client, adapter_client, updater_client]
    normal_rag = NormalRAG(
        retriever=retriever,
        llm_client=answer_client,
        top_k_sentences=args.sentence_top_k,
        final_top_k_sentences=args.sentence_top_k,
    )

    file_mode = "a" if args.resume and details_path.exists() else "w"
    processed_this_run = 0
    with details_path.open(file_mode, encoding="utf-8") as details_file:
        for index, example in pending:
            if args.scope_index_to_example:
                retriever.set_example_scope(example.example_id)
            else:
                retriever.set_example_scope(None)
            before = [llm_metrics_snapshot(client) for client in clients]
            started = time.perf_counter()
            result = run_iterative_resolution(
                original_question=example.question,
                dataset=args.dataset,
                question_generator=lambda **kwargs: generate_curiosity_questions(
                    **kwargs, llm_client=generator_client
                ),
                normal_rag=normal_rag,
                answer_fact_adapter=lambda **kwargs: adapt_answer_to_resolution(
                    **kwargs, llm_client=adapter_client
                ),
                resolution_rewriter=lambda **kwargs: rewrite_question_with_resolutions(
                    **kwargs, llm_client=updater_client
                ),
                ablation_config=ABLATION_PRESETS[args.ablation],
                verification_config=VERIFICATION_ABLATION_PRESETS[
                    args.verification_ablation
                ],
            )
            predicted = str(result.get("answer") or "")
            llm_by_stage = {
                label: llm_metrics_delta(before_item, llm_metrics_snapshot(client))
                for label, before_item, client in zip(
                    ("generator", "normal_rag", "adapter", "updater"),
                    before,
                    clients,
                )
            }
            llm_total = combine_llm_metrics(list(llm_by_stage.values()))
            state_metrics = result.get("state") or {}
            row = {
                "index": index,
                "example_id": example.example_id,
                "dataset_index": dataset_index_by_id[str(example.example_id)],
                "dataset": args.dataset,
                "original_question": example.question,
                "gold_answer": example.answer,
                "run_signature": signature,
                **result,
                "eval": {
                    "em": exact_match_score(predicted, example.answer),
                    "f1": f1_score(predicted, example.answer),
                },
                "runtime_metrics": {
                    "elapsed_seconds": time.perf_counter() - started,
                    "retrieval_calls": int(state_metrics.get("retrieval_calls", 0)),
                    "reader_calls": int(state_metrics.get("reader_calls", 0)),
                    "llm_calls": int(llm_total.get("llm_calls", 0)),
                    "llm": llm_by_stage,
                    "llm_total": llm_total,
                },
            }
            row["retrieval_recall"] = resolution_supporting_recall(
                example,
                result,
                document_top_k=args.supporting_doc_top_k,
                sentence_top_k=args.supporting_sentence_top_k,
            )
            row["candidate_recall"] = candidate_supporting_recall(
                example,
                result,
            )
            rows.append(row)
            details_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            details_file.flush()
            os.fsync(details_file.fileno())
            processed_this_run += 1
            checkpoint = summarize(rows, args)
            checkpoint.update(
                {
                    "resume_enabled": args.resume,
                    "resumed_questions": resumed_count,
                    "target_questions": args.num_questions,
                    "processed_this_run": processed_this_run,
                    "completed_questions": len(rows),
                    "remaining_questions": len(examples) - len(rows),
                    "retried_network_failures": retried_network_failures,
                    "trimmed_checkpoint_rows": trimmed_checkpoint_rows,
                    "details_output": str(details_path.resolve()),
                }
            )
            write_json_atomic(summary_path, checkpoint)
            if args.progress_every > 0 and processed_this_run % args.progress_every == 0:
                print(
                    f"Completed {len(rows)}/{len(examples)} em={row['eval']['em']} "
                    f"f1={row['eval']['f1']:.4f} "
                    f"doc_recall={_format_metric(row['retrieval_recall']['supporting_documents']['recall'])} "
                    f"sentence_recall={_format_metric(row['retrieval_recall']['supporting_sentences']['recall'])} "
                    f"retrievals={row['runtime_metrics']['retrieval_calls']} "
                    f"llm_calls={row['runtime_metrics']['llm_calls']} "
                    f"stop={row['stop_reason']}",
                    flush=True,
                )

    summary = summarize(rows, args)
    summary.update(
        {
            "resume_enabled": args.resume,
            "resumed_questions": resumed_count,
            "target_questions": args.num_questions,
            "processed_this_run": processed_this_run,
            "completed_questions": len(rows),
            "remaining_questions": len(examples) - len(rows),
            "retried_network_failures": retried_network_failures,
            "trimmed_checkpoint_rows": trimmed_checkpoint_rows,
            "details_output": str(details_path.resolve()),
        }
    )
    write_json_atomic(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Details written to {details_path.resolve()}")
    print(f"Summary written to {summary_path.resolve()}")
    return 0


def summarize(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    count = len(rows)
    stop_reasons = Counter(str(row.get("stop_reason") or "UNKNOWN") for row in rows)
    states = [row.get("state") or {} for row in rows]
    rounds = [
        round_item
        for state in states
        for round_item in (state.get("rounds") or [])
    ]
    query_specs = [
        item
        for round_item in rounds
        for item in (round_item.get("generated_query_specs") or [])
        if isinstance(item, dict)
    ]
    generator_results = [
        round_item.get("generator_result") or {}
        for round_item in rounds
        if round_item.get("generator_result") is not None
    ]
    local_results = [
        item.get("rag_result") or {}
        for state in states
        for item in (state.get("local_rag_history") or [])
    ]
    adapted = [
        item
        for round_item in rounds
        for item in (round_item.get("adapted_facts") or [])
    ]
    adapter_reject_reasons = Counter(
        str(item.get("reject_reason") or "")
        for item in adapted
        if item.get("usable") is False
    )
    adapter_cardinalities = Counter(
        str(item.get("cardinality") or "")
        for item in adapted
        if item.get("usable") is True
    )
    rewrite_results = [
        round_item.get("rewrite_result") or {}
        for round_item in rounds
        if round_item.get("rewrite_result")
    ]
    routing_tags = Counter(
        str(tag)
        for item in rewrite_results
        for tag in (item.get("routing_tags") or [])
    )
    policy_violations = Counter(
        str(violation)
        for item in rewrite_results
        for violation in (item.get("violations") or [])
    )
    rewrite_operations = Counter(
        str(item.get("operation") or "UNKNOWN") for item in rewrite_results
    )
    rewrite_reasons = Counter(
        str(item.get("reason") or "UNKNOWN") for item in rewrite_results
    )
    verified_resolutions = [
        item
        for state in states
        for item in (state.get("verified_resolutions") or state.get("verified_facts") or [])
        if isinstance(item, dict)
    ]
    verification_name = getattr(args, "verification_ablation", "full")
    verification = VERIFICATION_ABLATION_PRESETS[verification_name]
    accepted_candidates = [item for item in adapted if item.get("usable") is True]
    rejected_candidates = [item for item in adapted if item.get("usable") is False]
    adapter_candidates = [
        item
        for item in adapted
        if item.get("adapter_used") is True
        or (
            verification_name == "full"
            and "adapter_used" not in item
            and item.get("adapter_decision") != "NOT_USED"
        )
    ]
    parsing_only_candidates = [
        item for item in adapted if item.get("adapter_used") is False
    ]
    adapter_rejections = [
        item for item in adapter_candidates if item.get("usable") is False
    ]
    parsing_only_rejections = [
        item for item in parsing_only_candidates if item.get("usable") is False
    ]
    verification_policy_violations = _verification_policy_violations(
        adapted, verification_name
    )
    rewrite_operation_types = Counter(
        str(item.get("rewrite_operation_type") or "UNKNOWN")
        for item in verified_resolutions
    )
    failures = [
        item
        for state in states
        for item in (state.get("failures") or [])
        if isinstance(item, dict)
    ]
    failure_stages = Counter(str(item.get("failure_stage") or "UNKNOWN") for item in failures)
    failure_types = Counter(str(item.get("failure_type") or "UNKNOWN") for item in failures)
    unknown_answers = sum(
        _answer_is_unknown(str(item.get("answer") or "")) for item in local_results
    )
    document_metrics = [
        (row.get("retrieval_recall") or {}).get("supporting_documents") or {}
        for row in rows
    ]
    sentence_metrics = [
        (row.get("retrieval_recall") or {}).get("supporting_sentences") or {}
        for row in rows
    ]
    candidate_metrics = [
        (row.get("candidate_recall") or {}).get("supporting_documents") or {}
        for row in rows
    ]
    reader_results = [
        result for row in rows for result in _reader_results(row)
    ]
    evidence_token_counts = [
        int(result.get("evidence_token_count") or 0)
        for result in reader_results
    ]
    retrieval_diagnostics = [
        diagnostic
        for row in rows
        for diagnostic in _retrieval_call_diagnostics(row)
    ]
    channel_calls = Counter()
    for diagnostic in retrieval_diagnostics:
        channel_calls.update(
            {
                str(name): int(value or 0)
                for name, value in (
                    diagnostic.get("channel_call_counts") or {}
                ).items()
            }
        )
    retriever_policy_violations = _retriever_policy_violations(
        retrieval_diagnostics,
        mode=getattr(args, "retriever_mode", "hybrid"),
        candidate_top_k=getattr(args, "candidate_top_k", 200),
    )
    row_llm_metrics = [_row_llm_total(row) for row in rows]
    combined_llm = combine_llm_metrics(row_llm_metrics)
    llm_usage_by_stage = _summarize_llm_by_stage(rows, count)
    successful_llm_calls = int(combined_llm.get("successful_llm_calls", 0))
    token_reported_calls = int(combined_llm.get("token_usage_reported_calls", 0))
    normal_completions = sum(
        bool(row.get("completed_normally", row.get("stop_reason") == "NO_NEW_GAP"))
        for row in rows
    )
    ablation = ABLATION_PRESETS[getattr(args, "ablation", "qsr_rag")]
    return {
        "experiment": "qsr_rag_main",
        "ablation": ablation.to_dict(),
        "verification_ablation": verification.to_dict(),
        "retrieval": {
            "retriever_mode": getattr(args, "retriever_mode", "hybrid"),
            "dense_top_k": (
                getattr(args, "dense_only_top_k", 200)
                if getattr(args, "retriever_mode", "hybrid") == "dense_only"
                else getattr(args, "dense_top_k", 100)
            ),
            "candidate_top_k": getattr(args, "candidate_top_k", 200),
            "embedding_model": getattr(args, "embedding_model", "models/bge-m3"),
            "document_reranker": True,
            "sentence_reranker": True,
            "reranker_model": getattr(
                args, "reranker_model", "models/bge-reranker-v2-m3"
            ),
            "document_top_k": getattr(args, "document_top_k", 60),
            "sentence_top_k": getattr(args, "sentence_top_k", 20),
            "sentence_window_radius": getattr(args, "sentence_window_radius", 1),
            "reranker_batch_size": getattr(args, "reranker_batch_size", 4),
            "channel_call_counts": dict(channel_calls),
            "fusion_calls": sum(
                int(item.get("fusion_calls") or 0)
                for item in retrieval_diagnostics
            ),
            "maximum_candidate_count": max(
                (
                    int(item.get("candidate_count") or 0)
                    for item in retrieval_diagnostics
                ),
                default=0,
            ),
            "document_reranker_calls": sum(
                int(item.get("document_reranker_calls") or 0)
                for item in retrieval_diagnostics
            ),
            "sentence_reranker_calls": sum(
                int(item.get("sentence_reranker_calls") or 0)
                for item in retrieval_diagnostics
            ),
            "policy_compliant": not retriever_policy_violations,
            "policy_violations": retriever_policy_violations,
            "question_ids_digest": getattr(args, "question_ids_digest", None),
            "question_ids_source": getattr(args, "question_ids_from", None),
        },
        "dataset": args.dataset,
        "num_questions": count,
        "selection": args.selection,
        "seed": args.seed,
        "em": _average((row.get("eval") or {}).get("em") for row in rows),
        "f1": _average((row.get("eval") or {}).get("f1") for row in rows),
        "average_resolution_rounds": _average(
            len((row.get("state") or {}).get("rewrite_history") or []) for row in rows
        ),
        "average_local_resolution_calls": _average(
            len((row.get("state") or {}).get("local_rag_history") or [])
            for row in rows
        ),
        "average_retrieval_calls_per_question": _average(
            (row.get("state") or {}).get("retrieval_calls", 0) for row in rows
        ),
        "average_reader_calls_per_question": _average(
            (row.get("state") or {}).get("reader_calls", 0) for row in rows
        ),
        "average_evidence_tokens_per_reader_call": (
            sum(evidence_token_counts) / len(evidence_token_counts)
            if evidence_token_counts
            else None
        ),
        "average_evidence_tokens_per_question": (
            sum(evidence_token_counts) / count if count else None
        ),
        "evidence_token_count_method": "lexical_token_estimate_v1",
        "average_first_no_gap_round": _average(
            (row.get("state") or {}).get("first_no_gap_round") for row in rows
        ),
        "average_post_no_gap_retrievals": _average(
            (row.get("state") or {}).get("post_no_gap_retrievals", 0)
            for row in rows
        ),
        "average_llm_calls_per_question": _average(
            item.get("llm_calls", 0) for item in row_llm_metrics
        ),
        "llm_usage": {
            **combined_llm,
            "average_prompt_tokens_per_question_reported": _divide_or_none(
                combined_llm.get("prompt_tokens"), count
            ),
            "average_completion_tokens_per_question_reported": _divide_or_none(
                combined_llm.get("completion_tokens"), count
            ),
            "average_total_tokens_per_question_reported": _divide_or_none(
                combined_llm.get("total_tokens"), count
            ),
            "token_usage_call_coverage": (
                token_reported_calls / successful_llm_calls
                if successful_llm_calls else None
            ),
        },
        "content_filter_fallback": {
            "model_config": getattr(
                args, "content_filter_fallback_model_config", None
            ),
            "calls": int(combined_llm.get("content_filter_fallback_calls", 0)),
            "successes": int(
                combined_llm.get("content_filter_fallback_successes", 0)
            ),
            "failures": int(
                combined_llm.get("content_filter_fallback_failures", 0)
            ),
        },
        "llm_usage_by_stage": llm_usage_by_stage,
        "normal_completion_rate": normal_completions / count if count else None,
        "abnormal_stop_rate": (
            (count - normal_completions) / count if count else None
        ),
        "stop_reason_distribution": dict(stop_reasons),
        "failure_stage_distribution": dict(failure_stages),
        "failure_type_distribution": dict(failure_types),
        "query_spec_diagnostics": {
            "json_success_rate": _boolean_rate(
                item.get("generator_schema_valid") for item in generator_results
            ),
            "semantic_valid_rate": _boolean_rate(
                item.get("query_spec_semantic_valid")
                for item in generator_results
            ),
            "average_generated_queries": (
                len(query_specs) / len(generator_results)
                if generator_results else None
            ),
            "parallel_query_rate": (
                sum(
                    len(round_item.get("generated_query_specs") or []) == 2
                    for round_item in rounds
                    if round_item.get("generator_result") is not None
                ) / len(generator_results)
                if generator_results else None
            ),
        },
        "answer_exact_preservation_rate": _boolean_rate(
            _normalized_answer_value(item.get("rag_answer"))
            == _normalized_answer_value(item.get("answer_exact"))
            for item in adapted
            if item.get("usable")
        ),
        "resolution_utilization_rate": (
            sum(len(item.get("used_query_ids") or []) for item in rewrite_results)
            / len(verified_resolutions)
            if verified_resolutions else None
        ),
        "rewriter_routing_tag_distribution": dict(routing_tags),
        "rewrite_operation_type_distribution": dict(rewrite_operation_types),
        "accepted_rewrite_operation_distribution": dict(
            Counter(
                str(item.get("operation") or "UNKNOWN")
                for item in rewrite_results
                if item.get("accepted", item.get("update_success"))
            )
        ),
        "rewrite_operation_distribution": dict(rewrite_operations),
        "rewrite_reason_distribution": dict(rewrite_reasons),
        "proposed_answer_space_change_rate": (
            sum(bool(item.get("answer_space_changed")) for item in rewrite_results)
            / len(rewrite_results)
            if rewrite_results
            else None
        ),
        "rewrite_policy_violation_distribution": dict(policy_violations),
        "rewrite_policy_violation_rate": (
            sum(bool(item.get("violations")) for item in rewrite_results)
            / len(rewrite_results)
            if rewrite_results
            else None
        ),
        "deterministic_rewriter_result_count": sum(
            bool(item.get("deterministic_reason")) for item in rewrite_results
        ),
        "keep_rate": (
            sum(not bool(item.get("update_success")) for item in rewrite_results)
            / len(rewrite_results)
            if rewrite_results else None
        ),
        "rewriter_keep_due_to_validation_rate": _boolean_rate(
            bool(item.get("validation_errors"))
            or bool((item.get("raw_result") or {}).get("attempt_validation_errors"))
            for item in rewrite_results
        ),
        "question_version_average": _average(
            state.get("question_version", 0) for state in states
        ),
        "local_answer_unknown_rate": (
            unknown_answers / len(local_results) if local_results else None
        ),
        "adapter_usable_rate": (
            sum(bool(item.get("usable")) for item in adapted) / len(adapted)
            if adapted else None
        ),
        "adapter_reject_reason_distribution": dict(adapter_reject_reasons),
        "adapter_answer_cardinality_distribution": dict(adapter_cardinalities),
        "adapter_not_grounded_rate": (
            adapter_reject_reasons["NOT_GROUNDED"] / len(adapted)
            if adapted else None
        ),
        "adapter_parse_error_rate": (
            sum(bool(item.get("parse_error")) for item in adapted) / len(adapted)
            if adapted else None
        ),
        "rewrite_success_rate": (
            sum(
                bool(item.get("update_success", item.get("rewrite_success")))
                for item in rewrite_results
            )
            / len(rewrite_results)
            if rewrite_results else None
        ),
        "question_changed_rate": (
            sum(bool(item.get("update_success")) for item in rewrite_results)
            / len(rewrite_results)
            if rewrite_results else None
        ),
        "average_standard_facts": _average(
            len(state.get("standard_facts") or []) for state in states
        ),
        "resolution_acceptance": {
            "candidate_resolution_count": len(adapted),
            "accepted_resolution_count": len(accepted_candidates),
            "rejected_resolution_count": len(rejected_candidates),
            "average_candidates_per_question": (
                len(adapted) / count if count else None
            ),
            "average_accepted_resolutions_per_question": (
                len(accepted_candidates) / count if count else None
            ),
            "overall_rejection_rate": (
                len(rejected_candidates) / len(adapted) if adapted else None
            ),
            "adapter_candidate_count": len(adapter_candidates),
            "adapter_llm_call_count": sum(
                bool(item.get("llm_called")) for item in adapter_candidates
            ),
            "normalizer_candidate_count": sum(
                item.get("normalizer_used") is True for item in adapted
            ),
            "normalizer_llm_call_count": sum(
                item.get("normalizer_used") is True and bool(item.get("llm_called"))
                for item in adapted
            ),
            "verification_rejection_rate": (
                len(adapter_rejections) / len(adapter_candidates)
                if adapter_candidates else None
            ),
            "parsing_only_candidate_count": len(parsing_only_candidates),
            "parsing_only_rejection_rate": (
                len(parsing_only_rejections) / len(parsing_only_candidates)
                if parsing_only_candidates else None
            ),
            "adapter_decision_distribution": dict(
                Counter(
                    str(item.get("adapter_decision") or "UNKNOWN")
                    for item in adapted
                )
            ),
            "rejection_reason_distribution": dict(adapter_reject_reasons),
            "verification_status_distribution": dict(
                Counter(
                    str(item.get("verification_status") or "UNKNOWN")
                    if item.get("verification_status")
                    else (
                        "VERIFIED_EVIDENCE"
                        if verification_name == "full"
                        else "UNKNOWN"
                    )
                    for item in verified_resolutions
                )
            ),
            "policy_compliant": not verification_policy_violations,
            "policy_violations": verification_policy_violations,
        },
        "retrieval_recall": {
            "candidate_documents_at_200": summarize_support_recall(
                candidate_metrics
            ),
            "supporting_documents": summarize_support_recall(document_metrics),
            "supporting_sentences": summarize_support_recall(sentence_metrics),
        },
    }


def _boolean_rate(values) -> float | None:
    items = [bool(value) for value in values if isinstance(value, bool)]
    return sum(items) / len(items) if items else None


def _normalized_answer_value(value: Any) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(normalized.split())


def resolution_supporting_recall(
    example: Any,
    result: dict[str, Any],
    *,
    document_top_k: int,
    sentence_top_k: int,
) -> dict[str, Any]:
    """Measure recall from the union of each retrieval call's own Top-K results.

    Applying Top-K after concatenating calls makes the first call consume the
    entire cutoff and incorrectly hides evidence found in later rounds.  The
    iterative system can use every call, so the evaluation takes Top-K within
    each call and then unions the retrieved support items.
    """

    retrieval_calls = _retrieval_call_documents(result)
    gold_titles = gold_supporting_titles(example)
    retrieved_titles: set[str] = set()
    document_limit = max(0, document_top_k)
    for documents in retrieval_calls:
        retrieved_titles.update(
            normalize_title(str(item.get("title") or ""))
            for item in documents[:document_limit]
        )
    retrieved_titles.discard("")
    hit_titles = gold_titles.intersection(retrieved_titles)

    gold_facts = gold_supporting_facts(example)
    retrieved_facts: set[tuple[str, int]] = set()
    sentence_limit = max(0, sentence_top_k)
    for documents in retrieval_calls:
        for item in documents[:sentence_limit]:
            title = normalize_title(str(item.get("title") or ""))
            indices = list(item.get("covered_sentence_indices") or [])
            if not indices and item.get("sentence_index") is not None:
                indices = [item.get("sentence_index")]
            for value in indices:
                sentence_index = _optional_int(value)
                if title and sentence_index is not None:
                    retrieved_facts.add((title, sentence_index))
    hit_facts = gold_facts.intersection(retrieved_facts)
    return {
        "aggregation": "union_of_per_call_top_k",
        "retrieval_call_count": len(retrieval_calls),
        "document_top_k": max(0, document_top_k),
        "sentence_top_k": max(0, sentence_top_k),
        "supporting_documents": support_metric(
            gold_count=len(gold_titles),
            hit_count=len(hit_titles),
            gold_items=sorted(gold_titles),
            hit_items=sorted(hit_titles),
        ),
        "supporting_sentences": support_metric(
            gold_count=len(gold_facts),
            hit_count=len(hit_facts),
            gold_items=[
                {"title": title, "sentence_index": sentence_index}
                for title, sentence_index in sorted(gold_facts)
            ],
            hit_items=[
                {"title": title, "sentence_index": sentence_index}
                for title, sentence_index in sorted(hit_facts)
            ],
        ),
    }


def _retrieval_call_documents(result: dict[str, Any]) -> list[list[dict[str, Any]]]:
    """Return ranked document lists for local calls followed by the final call."""

    calls: list[list[dict[str, Any]]] = []
    state = result.get("state") or {}
    for history_item in state.get("local_rag_history") or []:
        rag_result = history_item.get("rag_result") or {}
        documents = rag_result.get("documents") or []
        if isinstance(documents, list):
            calls.append([dict(item) for item in documents if isinstance(item, dict)])

    final_result = result.get("final_rag_result") or {}
    final_documents = final_result.get("documents") or []
    if final_result and isinstance(final_documents, list):
        calls.append(
            [dict(item) for item in final_documents if isinstance(item, dict)]
        )

    # Compatibility fallback for synthetic tests and checkpoints created before
    # per-call RAG results were retained.
    if not calls:
        accumulated = state.get("accumulated_documents") or []
        if isinstance(accumulated, list):
            calls.append(
                [dict(item) for item in accumulated if isinstance(item, dict)]
            )
    return calls


def refresh_retrieval_recall(
    rows: list[dict[str, Any]],
    examples: list[Any],
    *,
    document_top_k: int,
    sentence_top_k: int,
) -> bool:
    """Recompute checkpoint metrics without rerunning retrieval or any LLM."""

    example_by_id = {str(item.example_id): item for item in examples}
    changed = False
    for row in rows:
        example = example_by_id.get(str(row.get("example_id") or ""))
        if example is None:
            continue
        updated = resolution_supporting_recall(
            example,
            row,
            document_top_k=document_top_k,
            sentence_top_k=sentence_top_k,
        )
        if row.get("retrieval_recall") != updated:
            row["retrieval_recall"] = updated
            changed = True
        updated_candidates = candidate_supporting_recall(example, row)
        if row.get("candidate_recall") != updated_candidates:
            row["candidate_recall"] = updated_candidates
            changed = True
    return changed


def build_run_signature(args: argparse.Namespace) -> dict[str, Any]:
    ablation = ABLATION_PRESETS[getattr(args, "ablation", "qsr_rag")]
    signature = {
        "version": "qsr_rag_main_v1",
        "contract_revision": "generator_gap_contract_v1",
        "ablation": ablation.to_dict(),
        "dataset": args.dataset,
        "data": str(Path(args.data).resolve()),
        "index_dir": str(Path(args.index_dir).resolve()),
        "selection": args.selection,
        "seed": args.seed,
        "question_ids_digest": getattr(args, "question_ids_digest", None),
        "scope_index_to_example": bool(args.scope_index_to_example),
        "generator_model_config": args.generator_model_config,
        "answer_model_config": args.answer_model_config,
        "adapter_model_config": args.adapter_model_config,
        "rewriter_model_config": args.rewriter_model_config,
        "retrieval": {
            "retriever_mode": getattr(args, "retriever_mode", "hybrid"),
            "dense_only_top_k": getattr(args, "dense_only_top_k", 200),
            "embedding_model": args.embedding_model,
            "reranker_model": args.reranker_model,
            "embedding_batch_size": args.embedding_batch_size,
            "reranker_batch_size": args.reranker_batch_size,
            "embedding_max_length": args.embedding_max_length,
            "reranker_max_length": args.reranker_max_length,
            **{
                name: getattr(args, name)
                for name in (
                    "bm25_top_k",
                    "dense_top_k",
                    "sparse_top_k",
                    "exact_top_k",
                    "candidate_top_k",
                    "document_top_k",
                    "sentence_top_k",
                    "sentence_window_radius",
                    "sentence_max_per_document",
                    "sentence_max_per_title",
                    "sentence_preselect_per_doc",
                    "sentence_global_candidate_limit",
                    "sentence_candidate_limit",
                    "sentence_document_top_k",
                )
            },
        },
        "supporting_doc_top_k": args.supporting_doc_top_k,
        "supporting_sentence_top_k": args.supporting_sentence_top_k,
    }
    verification_name = getattr(args, "verification_ablation", "full")
    if getattr(args, "verification_experiment_requested", False):
        signature["version"] = "controller_v7_verification_ablation_clean_v2"
        signature["verification_ablation"] = VERIFICATION_ABLATION_PRESETS[
            verification_name
        ].to_dict()
    return signature


def candidate_supporting_recall(
    example: Any,
    result: dict[str, Any],
) -> dict[str, Any]:
    """Measure gold-document recall in each call's pre-reranking candidates."""

    gold_titles = gold_supporting_titles(example)
    retrieved_titles: set[str] = set()
    candidate_counts: list[int] = []
    for diagnostic in _retrieval_call_diagnostics(result):
        candidates = diagnostic.get("candidate_documents") or []
        candidate_counts.append(len(candidates))
        retrieved_titles.update(
            normalize_title(str(item.get("title") or ""))
            for item in candidates
            if isinstance(item, dict)
        )
    retrieved_titles.discard("")
    hit_titles = gold_titles.intersection(retrieved_titles)
    return {
        "stage": "pre_document_reranking",
        "candidate_top_k": 200,
        "retrieval_call_count": len(candidate_counts),
        "candidate_counts": candidate_counts,
        "supporting_documents": support_metric(
            gold_count=len(gold_titles),
            hit_count=len(hit_titles),
            gold_items=sorted(gold_titles),
            hit_items=sorted(hit_titles),
        ),
    }


def _reader_results(result: dict[str, Any]) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    state = result.get("state") or {}
    for history_item in state.get("local_rag_history") or []:
        rag_result = history_item.get("rag_result") or {}
        if isinstance(rag_result, dict):
            values.append(rag_result)
    final_result = result.get("final_rag_result") or {}
    if isinstance(final_result, dict) and final_result:
        values.append(final_result)
    return values


def _retrieval_call_diagnostics(
    result: dict[str, Any],
) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    state = result.get("state") or {}
    for history_item in state.get("local_rag_history") or []:
        diagnostic = (
            (history_item.get("rag_result") or {}).get("retrieval_diagnostics")
            or {}
        )
        if isinstance(diagnostic, dict) and diagnostic:
            values.append(diagnostic)
    final_diagnostics = (
        (result.get("final_rag_result") or {}).get("retrieval_diagnostics")
        or {}
    )
    if not isinstance(final_diagnostics, dict):
        return values
    if "retriever_mode" in final_diagnostics:
        values.append(final_diagnostics)
        return values
    for name in ("target_question", "current_reasoning_question"):
        diagnostic = final_diagnostics.get(name) or {}
        if isinstance(diagnostic, dict) and diagnostic:
            values.append(diagnostic)
    return values


def _retriever_policy_violations(
    diagnostics: list[dict[str, Any]],
    *,
    mode: str,
    candidate_top_k: int,
) -> list[str]:
    if mode != "dense_only":
        return []
    violations: list[str] = []
    for index, item in enumerate(diagnostics, start=1):
        calls = item.get("channel_call_counts") or {}
        for channel in ("bm25", "sparse", "exact"):
            if int(calls.get(channel) or 0) != 0:
                violations.append(f"CALL_{index}_{channel.upper()}_NOT_ZERO")
        if int(item.get("fusion_calls") or 0) != 0:
            violations.append(f"CALL_{index}_FUSION_NOT_ZERO")
        if int(calls.get("dense") or 0) <= 0:
            violations.append(f"CALL_{index}_DENSE_NOT_USED")
        if int(item.get("candidate_count") or 0) > candidate_top_k:
            violations.append(f"CALL_{index}_CANDIDATE_BUDGET_EXCEEDED")
        if item.get("enable_document_rerank") is not True:
            violations.append(f"CALL_{index}_DOCUMENT_RERANK_DISABLED")
        if item.get("enable_sentence_rerank") is not True:
            violations.append(f"CALL_{index}_SENTENCE_RERANK_DISABLED")
    return violations


def _verification_policy_violations(
    candidates: list[dict[str, Any]],
    policy: str,
) -> list[str]:
    violations: list[str] = []
    for index, item in enumerate(candidates, start=1):
        # Legacy rows created before the verification ablation do not carry
        # these diagnostics and remain readable for Full-result reuse.
        if "adapter_used" not in item:
            continue
        prefix = f"CANDIDATE_{index}"
        if policy == "full":
            if item.get("adapter_used") is not True:
                violations.append(f"{prefix}_ADAPTER_NOT_USED")
            if item.get("evidence_visible") is not True:
                violations.append(f"{prefix}_EVIDENCE_NOT_VISIBLE")
        elif policy == "evidence_blind":
            if item.get("adapter_used") is not True:
                violations.append(f"{prefix}_ADAPTER_NOT_USED")
            if item.get("evidence_visible") is not False:
                violations.append(f"{prefix}_EVIDENCE_VISIBLE")
            adapter_input = item.get("adapter_input") or {}
            if "retrieved_evidence" in adapter_input:
                violations.append(f"{prefix}_EVIDENCE_IN_ADAPTER_INPUT")
        elif policy == "no_verification":
            if item.get("adapter_used") is not False:
                violations.append(f"{prefix}_ADAPTER_USED")
            if item.get("reject_reason") not in {
                "UNKNOWN_ANSWER",
                "INVALID_LOCAL_ANSWER",
            }:
                if item.get("normalizer_used") is not True:
                    violations.append(f"{prefix}_NORMALIZER_NOT_USED")
                if item.get("llm_called") is not True:
                    violations.append(f"{prefix}_NORMALIZER_LLM_NOT_CALLED")
            if item.get("adapter_decision") != "NOT_USED":
                violations.append(f"{prefix}_DECISION_NOT_NOT_USED")
            adapter_input = item.get("adapter_input") or {}
            for forbidden in (
                "local_question",
                "resolution_target",
                "retrieved_evidence",
                "supporting_evidence",
            ):
                if forbidden in adapter_input:
                    violations.append(f"{prefix}_{forbidden.upper()}_IN_INPUT")
    return violations


def is_retryable_network_failure(row: dict[str, Any]) -> bool:
    """Return true only for saved rows affected by model/API transport errors."""

    runtime_stages = ((row.get("runtime_metrics") or {}).get("llm") or {})
    if any(
        int(metrics.get("failed_llm_calls", 0) or 0) > 0
        for metrics in runtime_stages.values()
        if isinstance(metrics, dict)
    ):
        return True

    messages = [
        str(item.get("failure_reason") or "")
        for item in ((row.get("state") or {}).get("failures") or [])
        if isinstance(item, dict)
    ]
    messages.append(str((row.get("final_rag_result") or {}).get("error") or ""))
    return any(_looks_like_model_api_failure(message) for message in messages)


def _looks_like_model_api_failure(message: str) -> bool:
    normalized = " ".join(str(message or "").casefold().split())
    if not normalized:
        return False
    # Errors raised by scripts.model_client always carry both fields. This is
    # more precise than matching a generic word such as "network", which may
    # legitimately appear in a question or resolution target.
    if re.search(r"\bstage=\S+\s+status=", normalized):
        return True
    return any(
        marker in normalized
        for marker in (
            "timed out",
            "timeout",
            "connection reset",
            "connection refused",
            "remote end closed",
            "temporary failure in name resolution",
            "service unavailable",
            "bad gateway",
            "gateway timeout",
            "too many requests",
            "rate limit",
            "certificate_verify_failed",
            "unexpected_eof_while_reading",
        )
    )


def load_resume_rows(
    details_path: Path,
    *,
    expected_signature: dict[str, Any],
    resume: bool,
    compatible_question_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    if not resume or not details_path.exists():
        return []
    rows_by_id: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    dirty = False
    for line_number, line in enumerate(
        details_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            dirty = True
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            dirty = True
            break
        if not isinstance(row, dict):
            dirty = True
            break
        row_signature = row.get("run_signature")
        compatible_prefix_length = _compatible_question_id_prefix_length(
            row_signature,
            expected_signature,
            compatible_question_ids,
        )
        if not _resume_signatures_match(
            row_signature,
            expected_signature,
            compatible_question_ids=compatible_question_ids,
        ):
            raise SystemExit(
                "Existing checkpoint does not match this run configuration "
                f"(line {line_number}: {details_path}). Use --no-resume or a "
                "different --details-output path."
            )
        if row_signature != expected_signature:
            # ``num_questions`` used to be stored in every row. It is a target
            # size, not an experiment configuration, so migrate compatible
            # checkpoints in place and allow 100 -> 250 -> ... expansion.
            row["run_signature"] = dict(expected_signature)
            dirty = True
        example_id = str(row.get("example_id") or "").strip()
        if not example_id:
            dirty = True
            continue
        if (
            compatible_prefix_length is not None
            and compatible_question_ids is not None
            and example_id
            not in set(compatible_question_ids[:compatible_prefix_length])
        ):
            raise SystemExit(
                "Checkpoint contains an example outside the compatible "
                f"question-ID prefix: {example_id}"
            )
        if example_id not in rows_by_id:
            order.append(example_id)
        else:
            dirty = True
        rows_by_id[example_id] = row
    rows = [rows_by_id[item] for item in order]
    if dirty:
        write_jsonl_atomic(details_path, rows)
    return rows


def select_examples_for_run(
    all_examples: list[Any],
    *,
    target_count: int,
    selection: str,
    seed: int,
    completed_ids: list[str] | None = None,
    required_ids: list[str] | None = None,
) -> list[Any]:
    """Select a prefix-stable target set while preserving resumed examples."""

    if target_count <= 0:
        raise ValueError("target_count must be positive")
    if selection not in {"random", "first"}:
        raise ValueError(f"Unsupported selection mode: {selection}")

    by_id = {str(item.example_id): item for item in all_examples}
    if required_ids is not None:
        selected_ids = [str(item or "").strip() for item in required_ids[:target_count]]
        if len(selected_ids) < target_count:
            raise SystemExit(
                f"Question-ID source contains only {len(selected_ids)} unique IDs; "
                f"{target_count} are required."
            )
        if len(set(selected_ids)) != len(selected_ids):
            raise SystemExit("Question-ID source contains duplicate example IDs.")
        missing = [item for item in selected_ids if item not in by_id]
        if missing:
            raise SystemExit(
                "Question-ID source contains IDs absent from the current dataset: "
                + ", ".join(missing[:5])
            )
        outside = [
            str(item or "").strip()
            for item in (completed_ids or [])
            if str(item or "").strip() not in set(selected_ids)
        ]
        if outside:
            raise SystemExit(
                "Checkpoint contains IDs outside the paired question set: "
                + ", ".join(outside[:5])
            )
        return [by_id[item] for item in selected_ids]

    completed: list[Any] = []
    seen: set[str] = set()
    for raw_id in completed_ids or []:
        example_id = str(raw_id or "").strip()
        if not example_id or example_id in seen:
            continue
        example = by_id.get(example_id)
        if example is None:
            raise SystemExit(
                "Checkpoint contains an example that is absent from the current "
                f"dataset: {example_id}"
            )
        completed.append(example)
        seen.add(example_id)

    if len(completed) > target_count:
        raise SystemExit(
            f"Checkpoint already contains {len(completed)} questions, which is "
            f"greater than the requested target {target_count}. Increase "
            "--num-questions or use --no-resume with different output paths."
        )
    if len(completed) == target_count:
        return completed

    candidates = list(all_examples)
    if selection == "random":
        # Unlike random.sample(population, k), shuffle has a stable prefix when
        # k grows. Existing checkpoints created by the older sampler are still
        # preserved explicitly above and only the new tail uses this order.
        random.Random(seed).shuffle(candidates)

    selected = list(completed)
    for example in candidates:
        example_id = str(example.example_id)
        if example_id in seen:
            continue
        selected.append(example)
        seen.add(example_id)
        if len(selected) >= min(target_count, len(all_examples)):
            break
    return selected


def load_question_ids(
    source: str | Path | None,
    *,
    target_count: int,
) -> list[str] | None:
    """Load ordered, unique example IDs from a Full-run details JSONL."""

    if source is None:
        return None
    path = Path(source)
    if not path.exists():
        raise SystemExit(f"Question-ID source does not exist: {path}")
    ids: list[str] = []
    seen: set[str] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(
                f"Invalid JSON in question-ID source at line {line_number}: {path}"
            ) from exc
        example_id = str((row or {}).get("example_id") or "").strip()
        if not example_id:
            raise SystemExit(
                f"Missing example_id in question-ID source line {line_number}: {path}"
            )
        if example_id in seen:
            raise SystemExit(
                f"Duplicate example_id in question-ID source: {example_id}"
            )
        seen.add(example_id)
        ids.append(example_id)
    if len(ids) < target_count:
        raise SystemExit(
            f"Question-ID source contains {len(ids)} rows, but {target_count} are required: {path}"
        )
    return ids


def question_ids_digest(question_ids: list[str] | None) -> str | None:
    if question_ids is None:
        return None
    payload = "\n".join(question_ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _resume_signatures_match(
    actual: Any,
    expected: dict[str, Any],
    *,
    compatible_question_ids: list[str] | None = None,
) -> bool:
    if not isinstance(actual, dict):
        return False
    if _signature_without_target_size(actual) == _signature_without_target_size(
        expected
    ):
        return True
    return _compatible_question_id_prefix_length(
        actual,
        expected,
        compatible_question_ids,
    ) is not None


def _compatible_question_id_prefix_length(
    actual: Any,
    expected: dict[str, Any],
    question_ids: list[str] | None,
) -> int | None:
    """Return the old fixed-prefix length for an explicitly allowed expansion."""

    if not isinstance(actual, dict) or not question_ids:
        return None
    actual_value = _signature_without_target_size(actual)
    expected_value = _signature_without_target_size(expected)
    actual_digest = actual_value.pop("question_ids_digest", None)
    expected_digest = expected_value.pop("question_ids_digest", None)
    if actual_value != expected_value:
        return None
    if expected_digest != question_ids_digest(question_ids):
        return None
    if not isinstance(actual_digest, str) or not actual_digest:
        return None
    for length in range(1, len(question_ids)):
        if actual_digest == question_ids_digest(question_ids[:length]):
            return length
    return None


def _signature_without_target_size(signature: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(signature)
    normalized.pop("num_questions", None)
    return normalized


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    for attempt in range(6):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            # Windows antivirus/indexing can hold the destination briefly.
            time.sleep(0.2 * (attempt + 1))


def _model_client(args: argparse.Namespace, name: str, label: str, response_format: dict[str, Any] | None):
    config = select_model_config(args.config, name)
    fallback_name = getattr(args, "content_filter_fallback_model_config", None)
    fallback_config = (
        select_model_config(args.config, fallback_name)
        if fallback_name
        else None
    )
    return make_llm_client(
        config,
        client_label=label,
        response_format=response_format,
        max_output_tokens=config.max_output_tokens_by_stage.get(label, 1800),
        content_filter_fallback_config=fallback_config,
    )


def _row_llm_total(row: dict[str, Any]) -> dict[str, Any]:
    runtime = row.get("runtime_metrics") or {}
    total = runtime.get("llm_total")
    if isinstance(total, dict):
        return dict(total)
    stages = runtime.get("llm") or {}
    if isinstance(stages, dict):
        return combine_llm_metrics(
            [item for item in stages.values() if isinstance(item, dict)]
        )
    return combine_llm_metrics([])


def _summarize_llm_by_stage(
    rows: list[dict[str, Any]],
    question_count: int,
) -> dict[str, dict[str, Any]]:
    names = sorted(
        {
            str(name)
            for row in rows
            for name in (((row.get("runtime_metrics") or {}).get("llm") or {}).keys())
        }
    )
    result: dict[str, dict[str, Any]] = {}
    for name in names:
        metrics = combine_llm_metrics(
            [
                ((row.get("runtime_metrics") or {}).get("llm") or {}).get(name, {})
                for row in rows
            ]
        )
        result[name] = {
            **metrics,
            "average_llm_calls_per_question": _divide_or_none(
                metrics.get("llm_calls"), question_count
            ),
            "average_total_tokens_per_question_reported": _divide_or_none(
                metrics.get("total_tokens"), question_count
            ),
        }
    return result


def _divide_or_none(value: Any, denominator: int) -> float | None:
    if value is None or denominator <= 0:
        return None
    return float(value) / denominator


def _average(values) -> float | None:
    items = [float(item) for item in values if item is not None]
    return sum(items) / len(items) if items else None


def _answer_is_unknown(answer: str) -> bool:
    normalized = " ".join(answer.strip().casefold().split())
    return not normalized or normalized in {
        "unknown",
        "无法确定",
        "cannot determine",
        "not enough information",
        "insufficient information",
    }


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _format_metric(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "N/A"


if __name__ == "__main__":
    raise SystemExit(main())
