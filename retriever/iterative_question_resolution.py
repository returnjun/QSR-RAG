from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from .curiosity_question_prompts import normalize_curiosity_dataset_name
from .question_binding import (
    RewritePolicy,
    build_verified_resolution,
    normalize_rewrite_policy,
    render_verified_resolution,
)


MAX_ROUNDS = 4
MAX_QUERIES_PER_ROUND = 2

STOP_NO_NEW_GAP = "NO_NEW_GAP"
STOP_GENERATOR_FAILED = "GENERATOR_FAILED"
STOP_LOCAL_RESOLUTION_FAILED = "LOCAL_RESOLUTION_FAILED"
STOP_REWRITER_FAILED = "REWRITER_FAILED"
STOP_MAX_ROUNDS = "MAX_ROUNDS_REACHED"

DATASET_RUNTIME_CONFIG = {
    "hotpotqa": {
        "max_rounds": MAX_ROUNDS,
        "max_queries_per_round": MAX_QUERIES_PER_ROUND,
    },
    "2wikimultihopqa": {
        "max_rounds": MAX_ROUNDS,
        "max_queries_per_round": MAX_QUERIES_PER_ROUND,
    },
    "musique": {
        "max_rounds": MAX_ROUNDS,
        "max_queries_per_round": MAX_QUERIES_PER_ROUND,
    },
}


@dataclass(frozen=True, slots=True)
class AblationConfig:
    """One explicit, serializable configuration for an ablation run."""

    name: str = "qsr_rag"
    display_name: str = "QSR-RAG"
    termination_mode: str = "question_guided"
    rewrite_policy: RewritePolicy = RewritePolicy.HYBRID
    final_known_fact_mode: str = "always"
    max_rounds: int = MAX_ROUNDS

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["rewrite_policy"] = normalize_rewrite_policy(
            self.rewrite_policy
        ).value
        return value


ABLATION_PRESETS = {
    "qsr_rag": AblationConfig(
        name="qsr_rag",
        display_name="QSR-RAG",
    ),
    "memory_only": AblationConfig(
        name="memory_only",
        display_name="Memory Only",
        rewrite_policy=RewritePolicy.MEMORY_ONLY,
    ),
    "entity_binding_only": AblationConfig(
        name="entity_binding_only",
        display_name="Entity Binding Only",
        rewrite_policy=RewritePolicy.ENTITY_BINDING_ONLY,
    ),
    "context_injection_only": AblationConfig(
        name="context_injection_only",
        display_name="Context Injection Only",
        rewrite_policy=RewritePolicy.CONTEXT_INJECTION_ONLY,
    ),
}


@dataclass(frozen=True, slots=True)
class VerificationAblationConfig:
    """The only varying mechanism in the verification ablation."""

    name: str
    display_name: str
    adapter_used: bool
    evidence_visible: bool | None
    acceptance_policy: str
    normalizer_used: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


VERIFICATION_ABLATION_PRESETS = {
    "full": VerificationAblationConfig(
        name="full",
        display_name="Full Evidence-Grounded Verification",
        adapter_used=True,
        normalizer_used=True,
        evidence_visible=True,
        acceptance_policy="adapter_pass_only",
    ),
    "evidence_blind": VerificationAblationConfig(
        name="evidence_blind",
        display_name="Evidence-Blind Verification",
        adapter_used=True,
        normalizer_used=True,
        evidence_visible=False,
        acceptance_policy="adapter_pass_only",
    ),
    "no_verification": VerificationAblationConfig(
        name="no_verification",
        display_name="No Verification (Parsing Only)",
        adapter_used=False,
        normalizer_used=True,
        evidence_visible=None,
        acceptance_policy="parsed_answer_items_directly_accepted",
    ),
}

_VALID_TERMINATION_MODES = {"question_guided", "fixed_budget"}

QuestionGenerator = Callable[..., dict[str, Any]]
ResolutionAdapter = Callable[..., dict[str, Any]]
ResolutionRewriter = Callable[..., dict[str, Any]]


@dataclass(slots=True)
class ResolutionState:
    dataset: str
    target_question: str
    reasoning_question: str
    round_index: int = 0
    question_version: int = 0
    verified_resolutions: list[dict[str, Any]] = field(default_factory=list)
    executed_queries: list[str] = field(default_factory=list)
    local_rag_history: list[dict[str, Any]] = field(default_factory=list)
    rewrite_history: list[dict[str, Any]] = field(default_factory=list)
    accumulated_documents: list[dict[str, Any]] = field(default_factory=list)
    final_evidence_pool: list[dict[str, Any]] = field(default_factory=list)
    rounds: list[dict[str, Any]] = field(default_factory=list)
    failures: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str | None = None
    retrieval_calls: int = 0
    reader_calls: int = 0
    first_no_gap_round: int | None = None
    post_no_gap_retrievals: int = 0

    @property
    def original_question(self) -> str:
        return self.target_question

    @property
    def current_question(self) -> str:
        return self.reasoning_question

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["original_question"] = self.target_question
        value["current_question"] = self.reasoning_question
        value["verified_facts"] = [dict(item) for item in self.verified_resolutions]
        value["accepted_resolutions"] = [
            dict(item) for item in self.verified_resolutions
        ]
        value["standard_facts"] = _verified_resolution_texts(
            self.verified_resolutions
        )
        return value


def run_iterative_resolution(
    *,
    original_question: str,
    dataset: str,
    question_generator: QuestionGenerator,
    normal_rag: Any,
    answer_fact_adapter: ResolutionAdapter | None,
    resolution_rewriter: ResolutionRewriter | None = None,
    ablation_config: AblationConfig | None = None,
    verification_config: VerificationAblationConfig | None = None,
) -> dict[str, Any]:
    """Run the configured Generator -> local resolution -> state-update loop.

    Every registered rewrite experiment uses the same question-guided empty
    query-list stop condition. Component failures stop iteration, and all
    evidence collected so far is still sent to the final answerer.
    """

    config = ablation_config or ABLATION_PRESETS["qsr_rag"]
    verification = verification_config or VERIFICATION_ABLATION_PRESETS["full"]
    _validate_ablation_config(config)
    _validate_verification_config(verification)
    rewrite_policy = normalize_rewrite_policy(config.rewrite_policy)
    dataset_name = normalize_curiosity_dataset_name(dataset)
    target = str(original_question or "").strip()
    if not target:
        raise ValueError("original_question must not be empty")
    if (
        rewrite_policy is not RewritePolicy.MEMORY_ONLY
        and resolution_rewriter is None
    ):
        raise ValueError("resolution_rewriter is required")
    if answer_fact_adapter is None:
        raise ValueError("answer_fact_adapter is required for answer normalization")

    state = ResolutionState(dataset_name, target, target)

    for round_index in range(1, config.max_rounds + 1):
        state.round_index = round_index
        round_log: dict[str, Any] = {
            "round": round_index,
            "input_question": state.reasoning_question,
            "question_version": state.question_version,
            "generated_query_specs": [],
            "executed_query_specs": [],
            "normal_rag_results": [],
            "adapted_facts": [],
            "candidate_resolutions": [],
            "accepted_resolutions": [],
            "verified_resolutions": [],
            "rewrite_result": None,
            "output_question": state.reasoning_question,
            "failure": None,
        }
        state.rounds.append(round_log)

        try:
            generated = question_generator(
                original_question=state.target_question,
                current_question=state.reasoning_question,
                verified_facts=[dict(item) for item in state.verified_resolutions],
                previous_questions=list(state.executed_queries),
                dataset=state.dataset,
            )
        except Exception as exc:  # noqa: BLE001
            failure = _record_failure(
                state,
                stage="generator",
                failure_type="GENERATOR_EXCEPTION",
                reason=str(exc),
            )
            round_log["failure"] = failure
            state.stop_reason = STOP_GENERATOR_FAILED
            break

        round_log["generator_result"] = generated
        try:
            query_specs = _parse_generator_output(generated)
        except Exception as exc:  # noqa: BLE001
            failure = _record_failure(
                state,
                stage="generator",
                failure_type="INVALID_OUTPUT",
                reason=str(exc),
            )
            round_log["failure"] = failure
            state.stop_reason = STOP_GENERATOR_FAILED
            break

        if not query_specs:
            round_log["no_new_gap_detected"] = True
            if state.first_no_gap_round is None:
                state.first_no_gap_round = round_index
            if config.termination_mode == "question_guided":
                state.stop_reason = STOP_NO_NEW_GAP
                break
            try:
                _run_fixed_budget_fallback(
                    state=state,
                    normal_rag=normal_rag,
                    round_log=round_log,
                    round_index=round_index,
                )
            except Exception as exc:  # noqa: BLE001
                failure = _record_failure(
                    state,
                    stage="fixed_budget_fallback",
                    failure_type="RAG_EXCEPTION",
                    reason=str(exc),
                )
                round_log["failure"] = failure
                state.stop_reason = STOP_LOCAL_RESOLUTION_FAILED
                break
            round_log["output_question"] = state.reasoning_question
            continue

        resolved_target = next(
            (
                item["resolution_target"]
                for item in query_specs
                if _target_already_resolved(
                    item["resolution_target"], state.verified_resolutions
                )
            ),
            None,
        )
        if resolved_target is not None:
            failure = _record_failure(
                state,
                stage="generator",
                failure_type="GENERATED_RESOLVED_TARGET",
                reason=f"resolution_target already resolved: {resolved_target}",
            )
            round_log["failure"] = failure
            state.stop_reason = STOP_GENERATOR_FAILED
            break

        query_specs = [
            {**item, "query_id": f"R{round_index}Q{index}"}
            for index, item in enumerate(
                query_specs[:MAX_QUERIES_PER_ROUND], start=1
            )
        ]
        round_log["generated_query_specs"] = [dict(item) for item in query_specs]

        new_resolutions: list[dict[str, Any]] = []
        round_failed = False

        for branch_index, query_spec in enumerate(query_specs, start=1):
            question = query_spec["question"]
            state.executed_queries.append(question)
            round_log["executed_query_specs"].append(dict(query_spec))

            try:
                rag_result = normal_rag.run(
                    question=question,
                    return_documents=True,
                    answer_mode="local",
                )
                state.retrieval_calls += 1
                state.reader_calls += 1
            except Exception as exc:  # noqa: BLE001
                failure = _record_failure(
                    state,
                    stage="local_rag",
                    failure_type="RAG_EXCEPTION",
                    reason=str(exc),
                    branch=branch_index,
                    query=query_spec,
                )
                round_log["failure"] = failure
                round_failed = True
                break

            if not isinstance(rag_result, dict):
                failure = _record_failure(
                    state,
                    stage="local_rag",
                    failure_type="INVALID_RAG_OUTPUT",
                    reason="normal_rag.run must return a dictionary",
                    branch=branch_index,
                    query=query_spec,
                )
                round_log["failure"] = failure
                round_failed = True
                break

            documents = [
                dict(item)
                for item in (rag_result.get("documents") or [])
                if isinstance(item, dict)
            ]
            support_ids_raw = [
                str(item) for item in (rag_result.get("supporting_evidence_ids") or [])
            ]
            pool_documents, support_ids = _scope_local_evidence(
                documents,
                support_ids_raw,
                query_spec["query_id"] + "-",
            )
            rag_log = {
                "round": round_index,
                "branch": branch_index,
                **query_spec,
                "rag_result": rag_result,
                "pool_documents": pool_documents,
                "supporting_evidence_ids": support_ids,
            }
            state.local_rag_history.append(rag_log)
            round_log["normal_rag_results"].append(rag_log)
            state.accumulated_documents = _merge_documents(
                state.accumulated_documents, pool_documents
            )

            if rag_result.get("error"):
                failure = _record_failure(
                    state,
                    stage="local_rag",
                    failure_type="RAG_ERROR",
                    reason=str(rag_result["error"]),
                    branch=branch_index,
                    query=query_spec,
                )
                round_log["failure"] = failure
                round_failed = True
                break

            supporting_evidence = _select_supporting_evidence(
                documents, support_ids_raw
            )
            try:
                acceptance_inputs = {
                    "local_question": question,
                    "resolution_target": query_spec["resolution_target"],
                    "rag_answer": rag_result.get("answer", ""),
                    "reader_answer_type": str(rag_result.get("answer_type") or ""),
                    "supporting_evidence": supporting_evidence,
                }
                assert answer_fact_adapter is not None
                adapted = answer_fact_adapter(
                    **acceptance_inputs,
                    evidence_visible=(verification.evidence_visible is True),
                    verification_mode=verification.name,
                )
            except Exception as exc:  # noqa: BLE001
                failure = _record_failure(
                    state,
                    stage=("adapter" if verification.adapter_used else "normalizer"),
                    failure_type=(
                        "ADAPTER_EXCEPTION"
                        if verification.adapter_used
                        else "NORMALIZER_EXCEPTION"
                    ),
                    reason=str(exc),
                    branch=branch_index,
                    query=query_spec,
                )
                round_log["failure"] = failure
                round_failed = True
                break

            if not isinstance(adapted, dict):
                failure = _record_failure(
                    state,
                    stage=("adapter" if verification.adapter_used else "normalizer"),
                    failure_type=(
                        "INVALID_ADAPTER_OUTPUT"
                        if verification.adapter_used
                        else "INVALID_NORMALIZER_OUTPUT"
                    ),
                    reason="answer_fact_adapter must return a dictionary",
                    branch=branch_index,
                    query=query_spec,
                )
                round_log["failure"] = failure
                round_failed = True
                break

            adapted_log = {
                **adapted,
                "round": round_index,
                "branch": branch_index,
                **query_spec,
                "rag_answer": str(rag_result.get("answer") or ""),
                "supporting_evidence_ids": support_ids,
                "reasoning_question_before": state.reasoning_question,
            }
            round_log["adapted_facts"].append(adapted_log)
            round_log["candidate_resolutions"].append(adapted_log)

            if adapted.get("usable") is not True:
                adapted_log["accepted_resolution"] = None
                adapted_log["reasoning_question_after"] = state.reasoning_question
                failure_type = str(
                    adapted.get("reject_reason")
                    or ("INVALID_ADAPTER_PAYLOAD" if adapted.get("parse_error") else "ADAPTER_REJECTED")
                )
                failure = _record_failure(
                    state,
                    stage=("adapter" if verification.adapter_used else "normalizer"),
                    failure_type=failure_type,
                    reason=str(
                        adapted.get("error")
                        or adapted.get("parse_error")
                        or adapted.get("reject_reason")
                        or "local answer was not usable"
                    ),
                    branch=branch_index,
                    query=query_spec,
                )
                round_log["failure"] = failure
                round_failed = True
                break

            resolution = build_verified_resolution(
                query_id=query_spec["query_id"],
                question_version=state.question_version,
                local_question=question,
                resolution_target=query_spec["resolution_target"],
                adapted=adapted,
                round_index=round_index,
                branch_index=branch_index,
                supporting_evidence_ids=support_ids,
                current_question=state.reasoning_question,
            )
            state.verified_resolutions.append(resolution)
            new_resolutions.append(resolution)
            round_log["verified_resolutions"].append(resolution)
            round_log["accepted_resolutions"].append(resolution)
            adapted_log["accepted_resolution"] = dict(resolution)

        if round_failed:
            for item in round_log["candidate_resolutions"]:
                item.setdefault("reasoning_question_after", state.reasoning_question)
            state.stop_reason = STOP_LOCAL_RESOLUTION_FAILED
            break

        if rewrite_policy is RewritePolicy.MEMORY_ONLY:
            memory_log = {
                "mode": "MEMORY_ONLY",
                "policy": rewrite_policy.value,
                "update_success": False,
                "before": state.reasoning_question,
                "after": state.reasoning_question,
                "updated_question": state.reasoning_question,
                "used_query_ids": [],
                "stored_query_ids": [
                    str(item.get("query_id") or "") for item in new_resolutions
                ],
                "validation_errors": [],
                "violations": [],
                "question_version_before": state.question_version,
                "question_version_after": state.question_version,
                "llm_called": False,
            }
            round_log["rewrite_result"] = memory_log
            state.rewrite_history.append(memory_log)
            round_log["output_question"] = state.reasoning_question
            for item in round_log["candidate_resolutions"]:
                item["reasoning_question_after"] = state.reasoning_question
            continue

        try:
            assert resolution_rewriter is not None
            rewrite = resolution_rewriter(
                dataset=state.dataset,
                original_question=state.target_question,
                current_question=state.reasoning_question,
                question_version=state.question_version,
                new_resolutions=[dict(item) for item in new_resolutions],
                rewrite_policy=rewrite_policy,
            )
        except Exception as exc:  # noqa: BLE001
            failure = _record_failure(
                state,
                stage="rewriter",
                failure_type="REWRITER_EXCEPTION",
                reason=str(exc),
            )
            round_log["failure"] = failure
            state.stop_reason = STOP_REWRITER_FAILED
            break

        try:
            _validate_rewriter_output(rewrite)
        except Exception as exc:  # noqa: BLE001
            failure = _record_failure(
                state,
                stage="rewriter",
                failure_type="INVALID_OUTPUT",
                reason=str(exc),
            )
            round_log["failure"] = failure
            state.stop_reason = STOP_REWRITER_FAILED
            break

        rewrite_log = dict(rewrite)
        round_log["rewrite_result"] = rewrite_log
        state.rewrite_history.append(rewrite_log)

        if rewrite["update_success"]:
            state.reasoning_question = str(rewrite["updated_question"]).strip()
            state.question_version += 1

        round_log["output_question"] = state.reasoning_question
        for item in round_log["candidate_resolutions"]:
            item["reasoning_question_after"] = state.reasoning_question
        # UPDATE and KEEP both return control to the Generator.
    else:
        state.stop_reason = STOP_MAX_ROUNDS

    final_result = _run_final_evidence_fusion(state=state, normal_rag=normal_rag)
    return {
        "answer": str(final_result.get("answer") or ""),
        "stop_reason": state.stop_reason,
        "completed_normally": (
            state.stop_reason == STOP_NO_NEW_GAP
            or (
                config.termination_mode == "fixed_budget"
                and state.stop_reason == STOP_MAX_ROUNDS
            )
        ),
        "ablation_config": config.to_dict(),
        "verification_ablation": verification.to_dict(),
        "target_question": state.target_question,
        "reasoning_question": state.reasoning_question,
        "final_question": state.reasoning_question,
        "state": state.to_dict(),
        "final_rag_result": final_result,
    }


def _validate_ablation_config(config: AblationConfig) -> None:
    if not isinstance(config, AblationConfig):
        raise TypeError("ablation_config must be an AblationConfig")
    if config.termination_mode not in _VALID_TERMINATION_MODES:
        raise ValueError(
            f"unsupported termination_mode: {config.termination_mode}"
        )
    normalize_rewrite_policy(config.rewrite_policy)
    if config.final_known_fact_mode != "always":
        raise ValueError("final_known_fact_mode must be 'always'")
    if config.max_rounds <= 0:
        raise ValueError("ablation max_rounds must be positive")


def _validate_verification_config(config: VerificationAblationConfig) -> None:
    if not isinstance(config, VerificationAblationConfig):
        raise TypeError("verification_config must be a VerificationAblationConfig")
    if config.adapter_used and not isinstance(config.evidence_visible, bool):
        raise ValueError("Adapter-based verification must declare evidence visibility")
    if not config.adapter_used and config.evidence_visible is not None:
        raise ValueError("No-Verification must not declare evidence visibility")
    if not config.normalizer_used:
        raise ValueError("Every verification condition must preserve answer parsing")


def _run_fixed_budget_fallback(
    *,
    state: ResolutionState,
    normal_rag: Any,
    round_log: dict[str, Any],
    round_index: int,
) -> None:
    """Spend one real retrieval after a no-gap signal without inventing a gap."""

    rag_result = normal_rag.run(
        question=state.reasoning_question,
        return_documents=True,
        answer_mode="local",
    )
    state.retrieval_calls += 1
    state.reader_calls += 1
    state.post_no_gap_retrievals += 1
    if not isinstance(rag_result, dict):
        raise TypeError("normal_rag.run must return a dictionary")
    if rag_result.get("error"):
        raise RuntimeError(str(rag_result["error"]))

    documents = [
        dict(item)
        for item in (rag_result.get("documents") or [])
        if isinstance(item, dict)
    ]
    support_ids_raw = [
        str(item) for item in (rag_result.get("supporting_evidence_ids") or [])
    ]
    pool_documents, support_ids = _scope_local_evidence(
        documents,
        support_ids_raw,
        f"FB{round_index}-",
    )
    fallback_log = {
        "round": round_index,
        "branch": 0,
        "query_id": f"FB{round_index}",
        "question": state.reasoning_question,
        "resolution_target": "",
        "ablation_role": "FIXED_BUDGET_AFTER_NO_GAP",
        "rag_result": rag_result,
        "pool_documents": pool_documents,
        "supporting_evidence_ids": support_ids,
    }
    state.local_rag_history.append(fallback_log)
    round_log["fixed_budget_fallback"] = fallback_log
    state.accumulated_documents = _merge_documents(
        state.accumulated_documents,
        pool_documents,
    )


def _parse_generator_output(generated: Any) -> list[dict[str, str]]:
    if not isinstance(generated, dict):
        raise TypeError("question_generator must return a dictionary")

    if "generator_schema_valid" in generated and generated.get("generator_schema_valid") is not True:
        raise ValueError("generator JSON/schema validation failed")
    if "query_spec_semantic_valid" in generated and generated.get("query_spec_semantic_valid") is not True:
        errors = generated.get("validation_errors") or []
        raise ValueError(f"generator QuerySpec validation failed: {errors}")

    if "generated_queries" in generated:
        items = generated["generated_queries"]
    elif "questions" in generated:
        items = generated["questions"]
    elif isinstance(generated.get("result"), dict) and "questions" in generated["result"]:
        items = generated["result"]["questions"]
    else:
        raise ValueError("generator output is missing questions")

    if not isinstance(items, list):
        raise TypeError("generator questions must be a list")

    parsed: list[dict[str, str]] = []
    seen_questions: set[str] = set()
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise TypeError(f"generator question {index} must be an object")
        if set(item) != {"question", "resolution_target"}:
            raise ValueError(
                f"generator question {index} must contain exactly question and resolution_target"
            )
        question = item.get("question")
        resolution_target = item.get("resolution_target")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"generator question {index} has an invalid question")
        if not isinstance(resolution_target, str) or not resolution_target.strip():
            raise ValueError(
                f"generator question {index} has an invalid resolution_target"
            )
        normalized_question = _normalize_question(question)
        if normalized_question in seen_questions:
            continue
        seen_questions.add(normalized_question)
        parsed.append(
            {
                "question": " ".join(question.split()),
                "resolution_target": " ".join(resolution_target.split()),
            }
        )
    return parsed


def _validate_rewriter_output(rewrite: Any) -> None:
    if not isinstance(rewrite, dict):
        raise TypeError("resolution_rewriter must return a dictionary")
    required = {"update_success", "updated_question", "used_query_ids"}
    missing = required - set(rewrite)
    if missing:
        raise ValueError(f"rewriter output is missing fields: {sorted(missing)}")
    if not isinstance(rewrite["update_success"], bool):
        raise TypeError("rewriter update_success must be boolean")
    if not isinstance(rewrite["updated_question"], str) or not rewrite["updated_question"].strip():
        raise ValueError("rewriter updated_question must be a non-empty string")
    if not isinstance(rewrite["used_query_ids"], list) or not all(
        isinstance(item, str) and item.strip() for item in rewrite["used_query_ids"]
    ):
        raise TypeError("rewriter used_query_ids must be a string list")


def _record_failure(
    state: ResolutionState,
    *,
    stage: str,
    failure_type: str,
    reason: str,
    branch: int | None = None,
    query: dict[str, Any] | None = None,
) -> dict[str, Any]:
    failure = {
        "round": state.round_index,
        "failure_stage": stage,
        "failure_type": str(failure_type or "UNKNOWN"),
        "failure_reason": str(reason or "").strip(),
    }
    if branch is not None:
        failure["branch"] = branch
    if query is not None:
        failure["query"] = dict(query)
    state.failures.append(failure)
    return failure


def _run_final_evidence_fusion(*, state: ResolutionState, normal_rag: Any) -> dict[str, Any]:
    known_facts = _verified_resolution_texts(state.verified_resolutions)
    retrieve = getattr(normal_rag, "retrieve_documents", None)
    answer_from_evidence = getattr(normal_rag, "answer_from_evidence", None)
    if not callable(retrieve) or not callable(answer_from_evidence):
        result = normal_rag.run(
            question=state.reasoning_question,
            answer_mode="final",
            target_question=state.target_question,
            known_facts=known_facts,
            return_documents=True,
        )
        state.retrieval_calls += 1
        state.reader_calls += 1
        state.accumulated_documents = _merge_documents(
            state.accumulated_documents,
            list(result.get("documents") or []),
        )
        return _attach_final_reader_contract(
            result=result,
            state=state,
            known_facts=known_facts,
        )

    target_retrieval = retrieve(
        question=state.target_question,
        top_k_sentences=15,
        evidence_id_prefix="T-E",
    )
    state.retrieval_calls += 1

    reasoning_retrieval: dict[str, Any] | None = None
    if _normalize_question(state.reasoning_question) != _normalize_question(
        state.target_question
    ):
        reasoning_retrieval = retrieve(
            question=state.reasoning_question,
            top_k_sentences=15,
            evidence_id_prefix="C-E",
        )
        state.retrieval_calls += 1

    state.final_evidence_pool = _fuse_evidence_pool(
        cited_local=_cited_local_evidence(state.local_rag_history),
        target_documents=list(target_retrieval.get("documents") or []),
        reasoning_documents=list((reasoning_retrieval or {}).get("documents") or []),
        limit=30,
    )
    result = answer_from_evidence(
        target_question=state.target_question,
        reasoning_question=state.reasoning_question,
        evidence=state.final_evidence_pool,
        known_facts=known_facts,
    )
    state.reader_calls += 1
    state.accumulated_documents = _merge_documents(
        state.accumulated_documents, state.final_evidence_pool
    )
    return _attach_final_reader_contract(
        result=result,
        state=state,
        known_facts=known_facts,
    )
    result["retrieval_diagnostics"] = {
        "target_question": dict(
            target_retrieval.get("retrieval_diagnostics") or {}
        ),
        "current_reasoning_question": dict(
            (reasoning_retrieval or {}).get("retrieval_diagnostics") or {}
        ),
    }


def _attach_final_reader_contract(
    *,
    result: dict[str, Any],
    state: ResolutionState,
    known_facts: list[str],
) -> dict[str, Any]:
    value = dict(result)
    value["final_reader_inputs"] = {
        "target_question": state.target_question,
        "current_reasoning_question": state.reasoning_question,
        "verified_facts": list(known_facts),
        "known_fact_mode": "always",
    }
    return value


def _fuse_evidence_pool(
    *,
    cited_local: list[dict[str, Any]],
    target_documents: list[dict[str, Any]],
    reasoning_documents: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    ranked_groups = (
        _rank_documents(cited_local),
        _rank_documents(target_documents),
        _rank_documents(reasoning_documents),
    )
    fused: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for group in (item[:10] for item in ranked_groups):
        for item in group:
            key = _document_key(item)
            if key in seen:
                continue
            seen.add(key)
            fused.append(dict(item))
            if len(fused) >= limit:
                return fused
    for group in ranked_groups:
        for item in group:
            key = _document_key(item)
            if key in seen:
                continue
            seen.add(key)
            fused.append(dict(item))
            if len(fused) >= limit:
                return fused
    return fused


def _rank_documents(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        (dict(item) for item in documents if isinstance(item, dict)),
        key=lambda item: (
            item.get("score") is not None,
            float(item.get("score") or 0.0),
        ),
        reverse=True,
    )


def _cited_local_evidence(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in history:
        cited = set(item.get("supporting_evidence_ids") or [])
        if not cited:
            continue
        result.extend(
            dict(document)
            for document in item.get("pool_documents") or []
            if str(document.get("evidence_id") or "") in cited
        )
    return _merge_documents([], result)


def _scope_local_evidence(
    documents: list[dict[str, Any]],
    supporting_ids: list[str],
    prefix: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    id_map: dict[str, str] = {}
    scoped: list[dict[str, Any]] = []
    for index, item in enumerate(documents, start=1):
        copy = dict(item)
        old_id = str(copy.get("evidence_id") or f"E{index}")
        new_id = prefix + old_id
        id_map[old_id.casefold()] = new_id
        copy["evidence_id"] = new_id
        scoped.append(copy)
    mapped = [
        id_map[str(item).casefold()]
        for item in supporting_ids
        if str(item).casefold() in id_map
    ]
    return scoped, list(dict.fromkeys(mapped))


def _select_supporting_evidence(
    documents: list[dict[str, Any]], supporting_ids: list[str]
) -> list[dict[str, Any]]:
    cited = {
        str(item).strip().casefold() for item in supporting_ids if str(item).strip()
    }
    if not cited:
        return []
    return [
        dict(document)
        for document in documents
        if isinstance(document, dict)
        and str(
            document.get("evidence_id") or document.get("document_id") or ""
        ).strip().casefold()
        in cited
    ]


def _verified_resolution_texts(facts: list[dict[str, Any]]) -> list[str]:
    return [
        render_verified_resolution(item)
        for item in facts
        if str(item.get("resolution_target") or "").strip()
        and str(item.get("answer_exact") or "").strip()
    ]


def _normalize_resolution_value(value: str) -> str:
    tokens = re.findall(r"[\w]+", str(value or "").casefold())
    if tokens and tokens[0] in {"the", "a", "an"}:
        tokens = tokens[1:]
    return " ".join(tokens)


def _target_already_resolved(
    resolution_target: str,
    verified_resolutions: list[dict[str, Any]],
) -> bool:
    target_key = _normalize_resolution_value(resolution_target)
    return bool(
        target_key
        and any(
            _normalize_resolution_value(str(item.get("resolution_target") or ""))
            == target_key
            for item in verified_resolutions
        )
    )


def _merge_documents(
    existing: list[dict[str, Any]],
    new_documents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result = [dict(item) for item in existing]
    seen = {_document_key(item) for item in result}
    for item in new_documents:
        if not isinstance(item, dict):
            continue
        key = _document_key(item)
        if all(key) and key not in seen:
            result.append(dict(item))
            seen.add(key)
    return result


def _document_key(item: dict[str, Any]) -> tuple[str, str]:
    return (
        str(item.get("title") or "").strip().casefold(),
        str(item.get("text") or "").strip().casefold(),
    )


def _normalize_question(value: str) -> str:
    return " ".join(str(value or "").casefold().split()).rstrip(" ?")
