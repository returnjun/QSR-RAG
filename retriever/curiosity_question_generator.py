from __future__ import annotations

import inspect
import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable

from .curiosity_question_prompts import (
    build_curiosity_prompt,
    curiosity_prompt_version,
    is_explicit_parallel_question,
    normalize_curiosity_dataset_name,
)
from .normal_rag import parse_json_object


PromptClient = Callable[..., str]

GENERATED_QUERY_SCHEMA_G1A1: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["question", "resolution_target"],
    "properties": {
        "question": {"type": "string", "minLength": 1},
        "resolution_target": {"type": "string", "minLength": 1},
    },
}

CURIOSITY_RESPONSE_SCHEMA_G1A1: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "curiosity_query_generation_g1a1_query_only",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["questions"],
            "properties": {
                "questions": {
                    "type": "array",
                    "minItems": 0,
                    "maxItems": 2,
                    "items": GENERATED_QUERY_SCHEMA_G1A1,
                }
            },
        },
    },
}

# Compatibility alias for callers that have not yet renamed the imported symbol.
CURIOSITY_RESPONSE_SCHEMA_G1 = CURIOSITY_RESPONSE_SCHEMA_G1A1
CURIOSITY_RESPONSE_SCHEMA_V32 = CURIOSITY_RESPONSE_SCHEMA_G1A1
CURIOSITY_RESPONSE_SCHEMA = CURIOSITY_RESPONSE_SCHEMA_G1A1

DATASET_GENERATOR_CONFIGS: dict[str, dict[str, Any]] = {
    "hotpotqa": {"max_queries": 2},
    "2wikimultihopqa": {"max_queries": 2},
    "musique": {"max_queries": 2},
}

QUERY_SPEC_VALIDATION_ERROR_CODES = {
    "EMPTY_QUESTION_LIST",
    "TOO_MANY_QUESTIONS",
    "QUESTION_MARK_MISSING",
    "QUESTION_ALREADY_ASKED",
    "DUPLICATE_GENERATED_QUESTION",
    "CONTROL_TOKEN_INSIDE_QUESTION",
    "JSON_FRAGMENT_INSIDE_QUESTION",
    "DUPLICATE_RESOLUTION_TARGET",
}

WARNING_CODES = {
    "MULTIPLE_QUERIES_MAY_BE_DEPENDENT",
    "SYMMETRIC_BRANCH_MISSING",
}

_JSON_FRAGMENT_PATTERN = re.compile(
    r"[{}\[\]]|\"\s*(?:questions?|purpose)\s*\"\s*:",
    re.IGNORECASE,
)
@dataclass(slots=True)
class CuriosityQuerySpec:
    question: str
    resolution_target: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CuriosityQuerySpec":
        if not isinstance(value, dict):
            raise ValueError("each question must be a QuerySpec object")
        _require_exact_keys(
            value,
            {"question", "resolution_target"},
        )
        question_value = value.get("question")
        resolution_target_value = value.get("resolution_target")
        if not isinstance(question_value, str):
            raise ValueError("question must be a string")
        if not isinstance(resolution_target_value, str):
            raise ValueError("resolution_target must be a string")
        question = question_value.strip()
        resolution_target = resolution_target_value.strip()
        if not question:
            raise ValueError("question must be a non-empty string")
        if not resolution_target:
            raise ValueError("resolution_target must be a non-empty string")
        return cls(question, resolution_target)

    def to_dict(self) -> dict[str, str]:
        return {
            "question": self.question,
            "resolution_target": self.resolution_target,
        }


@dataclass(slots=True)
class CuriosityQueryResult:
    questions: list[CuriosityQuerySpec]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CuriosityQueryResult":
        if not isinstance(value, dict):
            raise ValueError("query generation result must be an object")
        _require_exact_keys(value, {"questions"})
        questions = value.get("questions")
        if not isinstance(questions, list):
            raise ValueError("questions must be an array of QuerySpec objects")
        return cls([CuriosityQuerySpec.from_dict(item) for item in questions])

    def to_dict(self) -> dict[str, Any]:
        return {"questions": [item.to_dict() for item in self.questions]}


def generate_curiosity_questions(
    *,
    original_question: str,
    verified_facts: list[dict[str, Any]],
    previous_questions: list[str],
    dataset: str,
    llm_client: PromptClient,
    response_format: dict[str, Any] | None = CURIOSITY_RESPONSE_SCHEMA_G1A1,
    current_question: str | None = None,
) -> dict[str, Any]:
    """Generate and mechanically normalize one G1 inside-out QuerySpec round."""

    original = str(original_question or "").strip()
    if not original:
        raise ValueError("original_question must not be empty")
    dataset_name = normalize_curiosity_dataset_name(dataset)
    active_question = str(current_question or original).strip()
    prompt = build_curiosity_prompt(
        original_question=original,
        current_question=active_question,
        verified_facts=verified_facts,
        previous_questions=previous_questions,
        dataset=dataset_name,
    )
    initial_raw_text = _call_generator(llm_client, prompt, response_format)
    result, parse_error = parse_query_generation(initial_raw_text)
    json_repair_attempted = result is None
    json_repaired_text = ""
    if result is None:
        json_repaired_text = _call_generator(
            llm_client,
            build_json_repair_prompt(prompt, initial_raw_text),
            response_format,
        )
        result, parse_error = parse_query_generation(json_repaired_text)

    raw_result = result.to_dict() if result is not None else None
    if result is None:
        normalized_result = None
        actions: list[dict[str, Any]] = []
        raw_errors: list[dict[str, Any]] = []
        errors = [
            _diagnostic(
                "GENERATOR_JSON_OR_SCHEMA_INVALID",
                "result",
                "",
                str(parse_error or "Invalid query-generator JSON."),
            )
        ]
        warnings: list[dict[str, Any]] = []
        generated_queries: list[dict[str, Any]] = []
    else:
        raw_errors, _ = validate_generated_queries(
            result=result,
            original_question=active_question,
            previous_questions=previous_questions,
            dataset=dataset_name,
        )
        normalized, actions = normalize_generated_queries(
            result=result,
            original_question=active_question,
            dataset=dataset_name,
        )
        errors, warnings = validate_generated_queries(
            result=normalized,
            original_question=active_question,
            previous_questions=previous_questions,
            dataset=dataset_name,
        )
        normalized_result = normalized.to_dict()
        generated_queries = [item.to_dict() for item in normalized.questions]

    return {
        "generator": {
            "name": "curiosity_query_generator",
            "version": "G1A1_QUERY_ONLY_INSIDE_OUT",
        },
        "dataset": dataset_name,
        "prompt_version": curiosity_prompt_version(dataset_name),
        "raw_result": raw_result,
        "normalized_result": normalized_result,
        "result": normalized_result,
        "generated_queries": generated_queries,
        "normalization_actions": actions,
        "raw_validation_errors": raw_errors,
        "valid": result is not None and not errors,
        "json_valid": result is not None,
        "generator_schema_valid": result is not None,
        "query_spec_semantic_valid": result is not None and not errors,
        "query_spec_contract_valid": result is not None and not errors,
        "validation_mode": "repair_then_diagnostic",
        "execution_allowed_despite_spec_error": False,
        "validation_errors": errors,
        "validation_warnings": warnings,
        "raw_text": json_repaired_text or initial_raw_text,
        "initial_raw_text": initial_raw_text,
        "json_repaired_text": json_repaired_text,
        "json_repair_attempted": json_repair_attempted,
        "parse_error": parse_error,
        "prompt": prompt,
        "input": {
            "original_question": original,
            "current_question": active_question,
            "target_question": original,
            "reasoning_question": active_question,
            "verified_facts": verified_facts,
            "previous_questions": previous_questions,
            "dataset": dataset_name,
        },
    }


def sanitize_generated_question(question: str) -> str:
    return " ".join(str(question or "").split()).strip()


def normalize_generated_queries(
    *,
    result: CuriosityQueryResult,
    original_question: str,
    dataset: str,
) -> tuple[CuriosityQueryResult, list[dict[str, Any]]]:
    """Clean model-control leakage without using another LLM call."""

    normalize_curiosity_dataset_name(dataset)
    actions: list[dict[str, Any]] = []
    retained: list[CuriosityQuerySpec] = []
    seen_questions: set[str] = set()
    for index, query_spec in enumerate(result.questions):
        sanitized = sanitize_generated_question(query_spec.question)
        if sanitized != query_spec.question:
            actions.append(
                {
                    "action": "QUERY_TEXT_SANITIZED",
                    "question_index": index,
                    "before": query_spec.question,
                    "after": sanitized,
                }
            )
        normalized_question = normalize_question_text(sanitized)
        if normalized_question in seen_questions:
            actions.append(
                {
                    "action": "DUPLICATE_GENERATED_QUERY_PRUNED",
                    "question_index": index,
                    "question": sanitized,
                    "reason": "Only the first identical retrieval query is retained",
                }
            )
            continue
        seen_questions.add(normalized_question)
        retained.append(
            CuriosityQuerySpec(
                question=sanitized,
                resolution_target=" ".join(query_spec.resolution_target.split()),
            )
        )

    return CuriosityQueryResult(retained), actions


def validate_generated_queries(
    *,
    result: CuriosityQueryResult,
    original_question: str,
    previous_questions: list[str],
    dataset: str = "hotpotqa",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dataset_name = normalize_curiosity_dataset_name(dataset)
    max_queries = _max_queries_for_question(dataset_name, original_question)
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    # An empty list is the Generator's explicit and only normal completion
    # signal: no unresolved information gap remains.
    if len(result.questions) > max_queries:
        errors.append(
            _diagnostic(
                "TOO_MANY_QUESTIONS",
                "questions",
                str(len(result.questions)),
                f"This question permits at most {max_queries} retrieval queries.",
            )
        )

    history = {
        normalize_question_text(item)
        for item in previous_questions
        if str(item or "").strip()
    }
    seen: set[str] = set()
    seen_targets: set[str] = set()
    question_texts = [item.question for item in result.questions]
    for index, query_spec in enumerate(result.questions):
        question = query_spec.question
        location = f"questions.{index}"
        if not question.rstrip().endswith(("?", "？")):
            errors.append(
                _diagnostic(
                    "QUESTION_MARK_MISSING",
                    location,
                    question,
                    "A retrieval query must end with a question mark.",
                )
            )
        if _JSON_FRAGMENT_PATTERN.search(question):
            errors.append(
                _diagnostic(
                    "JSON_FRAGMENT_INSIDE_QUESTION",
                    location,
                    question,
                    "A serialized JSON fragment leaked into the query string.",
                )
            )
        normalized = normalize_question_text(question)
        if normalized in history:
            errors.append(
                _diagnostic(
                    "QUESTION_ALREADY_ASKED",
                    location,
                    question,
                    "The retrieval query already appears in PREVIOUS QUESTIONS.",
                )
            )
        if normalized in seen:
            errors.append(
                _diagnostic(
                    "DUPLICATE_GENERATED_QUESTION",
                    location,
                    question,
                    "The same retrieval query was generated more than once.",
                )
            )
        seen.add(normalized)
        normalized_target = _normalize_resolution_key(query_spec.resolution_target)
        if normalized_target in seen_targets:
            errors.append(
                _diagnostic(
                    "DUPLICATE_RESOLUTION_TARGET",
                    location + ".resolution_target",
                    query_spec.resolution_target,
                    "Two queries in one round must resolve different targets.",
                )
            )
        seen_targets.add(normalized_target)
    if len(result.questions) > 1 and _queries_may_be_dependent(
        question_texts, original_question, dataset_name
    ):
        warnings.append(
            _diagnostic(
                "MULTIPLE_QUERIES_MAY_BE_DEPENDENT",
                "questions",
                "",
                "Multiple queries may represent different layers of one branch.",
            )
        )
    if (
        dataset_name == "2wikimultihopqa"
        and is_symmetric_question(original_question)
        and len(result.questions) != 2
    ):
        warnings.append(
            _diagnostic(
                "SYMMETRIC_BRANCH_MISSING",
                "questions",
                str(len(result.questions)),
                "A symmetric 2Wiki question should normally expose two branches.",
            )
        )
    return _dedupe_diagnostics(errors), _dedupe_diagnostics(warnings)


def is_symmetric_question(question: str) -> bool:
    return is_explicit_parallel_question("2wikimultihopqa", question)


def parse_query_generation(
    raw_text: str,
) -> tuple[CuriosityQueryResult | None, str | None]:
    try:
        return CuriosityQueryResult.from_dict(parse_json_object(raw_text)), None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def build_json_repair_prompt(prompt: str, raw_text: str) -> str:
    return (
        prompt
        + "\n\nThe previous response was not valid JSON matching the G1A1 "
        "QuerySpec schema. Repair JSON formatting and schema shape only. Each "
        "item must contain exactly: question and resolution_target. Do not "
        "alter the intended dependency edge. Do not add explanations. Return "
        "JSON only.\n\n"
        "Previous response:\n"
        + raw_text
    )


def normalize_question_text(text: str) -> str:
    folded = unicodedata.normalize("NFKC", str(text or "")).casefold()
    folded = folded.translate(
        str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-"})
    )
    return " ".join(re.findall(r"[\w]+", folded, flags=re.UNICODE))


def _normalize_resolution_key(text: str) -> str:
    tokens = normalize_question_text(text).split()
    if tokens and tokens[0] in {"the", "a", "an"}:
        tokens = tokens[1:]
    return " ".join(tokens)


def _max_queries_for_question(dataset: str, original_question: str) -> int:
    del original_question
    config = DATASET_GENERATOR_CONFIGS[dataset]
    return int(config["max_queries"])


def _queries_may_be_dependent(
    questions: list[str],
    original_question: str,
    dataset: str,
) -> bool:
    if is_explicit_parallel_question(dataset, original_question):
        return False
    deictic = re.compile(
        r"\b(?:that|this|former|latter|the person|the director|the author)\b",
        re.IGNORECASE,
    )
    return len(questions) > 1 and (
        any(deictic.search(item) for item in questions)
        or not is_symmetric_question(original_question)
    )


def _call_generator(
    llm_client: PromptClient,
    prompt: str,
    response_format: dict[str, Any] | None,
) -> str:
    try:
        signature = inspect.signature(llm_client)
        signature.bind(prompt, response_format)
    except (TypeError, ValueError):
        return llm_client(prompt)
    return llm_client(prompt, response_format)


def _require_exact_keys(value: dict[str, Any], expected: set[str]) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"schema keys mismatch: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _diagnostic(code: str, location: str, value: str, message: str) -> dict[str, Any]:
    return {"code": code, "location": location, "value": value, "message": message}


def _dedupe_diagnostics(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        key = json.dumps(item, ensure_ascii=False, sort_keys=True)
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result
