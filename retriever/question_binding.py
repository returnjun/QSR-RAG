from __future__ import annotations

import inspect
import json
import re
import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from .normal_rag import parse_json_object


PromptClient = Callable[..., str]


class RewritePolicy(str, Enum):
    HYBRID = "hybrid"
    MEMORY_ONLY = "memory_only"
    ENTITY_BINDING_ONLY = "entity_binding_only"
    CONTEXT_INJECTION_ONLY = "context_injection_only"

ANSWER_TYPES = [
    "PERSON",
    "LOCATION",
    "ORGANIZATION",
    "WORK",
    "DATE",
    "NUMBER",
    "BOOLEAN",
    "SET",
    "OTHER",
]
ADAPTER_REJECT_REASONS = {
    "UNKNOWN_ANSWER",
    "NO_SUPPORTING_EVIDENCE",
    "NOT_GROUNDED",
    "INVALID_ADAPTER_PAYLOAD",
    "INVALID_ANSWER_ITEMS",
    "TARGET_MISMATCH",
    "INVALID_NORMALIZER_PAYLOAD",
    "INVALID_LOCAL_ANSWER",
}

RESOLUTION_ADAPTER_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "resolution_adapter_v21",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["evidence_grounded", "answer_items"],
            "properties": {
                "evidence_grounded": {"type": "boolean"},
                "answer_items": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"type": "string", "minLength": 1},
                },
            },
        },
    },
}

RESOLUTION_REWRITE_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "resolution_driven_question_rewrite_v2_final_policies",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["update_success", "updated_question", "used_query_ids"],
            "properties": {
                "update_success": {"type": "boolean"},
                "updated_question": {"type": "string"},
                "used_query_ids": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                },
            },
        },
    },
}

RESOLUTION_ADAPTER_PROMPT = """You are a Local Resolution Validator.

Your only job is to determine whether SUPPORTING EVIDENCE directly supports
NORMAL RAG ANSWER as the answer to LOCAL QUESTION and the value of RESOLUTION
TARGET.

Do NOT decide whether LOCAL QUESTION was the ideal next multi-hop question.
Do NOT judge whether another intermediate should have been resolved first.
Do NOT decide how this resolution will be used later.
Do NOT rewrite the reasoning question or generate another query.
Do NOT repair or replace NORMAL RAG ANSWER.

EVIDENCE GROUNDING

Set evidence_grounded=true only when the supplied evidence directly supports
NORMAL RAG ANSWER for the relation asked by LOCAL QUESTION / RESOLUTION TARGET.
Validate the relation, not merely the presence of the answer string. A related
but different relation is not sufficient.

Examples:
- Target: director of Film A; Answer: John Smith; Evidence: Film A was directed
  by John Smith. -> grounded
- Target: creator of X; Answer: John Smith; Evidence: John Smith acted in X.
  -> not grounded

ANSWER MEMBERS

Always preserve NORMAL RAG ANSWER. Split it only when it contains multiple
genuine answer members. Multiple supported members are NOT automatically
ambiguous.

Examples:
- "Lloyd Kaufman and Michael Herz" -> ["Lloyd Kaufman", "Michael Herz"]
- "Chuck Lorre and Steven Molaro" -> ["Chuck Lorre", "Steven Molaro"]
- "Fitzwilliam, New Hampshire" -> ["Fitzwilliam, New Hampshire"]
- "16 March 453" -> ["16 March 453"]
- "10 years" -> ["10 years"]

If the evidence does not support the answer, still copy or split NORMAL RAG
ANSWER into answer_items, but set evidence_grounded=false.

Return only the required JSON object.
"""

EVIDENCE_BLIND_ADAPTER_PROMPT = """You are a Local Resolution Consistency Validator.

You are given only a LOCAL QUESTION, its RESOLUTION TARGET, and the NORMAL RAG
ANSWER. Determine whether the answer is semantically consistent with the local
question and directly fills the requested resolution target.

You do not have access to retrieved evidence. Do not infer or claim that the
answer is factually supported by any document. Do not use outside knowledge to
repair, replace, or embellish the answer.

TARGET CONSISTENCY

Set target_consistent=true only when NORMAL RAG ANSWER has the right semantic
form and relation to resolve LOCAL QUESTION / RESOLUTION TARGET. Set it to
false when the answer is insufficient, answers a different relation, or does
not resolve the requested target.

ANSWER MEMBERS

Always preserve NORMAL RAG ANSWER. Split it only when it contains multiple
genuine answer members. Multiple plausible members are not automatically a
reason to reject.

Return only the required JSON object.
"""

NO_VERIFICATION_PARSE_PROMPT = """You are an Answer-Member Parser.

You are given a NORMAL RAG ANSWER. Your only task is to preserve that answer
and split it into answer_items only when it clearly contains multiple genuine
answer members.

Do not judge factual correctness.
Do not judge whether the answer matches a question or target.
Do not use retrieved evidence or outside knowledge.
Do not repair, replace, or embellish the answer.

Examples:
- "Lloyd Kaufman and Michael Herz" -> ["Lloyd Kaufman", "Michael Herz"]
- "Chuck Lorre and Steven Molaro" -> ["Chuck Lorre", "Steven Molaro"]
- "Fitzwilliam, New Hampshire" -> ["Fitzwilliam, New Hampshire"]
- "16 March 453" -> ["16 March 453"]
- "10 years" -> ["10 years"]

Return only the required JSON object.
"""

VALID_REWRITE_POLICIES = {
    item.value for item in RewritePolicy
}

EVIDENCE_BLIND_ADAPTER_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "resolution_target_consistency_v1",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["target_consistent", "answer_items"],
            "properties": {
                "target_consistent": {"type": "boolean"},
                "answer_items": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"type": "string", "minLength": 1},
                },
            },
        },
    },
}

NO_VERIFICATION_PARSE_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "resolution_answer_members_v1",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["answer_items"],
            "properties": {
                "answer_items": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"type": "string", "minLength": 1},
                },
            },
        },
    },
}

CORE_REWRITE_POLICY_PROMPT = """SYSTEM RULE — QUESTION STATE UPDATE

Update CURRENT REASONING QUESTION using only NEW VERIFIED RESOLUTIONS.
ORIGINAL TARGET QUESTION is read-only context.

1. Preserve the final answer space and the original final information need.
2. Update only CURRENT REASONING QUESTION.
3. Follow the selected REWRITE POLICY as the highest-priority rule.
4. Preserve unresolved outer relations, candidate identity, comparison
   direction, and logical operators.
5. Do not infer another hop or answer the final comparison, boolean, count,
   ranking, intersection, or selection operation.
6. A resolution may be used only for the same unresolved slot named by its
   RESOLUTION TARGET.
7. Use only the smallest safe subset of NEW VERIFIED RESOLUTIONS.
8. Preserve answer_exact verbatim whenever possible.

If no resolution can be used under the selected policy:
- update_success=false
- updated_question must equal CURRENT REASONING QUESTION
- used_query_ids=[]

A successful update must change CURRENT REASONING QUESTION, end with a
question mark, and surface every used resolution.
"""

HYBRID_REWRITE_POLICY_PROMPT = """REWRITE POLICY — QSR-RAG HYBRID

You may perform either of these operations:

1. ENTITY BINDING: replace an unresolved entity description with its verified
   answer entity.
2. CONTEXT INJECTION: preserve the reasoning question and prepend a concise
   verified fact when binding would damage candidates, comparison structure,
   or the final answer space.

Choose the minimum safe transformation. Never replace an already resolved
value with a new value and never consume the unresolved final relation.
Every candidate update must pass the shared safe-transition checks; preserve
the question type and answer space exactly.
"""

MEMORY_ONLY_REWRITE_POLICY_PROMPT = """REWRITE POLICY — MEMORY ONLY

Do not change CURRENT REASONING QUESTION. Verified resolutions remain external
memory for the Generator and final Reader. Always return KEEP.
"""

ENTITY_BINDING_ONLY_POLICY = """REWRITE POLICY — ENTITY BINDING ONLY

Your only allowed operation is replacing an unresolved entity description
with its verified answer entity. A resolution may be substituted only when
rewrite_operation_type is ENTITY_BINDING.

Allowed:
Before: Who is the spouse of the author of Book A?
Verified: author of Book A = John Smith
After: Who is the spouse of John Smith?

Forbidden:
- Prepending, appending, or inserting a verified fact as a premise.
- Replacing an already resolved value with another value.
- John Smith -> London.
- born in London -> 1950.
- country of birth -> United Kingdom.

If no ENTITY_BINDING resolution can safely replace an unresolved entity
description, return KEEP.

Do not bind comparison or selection descriptions such as younger/older,
winner, or multi-candidate targets. Do not replace descriptions anchored to
explicit answer candidates in an A-or-B / A-and-B choice question. Preserve
the question type exactly.
"""

CONTEXT_INJECTION_ONLY_POLICY = """REWRITE POLICY — CONTEXT INJECTION ONLY

The complete CURRENT REASONING QUESTION must remain unchanged. Do not replace,
delete, consume, or paraphrase any phrase in it. Only prepend selected verified
facts as concise declarative premises.

Allowed:
Film A is directed by John Smith.
Where was the director of Film A born?

Forbidden:
Where was John Smith born?
"""

REWRITE_POLICY_PROMPTS = {
    RewritePolicy.HYBRID: HYBRID_REWRITE_POLICY_PROMPT,
    RewritePolicy.MEMORY_ONLY: MEMORY_ONLY_REWRITE_POLICY_PROMPT,
    RewritePolicy.ENTITY_BINDING_ONLY: ENTITY_BINDING_ONLY_POLICY,
    RewritePolicy.CONTEXT_INJECTION_ONLY: CONTEXT_INJECTION_ONLY_POLICY,
}

ENTITY_BINDING_OPERATION = "ENTITY_BINDING"
CONTEXT_FACT_OPERATION = "CONTEXT_FACT"

COMPARISON_GUIDANCE = """PARALLEL / COMPARISON UPDATE

CURRENT QUESTION performs a final comparison, choice, same/different,
both, common, or other operation over original candidates.

Verified branch values are premises, not replacements for the final
candidates.

Preserve every original candidate and preserve the final operation.

Example:

CURRENT:
Which film has the younger director, Film A or Film B?

RESOLUTIONS:
director of Film A = John
director of Film B = Peter

CORRECT:
Film A is directed by John and Film B is directed by Peter.
Which film has the younger director, Film A or Film B?

INCORRECT:
Who is younger, John or Peter?

Do not re-ask already resolved branch facts.
"""

MULTIPLE_SAFE_SET_GUIDANCE = """MULTIPLE VALUE: SET-SAFE USE

The MULTIPLE resolution may be used because CURRENT QUESTION explicitly
operates over a set, alternatives, intersection, common property, or member
selection.

Preserve all answer members and preserve the final set operation.

Do not collapse the set to one arbitrary member.
Do not answer the final selection or intersection.
"""

MULTIPLE_UNSAFE_BRANCH_GUIDANCE = """MULTIPLE VALUE: BRANCHING RISK

The MULTIPLE resolution fills a slot that CURRENT QUESTION treats as one
single subject for a downstream relation.

Different answer members may require different downstream answers.

Do not combine them into one artificial subject.
Do not choose one member arbitrarily.

Prefer leaving this resolution unused.
If no other resolution can safely update CURRENT QUESTION, KEEP.
"""

HOTPOT_EMBEDDED_WH_GUIDANCE = """EMBEDDED FINAL ANSWER SLOT

CURRENT QUESTION contains its final answer slot inside or near the end
of the sentence, such as:

"... created by whom?"
"... at which theatre?"
"... owned by who?"

Treat that WH expression as the final answer slot.

If a resolution directly fills this slot, the current question is already
answered by that resolution. KEEP.

Do not replace the WH phrase with answer_exact.

INCORRECT:
Film A was directed by John?

Do not append answer_exact using:
"specifically", "namely", or equivalent answer-revealing wording.
"""

TWO_WIKI_KINSHIP_GUIDANCE = """KINSHIP SLOT SAFETY

Use only the kinship edge actually resolved by RESOLUTION TARGET.

Do not move answer_exact to a different family relation.

If the supplied relation does not resolve a genuine intermediate slot of
CURRENT QUESTION, leave it unused.

For example:

CURRENT:
What nationality is X's father?

RESOLUTION:
father of X = John

CORRECT:
What nationality is John?

INCORRECT:
What nationality is John's father?

A resolved father of X does NOT help rewrite:
Who is X's mother?
"""

MUSIQUE_NESTED_GUIDANCE = """NESTED DEPENDENCY UPDATE

CURRENT QUESTION contains nested unresolved relations.

Apply answer_exact only to the SAME inner slot named by RESOLUTION TARGET.

Consume exactly that resolved inner edge and preserve all outer unresolved
edges.

Do not attach answer_exact to a nearby outer noun.

Example:

CURRENT:
... book by the author who wrote a story featuring
the author of Work X ...

RESOLUTION:
author of Work X = Alice

CORRECT:
... book by the author who wrote a story featuring Alice ...

INCORRECT:
... book by Alice who wrote a story featuring
the author of Work X ...

Prefer replacing the resolved inner description instead of keeping both
answer_exact and the old description as an appositive.
"""


@dataclass(slots=True)
class ResolutionDecision:
    evidence_grounded: bool
    answer_items: list[str]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ResolutionDecision":
        _require_exact_keys(value, {"evidence_grounded", "answer_items"})
        if not isinstance(value.get("evidence_grounded"), bool):
            raise ValueError("evidence_grounded must be boolean")
        answer_items = value.get("answer_items")
        if (
            not isinstance(answer_items, list)
            or not answer_items
            or not all(isinstance(item, str) and item.strip() for item in answer_items)
        ):
            raise ValueError("answer_items must be a non-empty string array")
        return cls(
            evidence_grounded=value["evidence_grounded"],
            answer_items=[item.strip() for item in answer_items],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_grounded": self.evidence_grounded,
            "answer_items": list(self.answer_items),
        }


@dataclass(slots=True)
class BlindResolutionDecision:
    target_consistent: bool
    answer_items: list[str]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BlindResolutionDecision":
        _require_exact_keys(value, {"target_consistent", "answer_items"})
        if not isinstance(value.get("target_consistent"), bool):
            raise ValueError("target_consistent must be boolean")
        return cls(
            target_consistent=value["target_consistent"],
            answer_items=_validated_answer_items(value.get("answer_items")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_consistent": self.target_consistent,
            "answer_items": list(self.answer_items),
        }


@dataclass(slots=True)
class ParseOnlyDecision:
    answer_items: list[str]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ParseOnlyDecision":
        _require_exact_keys(value, {"answer_items"})
        return cls(answer_items=_validated_answer_items(value.get("answer_items")))

    def to_dict(self) -> dict[str, Any]:
        return {"answer_items": list(self.answer_items)}


def _validated_answer_items(value: Any) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item.strip() for item in value)
    ):
        raise ValueError("answer_items must be a non-empty string array")
    return [item.strip() for item in value]


@dataclass(slots=True)
class VerifiedResolution:
    query_id: str
    question_version: int
    local_question: str
    resolution_target: str
    answer_exact: str
    answer_items: list[str]
    cardinality: str
    item_count: int
    answer_type: str
    rewrite_operation_type: str
    round: int
    branch: int
    supporting_evidence_ids: list[str]
    verification_policy: str
    verification_status: str
    evidence_visible: bool | None
    adapter_decision: str
    accepted: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "question_version": self.question_version,
            "local_question": self.local_question,
            "resolution_target": self.resolution_target,
            "answer_exact": self.answer_exact,
            "answer_items": list(self.answer_items),
            "cardinality": self.cardinality,
            "item_count": self.item_count,
            "answer_type": self.answer_type,
            "rewrite_operation_type": self.rewrite_operation_type,
            "round": self.round,
            "branch": self.branch,
            "supporting_evidence_ids": list(self.supporting_evidence_ids),
            "verification_policy": self.verification_policy,
            "verification_status": self.verification_status,
            "evidence_visible": self.evidence_visible,
            "adapter_decision": self.adapter_decision,
            "accepted": self.accepted,
        }


def build_verified_resolution(
    *,
    query_id: str,
    question_version: int,
    local_question: str,
    resolution_target: str,
    adapted: dict[str, Any],
    round_index: int,
    branch_index: int,
    supporting_evidence_ids: list[str],
    current_question: str | None = None,
) -> dict[str, Any]:
    answer_items = [
        str(item).strip()
        for item in (adapted.get("answer_items") or [])
        if str(item).strip()
    ]
    resolution_target = str(resolution_target or "").strip()
    answer_type = str(adapted.get("answer_type") or "OTHER")
    verification_policy = str(
        adapted.get("verification_policy") or "evidence_grounded"
    )
    verification_status = {
        "evidence_grounded": "VERIFIED_EVIDENCE",
        "evidence_blind": "VERIFIED_BLIND",
        "no_verification": "UNVERIFIED",
    }.get(verification_policy, "UNKNOWN")
    return VerifiedResolution(
        query_id=str(query_id or "").strip(),
        question_version=int(question_version),
        local_question=str(local_question or "").strip(),
        resolution_target=resolution_target,
        answer_exact=str(adapted.get("answer_exact") or "").strip(),
        answer_items=answer_items,
        cardinality=str(adapted.get("cardinality") or "ONE"),
        item_count=len(answer_items),
        answer_type=answer_type,
        rewrite_operation_type=classify_rewrite_operation_type(
            current_question=str(current_question or ""),
            resolution_target=resolution_target,
            answer_type=answer_type,
            answer_exact=str(adapted.get("answer_exact") or "").strip(),
        ),
        round=int(round_index),
        branch=int(branch_index),
        supporting_evidence_ids=list(supporting_evidence_ids),
        verification_policy=verification_policy,
        verification_status=verification_status,
        evidence_visible=adapted.get("evidence_visible"),
        adapter_decision=str(adapted.get("adapter_decision") or "PASS"),
        accepted=True,
    ).to_dict()


def render_verified_resolution(fact: dict[str, Any]) -> str:
    text = (
        f"{str(fact.get('resolution_target') or '').strip()} = "
        f"{str(fact.get('answer_exact') or '').strip()}"
    ).strip()
    item_count = int(fact.get("item_count") or 0)
    return text + (f"; item_count = {item_count}" if item_count > 1 else "")


_ENTITY_BINDING_ANSWER_TYPES = {"PERSON", "LOCATION", "ORGANIZATION", "WORK"}
_NON_ENTITY_VALUE_TARGET = re.compile(
    r"\b(?:country\s+of\s+birth|nationality|citizenship|"
    r"birth\s+date|date\s+of\s+birth|death\s+date|date\s+of\s+death|"
    r"year|date|time|age|population|duration|distance|number|count|"
    r"amount|height|weight|temperature)\b",
    flags=re.IGNORECASE,
)
_ENTITY_DESCRIPTION_RELATION = re.compile(
    r"(?:\b(?:of|by|who|that|which|whose)\b|(?:'s|’s)\b)",
    flags=re.IGNORECASE,
)
_UNSAFE_ENTITY_BINDING_TARGET = re.compile(
    r"\b(?:younger|older|larger|smaller|winner|which|compare|comparison|"
    r"choice\s+candidate|candidate|earlier|later|first|last|more|less|fewer)\b",
    flags=re.IGNORECASE,
)
_COORDINATED_ENTITY_TARGET = re.compile(r"\b(?:and|or)\b", flags=re.IGNORECASE)


def is_safe_entity_binding(
    current_question: str,
    resolution_target: str,
    answer_exact: str,
) -> bool:
    """Return whether a resolution is a safe hidden-entity substitution.

    The target must be one unresolved relational entity description in the
    current question.  Comparison/selection targets and descriptions anchored
    to an explicit answer candidate are deliberately routed to context
    injection instead of substitution.
    """

    question = str(current_question or "").strip()
    target = " ".join(str(resolution_target or "").split())
    answer = " ".join(str(answer_exact or "").split())
    if not question or not target:
        return False
    if _UNSAFE_ENTITY_BINDING_TARGET.search(target):
        return False
    if _COORDINATED_ENTITY_TARGET.search(target):
        return False
    if not _ENTITY_DESCRIPTION_RELATION.search(target):
        return False
    anchor = _entity_description_anchor(target)
    if not anchor or _normalized_text(anchor) not in _normalized_text(question):
        return False
    if contains_answer_candidate(question, target):
        return False
    if answer and contains_answer_candidate(question, answer):
        return False
    return True


def _entity_description_anchor(resolution_target: str) -> str:
    target = " ".join(str(resolution_target or "").split()).strip(" ?")
    of_match = re.search(r"\bof\s+(.+)$", target, flags=re.IGNORECASE)
    if of_match:
        return of_match.group(1).strip()
    possessive_match = re.match(r"(.+?)(?:'s|’s)\s+\S+", target)
    if possessive_match:
        return possessive_match.group(1).strip()
    relative_match = re.search(
        r"\b(?:who|that|which|whose)\b\s+(.+)$", target, flags=re.IGNORECASE
    )
    return relative_match.group(1).strip() if relative_match else ""


def contains_answer_candidate(question: str, resolution_answer: str) -> bool:
    """Detect whether a target/value overlaps an explicit choice candidate."""

    value = _normalized_text(resolution_answer)
    if not value:
        return False
    return any(
        _normalized_text(candidate) in value
        for candidate in _extract_answer_candidates(question)
        if _normalized_text(candidate)
    )


def _extract_answer_candidates(question: str) -> list[str]:
    text = str(question or "").strip()
    candidates = [
        match.group(2).strip()
        for match in re.finditer(r"(['\"])(.+?)\1", text)
        if match.group(2).strip()
    ]
    final_clause = _final_interrogative_clause(text).rstrip("?？").strip()
    choice_scope = ""
    if "," in final_clause:
        choice_scope = final_clause.rsplit(",", 1)[-1].strip()
    else:
        between = re.search(r"\bbetween\s+(.+)$", final_clause, re.IGNORECASE)
        if between:
            choice_scope = between.group(1).strip()
    if choice_scope:
        pair = re.fullmatch(
            r"(.{1,120}?)\s+(?:or|and)\s+(.{1,120})",
            choice_scope,
            flags=re.IGNORECASE,
        )
        if pair:
            candidates.extend([pair.group(1).strip(), pair.group(2).strip()])
    return list(dict.fromkeys(item for item in candidates if item))


def classify_rewrite_operation_type(
    *,
    current_question: str,
    resolution_target: str,
    answer_type: str,
    answer_exact: str = "",
) -> str:
    """Conservatively identify resolutions eligible for entity substitution."""

    target = " ".join(str(resolution_target or "").split())
    normalized_type = str(answer_type or "").strip().upper()
    if normalized_type not in _ENTITY_BINDING_ANSWER_TYPES:
        return CONTEXT_FACT_OPERATION
    if not target or _NON_ENTITY_VALUE_TARGET.search(target):
        return CONTEXT_FACT_OPERATION
    if not is_safe_entity_binding(
        current_question=current_question,
        resolution_target=target,
        answer_exact=answer_exact,
    ):
        return CONTEXT_FACT_OPERATION
    return ENTITY_BINDING_OPERATION


def adapt_answer_to_resolution(
    *,
    local_question: str,
    resolution_target: str,
    rag_answer: str,
    reader_answer_type: str,
    supporting_evidence: list[dict[str, Any]],
    llm_client: PromptClient,
    response_format: dict[str, Any] | None = None,
    evidence_visible: bool = True,
    verification_mode: str | None = None,
) -> dict[str, Any]:
    """Normalize and conditionally verify one local answer.

    The three modes form a nested information design:
    no_verification parses answer members only; evidence_blind additionally
    checks target consistency; full additionally checks retrieved evidence.
    """

    mode = str(
        verification_mode
        or ("full" if evidence_visible else "evidence_blind")
    ).strip().casefold()
    if mode not in {"full", "evidence_blind", "no_verification"}:
        raise ValueError(f"unsupported verification_mode: {verification_mode!r}")

    selected_schema = response_format or {
        "full": RESOLUTION_ADAPTER_SCHEMA,
        "evidence_blind": EVIDENCE_BLIND_ADAPTER_SCHEMA,
        "no_verification": NO_VERIFICATION_PARSE_SCHEMA,
    }[mode]
    answer_exact = str(rag_answer or "").strip()
    if mode == "no_verification":
        adapter_input = {
            "candidate_answer": answer_exact,
            "reader_answer_type": str(reader_answer_type or "").strip(),
            "verification_mode": mode,
        }
    else:
        adapter_input = _resolution_adapter_input(
            local_question=local_question,
            resolution_target=resolution_target,
            rag_answer=answer_exact,
            reader_answer_type=reader_answer_type,
            supporting_evidence=supporting_evidence,
            evidence_visible=(mode == "full"),
        )
        adapter_input["verification_mode"] = mode

    def finish(result: dict[str, Any]) -> dict[str, Any]:
        value = dict(result)
        semantic_verifier_used = mode != "no_verification"
        value.update(
            {
                "verification_policy": (
                    "evidence_grounded" if mode == "full" else mode
                ),
                "verification_criterion": {
                    "full": "evidence_support_and_target_alignment",
                    "evidence_blind": "target_alignment_only",
                    "no_verification": "answer_parsing_only",
                }[mode],
                "evidence_visible": (
                    True if mode == "full" else False if mode == "evidence_blind" else None
                ),
                "adapter_used": semantic_verifier_used,
                "normalizer_used": bool(value.get("llm_called")),
                "adapter_decision": (
                    "PASS" if value.get("usable") else "REJECT"
                ) if semantic_verifier_used else "NOT_USED",
                "verification_passed": (
                    bool(value.get("usable")) if semantic_verifier_used else None
                ),
                "adapter_input": adapter_input,
            }
        )
        if mode != "full":
            value["evidence_grounded"] = None
        if mode == "no_verification":
            value["target_consistent"] = None
        return value

    if not _local_answer_is_syntactically_valid(rag_answer):
        return finish(_resolution_reject_result(
            answer_exact=answer_exact,
            reader_answer_type=reader_answer_type,
            resolution_target=resolution_target,
            reject_reason=(
                "UNKNOWN_ANSWER" if isinstance(rag_answer, str) else "INVALID_LOCAL_ANSWER"
            ),
            llm_called=False,
        ))
    if mode == "full" and not supporting_evidence:
        return finish(_resolution_reject_result(
            answer_exact=answer_exact,
            reader_answer_type=reader_answer_type,
            resolution_target=resolution_target,
            reject_reason="NO_SUPPORTING_EVIDENCE",
            llm_called=False,
        ))

    if mode == "no_verification":
        prompt = build_no_verification_parse_prompt(
            rag_answer=answer_exact,
            reader_answer_type=reader_answer_type,
        )
    else:
        prompt = build_resolution_adapter_prompt(
            local_question=local_question,
            resolution_target=resolution_target,
            rag_answer=answer_exact,
            reader_answer_type=reader_answer_type,
            supporting_evidence=(supporting_evidence if mode == "full" else []),
            evidence_visible=(mode == "full"),
        )
    raw_text = _call_client(llm_client, prompt, selected_schema)
    try:
        payload = parse_json_object(raw_text)
        if mode == "full":
            decision: ResolutionDecision | BlindResolutionDecision | ParseOnlyDecision
            decision = ResolutionDecision.from_dict(payload)
        elif mode == "evidence_blind":
            decision = BlindResolutionDecision.from_dict(payload)
        else:
            decision = ParseOnlyDecision.from_dict(payload)
    except Exception as exc:  # noqa: BLE001
        return finish(_resolution_reject_result(
            answer_exact=answer_exact,
            reader_answer_type=reader_answer_type,
            resolution_target=resolution_target,
            reject_reason=(
                "INVALID_NORMALIZER_PAYLOAD"
                if mode == "no_verification"
                else "INVALID_ADAPTER_PAYLOAD"
            ),
            validation_errors=[
                "INVALID_NORMALIZER_PAYLOAD"
                if mode == "no_verification"
                else "INVALID_ADAPTER_PAYLOAD"
            ],
            raw_text=raw_text,
            prompt=prompt,
            parse_error=str(exc),
            llm_called=True,
        ))

    validation_errors = validate_answer_items(
        decision.answer_items, rag_answer=answer_exact
    )
    if validation_errors:
        return finish(_resolution_reject_result(
            answer_exact=answer_exact,
            reader_answer_type=reader_answer_type,
            resolution_target=resolution_target,
            reject_reason="INVALID_ANSWER_ITEMS",
            validation_errors=validation_errors,
            raw_text=raw_text,
            prompt=prompt,
            llm_called=True,
            model_decision=decision.to_dict(),
        ))
    if mode == "full" and isinstance(decision, ResolutionDecision) and not decision.evidence_grounded:
        return finish(_resolution_reject_result(
            answer_exact=answer_exact,
            reader_answer_type=reader_answer_type,
            resolution_target=resolution_target,
            reject_reason="NOT_GROUNDED",
            raw_text=raw_text,
            prompt=prompt,
            llm_called=True,
            model_decision=decision.to_dict(),
        ))
    if (
        mode == "evidence_blind"
        and isinstance(decision, BlindResolutionDecision)
        and not decision.target_consistent
    ):
        rejected = _resolution_reject_result(
            answer_exact=answer_exact,
            reader_answer_type=reader_answer_type,
            resolution_target=resolution_target,
            reject_reason="TARGET_MISMATCH",
            raw_text=raw_text,
            prompt=prompt,
            llm_called=True,
            model_decision=decision.to_dict(),
        )
        rejected["target_consistent"] = False
        return finish(rejected)

    answer_items = list(decision.answer_items)
    cardinality = "ONE" if len(answer_items) == 1 else "MULTIPLE"
    return finish({
        "usable": True,
        "evidence_grounded": True if mode == "full" else None,
        "target_consistent": True if mode in {"full", "evidence_blind"} else None,
        "answer_exact": answer_exact,
        "answer_items": answer_items,
        "cardinality": cardinality,
        "item_count": len(answer_items),
        "answer_type": normalize_reader_answer_type(reader_answer_type),
        "resolution_target": str(resolution_target or "").strip(),
        "canonical_fact": f"{str(resolution_target or '').strip()} = {answer_exact}",
        "reject_reason": "NONE",
        "validation_errors": [],
        "raw_text": raw_text,
        "parse_error": None,
        "prompt": prompt,
        "llm_called": True,
        "model_decision": decision.to_dict(),
    })


def build_resolution_adapter_prompt(
    *,
    local_question: str,
    resolution_target: str,
    rag_answer: str,
    reader_answer_type: str,
    supporting_evidence: list[dict[str, Any]],
    evidence_visible: bool = True,
) -> str:
    sections = [
        (
            RESOLUTION_ADAPTER_PROMPT.strip()
            if evidence_visible
            else EVIDENCE_BLIND_ADAPTER_PROMPT.strip()
        ),
        "RESOLUTION TARGET:\n" + str(resolution_target or "").strip(),
        "LOCAL QUESTION:\n" + local_question.strip(),
        "NORMAL RAG ANSWER:\n" + str(rag_answer or "").strip(),
        "READER ANSWER TYPE:\n" + str(reader_answer_type or "").strip(),
    ]
    if evidence_visible:
        sections.append(
            "SUPPORTING EVIDENCE:\n"
            + json.dumps(supporting_evidence, ensure_ascii=False, indent=2)
        )
    return "\n\n".join(sections)


def build_no_verification_parse_prompt(
    *,
    rag_answer: str,
    reader_answer_type: str,
) -> str:
    """Build the parsing-only prompt without semantic-verification inputs."""

    return "\n\n".join(
        [
            NO_VERIFICATION_PARSE_PROMPT.strip(),
            "NORMAL RAG ANSWER:\n" + str(rag_answer or "").strip(),
            "READER ANSWER TYPE:\n" + str(reader_answer_type or "").strip(),
        ]
    )


def _resolution_adapter_input(
    *,
    local_question: str,
    resolution_target: str,
    rag_answer: str,
    reader_answer_type: str,
    supporting_evidence: list[dict[str, Any]],
    evidence_visible: bool,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "local_question": str(local_question or "").strip(),
        "resolution_target": str(resolution_target or "").strip(),
        "candidate_answer": str(rag_answer or "").strip(),
        "reader_answer_type": str(reader_answer_type or "").strip(),
        "evidence_visible": bool(evidence_visible),
    }
    if evidence_visible:
        value["retrieved_evidence"] = [dict(item) for item in supporting_evidence]
    return value


def rewrite_question_with_resolutions(
    *,
    dataset: str,
    original_question: str,
    current_question: str,
    question_version: int,
    new_resolutions: list[dict[str, Any]],
    llm_client: PromptClient,
    response_format: dict[str, Any] | None = RESOLUTION_REWRITE_SCHEMA,
    rewrite_policy: RewritePolicy | str = RewritePolicy.HYBRID,
) -> dict[str, Any]:
    policy = normalize_rewrite_policy(rewrite_policy)
    before = str(current_question or "").strip()
    prepared_resolutions = _prepare_rewrite_resolutions(
        current_question=before,
        resolutions=new_resolutions,
    )

    def rewrite_result(**kwargs: Any) -> dict[str, Any]:
        proposed_after = str(kwargs.get("after") or before)
        result = _resolution_rewrite_result(**kwargs)
        result["policy"] = policy.value
        result["violations"] = list(result.get("validation_errors") or [])
        result["operation"] = _infer_rewrite_operation(
            policy=policy,
            before=before,
            proposed_after=proposed_after,
        )
        result["accepted"] = bool(result.get("update_success"))
        result["reason"] = _rewrite_audit_reason(result)
        old_type = classify_question_type(before)
        new_type = classify_question_type(proposed_after)
        result["answer_space_changed"] = bool(
            old_type and new_type and old_type != new_type
        )
        return result

    if policy is RewritePolicy.MEMORY_ONLY or not prepared_resolutions:
        return rewrite_result(
            success=False,
            before=before,
            after=before,
            used_query_ids=[],
            validation_errors=[],
            raw_text="",
            parse_error=None,
            prompt="",
            question_version=question_version,
            routing_tags=[],
            llm_called=False,
        )

    dataset_name = _normalized_dataset(dataset)
    binding_resolutions = [
        item
        for item in prepared_resolutions
        if item.get("rewrite_operation_type") == ENTITY_BINDING_OPERATION
    ]
    allow_deterministic_binding = policy in {
        RewritePolicy.HYBRID,
        RewritePolicy.ENTITY_BINDING_ONLY,
    }
    if policy is RewritePolicy.ENTITY_BINDING_ONLY and not binding_resolutions:
        return rewrite_result(
            success=False,
            before=before,
            after=before,
            used_query_ids=[],
            validation_errors=[],
            raw_text="",
            parse_error=None,
            prompt="",
            question_version=question_version,
            routing_tags=["NO_SAFE_BINDING"],
            llm_called=False,
        )
    if (
        allow_deterministic_binding
        and dataset_name == "2wikimultihopqa"
        and _requires_2wiki_step_keep(
            before, binding_resolutions
        )
    ):
        return rewrite_result(
            success=False,
            before=before,
            after=before,
            used_query_ids=[],
            validation_errors=[],
            raw_text="",
            parse_error=None,
            prompt="",
            question_version=question_version,
            routing_tags=["2WIKI_STEP_KEEP"],
            deterministic_reason="2WIKI_STEP_KEEP",
            llm_called=False,
        )

    if allow_deterministic_binding and dataset_name == "2wikimultihopqa":
        immediate = try_deterministic_2wiki_immediate_kinship(
            current_question=before,
            resolutions=binding_resolutions,
        )
        if immediate is not None:
            if immediate["action"] == "KEEP":
                return rewrite_result(
                    success=False,
                    before=before,
                    after=before,
                    used_query_ids=[],
                    validation_errors=[],
                    raw_text="",
                    parse_error=None,
                    prompt="",
                    question_version=question_version,
                    routing_tags=["2WIKI_IMMEDIATE_KINSHIP_KEEP"],
                    deterministic_reason=str(immediate["reason"]),
                    llm_called=False,
                )
            candidate = str(immediate["updated_question"])
            used_ids = list(immediate["used_query_ids"])
            validation_errors = validate_resolution_rewrite(
                dataset=dataset,
                old_question=before,
                new_question=candidate,
                resolutions=prepared_resolutions,
                used_query_ids=used_ids,
            )
            validation_errors.extend(
                validate_rewrite_policy(
                    policy=policy,
                    old_question=before,
                    new_question=candidate,
                    resolutions=prepared_resolutions,
                    used_query_ids=used_ids,
                )
            )
            return rewrite_result(
                success=not validation_errors,
                before=before,
                after=candidate,
                used_query_ids=used_ids,
                validation_errors=validation_errors,
                raw_text="",
                parse_error=None,
                prompt="",
                question_version=question_version,
                routing_tags=["2WIKI_DETERMINISTIC_IMMEDIATE_KINSHIP"],
                deterministic_reason=str(immediate["reason"]),
                llm_called=False,
            )

        deterministic = try_deterministic_2wiki_kinship(
            current_question=before,
            resolutions=binding_resolutions,
        )
        if deterministic is not None:
            candidate = str(deterministic["updated_question"])
            used_ids = list(deterministic["used_query_ids"])
            validation_errors = validate_resolution_rewrite(
                dataset=dataset,
                old_question=before,
                new_question=candidate,
                resolutions=prepared_resolutions,
                used_query_ids=used_ids,
            )
            validation_errors.extend(
                validate_rewrite_policy(
                    policy=policy,
                    old_question=before,
                    new_question=candidate,
                    resolutions=prepared_resolutions,
                    used_query_ids=used_ids,
                )
            )
            return rewrite_result(
                success=not validation_errors,
                before=before,
                after=candidate,
                used_query_ids=used_ids,
                validation_errors=validation_errors,
                raw_text="",
                parse_error=None,
                prompt="",
                question_version=question_version,
                routing_tags=["2WIKI_DETERMINISTIC_KINSHIP"],
                deterministic_reason=str(deterministic["reason"]),
                llm_called=False,
            )

    location_container = (
        try_deterministic_location_container(
            current_question=before,
            resolutions=binding_resolutions,
        )
        if allow_deterministic_binding
        else None
    )
    if location_container is not None:
        candidate = str(location_container["updated_question"])
        used_ids = list(location_container["used_query_ids"])
        validation_errors = validate_resolution_rewrite(
            dataset=dataset,
            old_question=before,
            new_question=candidate,
            resolutions=prepared_resolutions,
            used_query_ids=used_ids,
        )
        validation_errors.extend(
            validate_rewrite_policy(
                policy=policy,
                old_question=before,
                new_question=candidate,
                resolutions=prepared_resolutions,
                used_query_ids=used_ids,
            )
        )
        return rewrite_result(
            success=not validation_errors,
            before=before,
            after=candidate,
            used_query_ids=used_ids,
            validation_errors=validation_errors,
            raw_text="",
            parse_error=None,
            prompt="",
            question_version=question_version,
            routing_tags=["LOCATION_CONTAINER_DETERMINISTIC"],
            deterministic_reason=str(location_container["reason"]),
            llm_called=False,
        )

    prompt, routing_tags = build_resolution_rewrite_prompt(
        dataset=dataset,
        original_question=original_question,
        current_question=before,
        new_resolutions=prepared_resolutions,
        rewrite_policy=policy,
    )
    raw_text = _call_client(llm_client, prompt, response_format)
    try:
        parsed = parse_json_object(raw_text)
        _require_exact_keys(
            parsed, {"update_success", "updated_question", "used_query_ids"}
        )
        if not isinstance(parsed.get("update_success"), bool):
            raise ValueError("update_success must be boolean")
        if not isinstance(parsed.get("updated_question"), str):
            raise ValueError("updated_question must be a string")
        if not isinstance(parsed.get("used_query_ids"), list) or not all(
            isinstance(item, str) and item.strip()
            for item in parsed["used_query_ids"]
        ):
            raise ValueError("used_query_ids must be a string array")
        candidate = parsed["updated_question"].strip()
        used_ids = [str(item).strip() for item in parsed["used_query_ids"]]
    except Exception as exc:  # noqa: BLE001
        return rewrite_result(
            success=False,
            before=before,
            after=before,
            used_query_ids=[],
            validation_errors=["INVALID_REWRITE_PAYLOAD"],
            raw_text=raw_text,
            parse_error=str(exc),
            prompt=prompt,
            question_version=question_version,
            routing_tags=routing_tags,
        )

    if not parsed["update_success"]:
        keep_errors = []
        if candidate != before or used_ids:
            keep_errors.append("INVALID_KEEP_PAYLOAD")
        return rewrite_result(
            success=False,
            before=before,
            after=before,
            used_query_ids=[],
            validation_errors=keep_errors,
            raw_text=raw_text,
            parse_error=None,
            prompt=prompt,
            question_version=question_version,
            routing_tags=routing_tags,
        )

    validation_errors = validate_resolution_rewrite(
        dataset=dataset,
        old_question=before,
        new_question=candidate,
        resolutions=prepared_resolutions,
        used_query_ids=used_ids,
    )
    validation_errors.extend(
        validate_rewrite_policy(
            policy=policy,
            old_question=before,
            new_question=candidate,
            resolutions=prepared_resolutions,
            used_query_ids=used_ids,
        )
    )
    return rewrite_result(
        success=not validation_errors,
        before=before,
        after=candidate,
        used_query_ids=used_ids,
        validation_errors=validation_errors,
        raw_text=raw_text,
        parse_error=None,
        prompt=prompt,
        question_version=question_version,
        routing_tags=routing_tags,
    )


def build_resolution_rewrite_prompt(
    *,
    dataset: str,
    original_question: str,
    current_question: str,
    new_resolutions: list[dict[str, Any]],
    rewrite_policy: RewritePolicy | str = RewritePolicy.HYBRID,
) -> tuple[str, list[str]]:
    policy = normalize_rewrite_policy(rewrite_policy)
    prepared_resolutions = _prepare_rewrite_resolutions(
        current_question=current_question,
        resolutions=new_resolutions,
    )
    values = [
        {
            "query_id": item.get("query_id"),
            "local_question": item.get("local_question"),
            "resolution_target": item.get("resolution_target"),
            "answer_exact": item.get("answer_exact"),
            "answer_items": item.get("answer_items"),
            "cardinality": item.get("cardinality"),
            "item_count": item.get("item_count"),
            "answer_type": item.get("answer_type"),
            "rewrite_operation_type": item.get("rewrite_operation_type"),
        }
        for item in prepared_resolutions
    ]
    guidance_blocks, routing_tags = collect_rewrite_guidance(
        dataset=dataset,
        current_question=current_question,
        resolutions=prepared_resolutions,
    )
    sections = [
        CORE_REWRITE_POLICY_PROMPT.strip(),
        REWRITE_POLICY_PROMPTS[policy].strip(),
    ]
    if guidance_blocks:
        sections.append(
            "RELEVANT STRUCTURAL GUIDANCE (subordinate to the selected "
            "REWRITE POLICY; it never authorizes another operation):\n\n"
            + "\n\n".join(block.strip() for block in guidance_blocks)
        )
    sections.extend(
        [
            "ORIGINAL TARGET QUESTION:\n" + str(original_question or "").strip(),
            "CURRENT REASONING QUESTION:\n" + str(current_question or "").strip(),
            "NEW VERIFIED RESOLUTIONS:\n"
            + json.dumps(values, ensure_ascii=False, indent=2),
        ]
    )
    return "\n\n".join(sections), [
        f"REWRITE_POLICY_{policy.value.upper()}",
        *routing_tags,
    ]


def validate_rewrite_policy(
    *,
    policy: RewritePolicy | str,
    old_question: str,
    new_question: str,
    resolutions: list[dict[str, Any]] | None = None,
    used_query_ids: list[str] | None = None,
) -> list[str]:
    """Mechanically enforce isolation between question-update ablations."""

    active_policy = normalize_rewrite_policy(policy)
    errors: list[str] = []
    old_norm = _normalized_text(old_question)
    new_norm = _normalized_text(new_question)
    if active_policy is RewritePolicy.MEMORY_ONLY and old_norm != new_norm:
        errors.append("MEMORY_ONLY_QUESTION_CHANGED")
    if (
        active_policy is RewritePolicy.CONTEXT_INJECTION_ONLY
        and not new_norm.endswith(old_norm)
    ):
        errors.append("CONTEXT_INJECTION_ORIGINAL_QUESTION_NOT_PRESERVED")
    if active_policy is RewritePolicy.ENTITY_BINDING_ONLY:
        if old_norm in new_norm and old_norm != new_norm:
            errors.append("ENTITY_BINDING_FACT_INJECTION_DETECTED")
        if _adds_declarative_premise(old_question, new_question):
            errors.append("ENTITY_BINDING_FACT_INJECTION_DETECTED")
        resolution_by_id = {
            str(item.get("query_id") or ""): item
            for item in (resolutions or [])
        }
        for query_id in used_query_ids or []:
            fact = resolution_by_id.get(query_id)
            if fact is not None and (
                fact.get("rewrite_operation_type") != ENTITY_BINDING_OPERATION
                or not is_safe_entity_binding(
                    current_question=old_question,
                    resolution_target=str(fact.get("resolution_target") or ""),
                    answer_exact=str(fact.get("answer_exact") or ""),
                )
            ):
                errors.append(
                    f"ENTITY_BINDING_INELIGIBLE_RESOLUTION:{query_id}"
                )
    return _dedupe(errors)


def normalize_rewrite_policy(policy: RewritePolicy | str) -> RewritePolicy:
    if isinstance(policy, RewritePolicy):
        return policy
    try:
        return RewritePolicy(str(policy or "").strip())
    except ValueError as exc:
        raise ValueError(
            f"unsupported rewrite_policy {policy!r}; expected one of "
            f"{sorted(VALID_REWRITE_POLICIES)}"
        ) from exc


def _prepare_rewrite_resolutions(
    *,
    current_question: str,
    resolutions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for item in resolutions:
        value = dict(item)
        operation_type = classify_rewrite_operation_type(
            current_question=current_question,
            resolution_target=str(value.get("resolution_target") or ""),
            answer_type=str(value.get("answer_type") or "OTHER"),
            answer_exact=str(value.get("answer_exact") or ""),
        )
        value["rewrite_operation_type"] = operation_type
        prepared.append(value)
    return prepared


def _adds_declarative_premise(old_question: str, new_question: str) -> bool:
    old = str(old_question or "").strip()
    new = str(new_question or "").strip()
    if not old or not new:
        return False
    old_prefix_marks = old.rstrip("?？").count(".") + old.count("\n")
    new_prefix_marks = new.rstrip("?？").count(".") + new.count("\n")
    return new_prefix_marks > old_prefix_marks


def classify_question_type(question: str) -> str:
    """Return the final answer-space type of a reasoning question."""

    return _answer_space_signature(_final_interrogative_clause(question))


def validate_safe_transition(
    *, old_question: str, new_question: str
) -> list[str]:
    """Shared safety gate for Hybrid and all question-changing ablations."""

    errors: list[str] = []
    old_type = classify_question_type(old_question)
    new_type = classify_question_type(new_question)
    if old_type and new_type and old_type != new_type:
        errors.append(f"QUESTION_TYPE_CHANGED:{old_type}->{new_type}")
    errors.extend(_validate_answer_space_preserved(old_question, new_question))
    errors.extend(_validate_interrogative_slot_preserved(old_question, new_question))
    errors.extend(
        _validate_explicit_comparison_not_reversed(old_question, new_question)
    )
    return _dedupe(errors)


def validate_resolution_rewrite(
    *,
    dataset: str,
    old_question: str,
    new_question: str,
    resolutions: list[dict[str, Any]],
    used_query_ids: list[str],
) -> list[str]:
    """Block only high-confidence structural or semantic-preservation failures.

    This validator is a disaster guard, not a second semantic judge.  It must
    not decide whether the rewrite is the optimal reasoning update.
    """

    errors: list[str] = []
    if not str(new_question or "").strip():
        errors.append("EMPTY_UPDATE")
    if not str(new_question or "").rstrip().endswith(("?", "？")):
        errors.append("UPDATE_NOT_A_QUESTION")
    if _normalized_text(new_question) == _normalized_text(old_question):
        errors.append("UPDATE_NOT_CHANGED")
    if not used_query_ids:
        errors.append("NO_RESOLUTION_USED")
    if len(set(used_query_ids)) != len(used_query_ids):
        errors.append("DUPLICATE_USED_QUERY_ID")

    resolution_by_id = {
        str(item.get("query_id") or ""): item for item in resolutions
    }
    for query_id in used_query_ids:
        if query_id not in resolution_by_id:
            errors.append(f"INVALID_USED_QUERY_ID:{query_id}")

    errors.extend(
        validate_safe_transition(
            old_question=old_question,
            new_question=new_question,
        )
    )
    used_resolutions = [
        resolution_by_id[query_id]
        for query_id in used_query_ids
        if query_id in resolution_by_id
    ]
    if _looks_like_direct_answer_reask(
        old_question=old_question,
        new_question=new_question,
        resolutions=used_resolutions,
    ):
        errors.append("DIRECT_ANSWER_REASK")
    for fact in used_resolutions:
        if _classify_multiple_usage(old_question, fact) == MULTIPLE_UNSAFE_BRANCH:
            errors.append(
                f"MULTIPLE_UNSAFE_DOWNSTREAM:{fact.get('query_id') or ''}"
            )
    if _normalized_dataset(dataset) == "hotpotqa" and _looks_like_explicit_answer_leak(
        new_question=new_question, resolutions=used_resolutions
    ):
        errors.append("DIRECT_ANSWER_LEAK")
    if _normalized_dataset(dataset) == "hotpotqa":
        errors.extend(
            _validate_hotpot_embedded_wh_slot(
                old_question=old_question,
                new_question=new_question,
                resolutions=used_resolutions,
            )
        )

    for query_id in used_query_ids:
        fact = resolution_by_id.get(query_id)
        if fact is not None and not _resolution_value_appears(fact, new_question):
            errors.append(f"RESOLUTION_VALUE_NOT_SURFACED:{query_id}")
    return _dedupe(errors)


def _resolution_rewrite_result(
    *,
    success: bool,
    before: str,
    after: str,
    used_query_ids: list[str],
    validation_errors: list[str],
    raw_text: str,
    parse_error: str | None,
    prompt: str,
    question_version: int,
    routing_tags: list[str],
    deterministic_reason: str | None = None,
    llm_called: bool = True,
) -> dict[str, Any]:
    changed = _normalized_text(before) != _normalized_text(after)
    accepted = bool(success and changed and not validation_errors)
    final_question = after if accepted else before
    return {
        "update_success": accepted,
        "before": before,
        "after": final_question,
        "updated_question": final_question,
        "question_changed": accepted,
        "used_query_ids": (
            list(dict.fromkeys(used_query_ids)) if accepted else []
        ),
        "validation_errors": list(validation_errors),
        "question_version_before": int(question_version),
        "question_version_after": int(question_version) + (1 if accepted else 0),
        "raw_text": raw_text,
        "parse_error": parse_error,
        "prompt": prompt,
        "routing_tags": list(dict.fromkeys(routing_tags)),
        "deterministic_reason": deterministic_reason,
        "llm_called": llm_called,
    }


def _infer_rewrite_operation(
    *,
    policy: RewritePolicy,
    before: str,
    proposed_after: str,
) -> str:
    if policy is RewritePolicy.MEMORY_ONLY:
        return "MEMORY_ONLY"
    if policy is RewritePolicy.CONTEXT_INJECTION_ONLY:
        return "CONTEXT_INJECTION"
    if policy is RewritePolicy.ENTITY_BINDING_ONLY:
        return ENTITY_BINDING_OPERATION
    before_norm = _normalized_text(before)
    after_norm = _normalized_text(proposed_after)
    if after_norm != before_norm and after_norm.endswith(before_norm):
        return "CONTEXT_INJECTION"
    if after_norm != before_norm:
        return ENTITY_BINDING_OPERATION
    return "KEEP"


def _rewrite_audit_reason(result: dict[str, Any]) -> str:
    if result.get("update_success"):
        operation = str(result.get("operation") or "")
        if operation == ENTITY_BINDING_OPERATION:
            return "hidden_entity_resolution"
        if operation == "CONTEXT_INJECTION":
            return "verified_fact_injection"
        return "safe_transition"
    violations = result.get("violations") or []
    if violations:
        return str(violations[0])
    tags = result.get("routing_tags") or []
    if "NO_SAFE_BINDING" in tags:
        return "no_safe_binding"
    if tags:
        return str(tags[0]).casefold()
    return "question_unchanged"


def validate_resolution_decision(
    decision: ResolutionDecision,
    *,
    rag_answer: str,
) -> list[str]:
    return validate_answer_items(decision.answer_items, rag_answer=rag_answer)


def validate_answer_items(
    answer_items: list[str],
    *,
    rag_answer: str,
) -> list[str]:
    """Validate lossless answer-member parsing for every treatment."""

    errors: list[str] = []
    normalized_rag = _normalize_answer_member(rag_answer)
    normalized_items = [_normalize_answer_member(item) for item in answer_items]
    if not normalized_items:
        return ["EMPTY_ANSWER_ITEMS"]
    if len(set(normalized_items)) != len(normalized_items):
        errors.append("DUPLICATE_ANSWER_ITEMS")
    for item in answer_items:
        normalized_item = _normalize_answer_member(item)
        if not normalized_item:
            errors.append("EMPTY_ANSWER_ITEM")
            continue
        if normalized_item not in normalized_rag:
            errors.append("ANSWER_ITEM_NOT_IN_RAG_ANSWER")
    return _dedupe(errors)


def normalize_reader_answer_type(value: str) -> str:
    normalized = _normalized_text(value).replace(" ", "_")
    return {
        "person": "PERSON",
        "people": "PERSON",
        "place": "LOCATION",
        "location": "LOCATION",
        "organization": "ORGANIZATION",
        "organisation": "ORGANIZATION",
        "work": "WORK",
        "date": "DATE",
        "number": "NUMBER",
        "quantity": "NUMBER",
        "boolean": "BOOLEAN",
        "bool": "BOOLEAN",
        "list": "SET",
        "set": "SET",
    }.get(normalized, "OTHER")


def _answer_is_unusable(answer: str) -> bool:
    normalized = " ".join(str(answer or "").strip().casefold().split())
    return not normalized or normalized in {
        "unknown",
        "无法确定",
        "不知道",
        "cannot determine",
        "not enough information",
        "insufficient information",
    }


def _local_answer_is_syntactically_valid(answer: Any) -> bool:
    if not isinstance(answer, str) or _answer_is_unusable(answer):
        return False
    normalized = " ".join(answer.strip().casefold().split())
    if normalized in {
        "error",
        "none",
        "null",
        "{}",
        "[]",
        "n/a",
        "nan",
    }:
        return False
    if normalized.startswith(("error:", "exception:", "traceback")):
        return False
    if answer.lstrip().startswith(("{", "[")):
        try:
            parsed = json.loads(answer)
        except (TypeError, ValueError, json.JSONDecodeError):
            return False
        if not isinstance(parsed, str) or not parsed.strip():
            return False
    return True


def _resolution_reject_result(
    *,
    answer_exact: str,
    reader_answer_type: str,
    resolution_target: str,
    reject_reason: str,
    validation_errors: list[str] | None = None,
    raw_text: str = "",
    prompt: str = "",
    parse_error: str | None = None,
    llm_called: bool,
    model_decision: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if reject_reason not in ADAPTER_REJECT_REASONS:
        raise ValueError(f"invalid Adapter reject reason: {reject_reason}")
    return {
        "usable": False,
        "evidence_grounded": False,
        "answer_exact": str(answer_exact or "").strip(),
        "answer_items": [],
        "cardinality": "",
        "item_count": 0,
        "answer_type": normalize_reader_answer_type(reader_answer_type),
        "resolution_target": str(resolution_target or "").strip(),
        "canonical_fact": "",
        "reject_reason": reject_reason,
        "validation_errors": list(validation_errors or []),
        "raw_text": raw_text,
        "parse_error": parse_error,
        "prompt": prompt,
        "llm_called": llm_called,
        "model_decision": model_decision,
    }


_OPPOSITE_COMPARISON = {
    "earlier": {"later"},
    "later": {"earlier"},
    "older": {"younger"},
    "younger": {"older"},
    "more": {"less", "fewer"},
    "less": {"more"},
    "fewer": {"more"},
    "same": {"different"},
    "different": {"same"},
    "first": {"last"},
    "last": {"first"},
}
_COMPARISON_MARKER = re.compile(
    r"\b(earlier|later|older|younger|more|less|fewer|same|different|first|last)\b"
)
_COMPARISON_EVENT_TAIL = re.compile(
    r"\b(?:born|released|published|established|founded|created|died)\s+"
    r"(first|last|earlier|later|older|younger)\s*\?$"
)
_INTERROGATIVE_START = re.compile(
    r"^(who|whom|what|which|where|when|how|"
    r"in|at|on|from|to|by|for|with|of|"
    r"are|is|was|were|do|does|did|"
    r"have|has|had|can|could|will|would)\b",
    flags=re.IGNORECASE,
)
_WH_MARKER = re.compile(
    r"\b(?:who|whom|what|which|where|when)\b"
    r"|\bhow\s+(?:many|much|long|old|far|often)\b",
    flags=re.IGNORECASE,
)
_FINAL_EMBEDDED_WH = re.compile(
    r"\b(?:by|as|with|for|from|to|at|in|on|of|above|under)\s+"
    r"(who|whom|what(?:\s+[\w-]+){0,4}|which(?:\s+[\w-]+){0,4})\s*\?$",
    flags=re.IGNORECASE,
)
_FINAL_WH_TAIL = re.compile(
    r"\b(who|whom|what(?:\s+[\w-]+){0,4}|which(?:\s+[\w-]+){0,4})\s*\?$",
    flags=re.IGNORECASE,
)
_ANSWER_HEADS = {
    "person",
    "actor",
    "actress",
    "singer",
    "director",
    "writer",
    "author",
    "star",
    "filmmaker",
    "politician",
    "film",
    "movie",
    "song",
    "album",
    "book",
    "novel",
    "game",
    "country",
    "city",
    "state",
    "province",
    "county",
    "company",
    "organization",
    "school",
    "university",
    "profession",
    "occupation",
    "award",
    "language",
}
_KINSHIP_COMPOSITION = {
    "paternal grandfather": ("father", "father"),
    "paternal grandmother": ("father", "mother"),
    "maternal grandfather": ("mother", "father"),
    "maternal grandmother": ("mother", "mother"),
    "father-in-law": ("spouse", "father"),
    "mother-in-law": ("spouse", "mother"),
}
_STEP_RELATIONS = {"stepmother", "stepfather"}
_IMMEDIATE_KINSHIP_RELATIONS = {"father", "mother", "husband", "wife", "spouse"}
_KINSHIP_TERMS = re.compile(
    r"\b(?:father|mother|parent|grandfather|grandmother|"
    r"spouse|husband|wife|uncle|aunt|sibling|brother|"
    r"sister|stepmother|stepfather|in-law)\b",
    flags=re.IGNORECASE,
)

MULTIPLE_NONE = "NONE"
MULTIPLE_SAFE_SET = "SAFE_SET"
MULTIPLE_UNSAFE_BRANCH = "UNSAFE_BRANCH"
MULTIPLE_AMBIGUOUS = "AMBIGUOUS"
_SAFE_SET_PATTERNS = (
    r"\bwhich\s+of\b",
    r"\bwhich\b[^?]{0,80}\bamong\b",
    r"\bwhat\b[^?]{0,100}\b(?:share|common)\b",
    r"\bin\s+common\b",
    r"\bboth\b",
    r"\bwhich\s+(?:member|actor|star|person|candidate)\b",
)
_SINGULAR_SLOT_HEADS = {
    "author",
    "creator",
    "winner",
    "director",
    "performer",
    "person",
    "team",
    "member",
    "composer",
    "singer",
    "actor",
    "player",
    "spouse",
    "father",
    "mother",
    "album",
    "show",
    "channel",
    "film",
    "city",
    "location",
}
_LOCATION_CONTAINERS = {
    "district",
    "county",
    "state",
    "province",
    "region",
    "country",
    "continent",
}
_LOCATION_TARGET_HEADS = {
    "birthplace",
    "birth place",
    "burial location",
    "headquarters location",
    "location",
    "city",
    "town",
    "place",
}


def _validate_answer_space_preserved(
    old_question: str, new_question: str
) -> list[str]:
    old_signature = _answer_space_signature(_final_interrogative_clause(old_question))
    new_signature = _answer_space_signature(_final_interrogative_clause(new_question))
    if old_signature and not new_signature:
        return [f"FINAL_ANSWER_SPACE_LOST:{old_signature}"]
    if (
        old_signature
        and new_signature
        and old_signature != new_signature
        and "attribute" not in {old_signature, new_signature}
    ):
        return [f"FINAL_ANSWER_SPACE_CHANGED:{old_signature}->{new_signature}"]
    return []


def _validate_interrogative_slot_preserved(
    old_question: str, new_question: str
) -> list[str]:
    old_has_wh = bool(_WH_MARKER.search(_final_interrogative_clause(old_question)))
    new_has_wh = bool(_WH_MARKER.search(_final_interrogative_clause(new_question)))
    if old_has_wh and not new_has_wh:
        return ["FINAL_INTERROGATIVE_SLOT_LOST"]
    return []


def _validate_explicit_comparison_not_reversed(
    old_question: str, new_question: str
) -> list[str]:
    old_operation = _comparison_operation(old_question)
    if not old_operation:
        return []
    new_operation = _comparison_operation(new_question)
    if new_operation in _OPPOSITE_COMPARISON.get(old_operation, set()):
        return [f"EXPLICIT_COMPARISON_REVERSED:{old_operation}->{new_operation}"]
    return []


def _comparison_operation(question: str) -> str:
    text = _normalized_text(_final_interrogative_clause(question))
    if not text:
        return ""
    comparison_scope = ""
    if "," in text:
        comparison_scope = text.split(",", 1)[0]
    elif " between " in text:
        comparison_scope = text.split(" between ", 1)[0]
    else:
        tail = _COMPARISON_EVENT_TAIL.search(text)
        if tail and " or " in text:
            return tail.group(1)
        if re.match(
            r"^(?:are|is|was|were|do|does|did|have|has|had|can|could|will|would)\b",
            text,
        ) and re.search(r"\b(?:and|or|both)\b", text):
            comparison_scope = text
    matches = list(_COMPARISON_MARKER.finditer(comparison_scope))
    return matches[-1].group(1) if matches else ""


def _looks_like_direct_answer_reask(
    *,
    old_question: str,
    new_question: str,
    resolutions: list[dict[str, Any]],
) -> bool:
    del old_question
    new_norm = _normalized_text(new_question).strip()
    for fact in resolutions:
        answer = _normalized_text(fact.get("answer_exact") or "").strip(" .!?？")
        if answer and new_norm in {f"who is {answer}?", f"who was {answer}?"}:
            return True
    return False


def _looks_like_explicit_answer_leak(
    *,
    new_question: str,
    resolutions: list[dict[str, Any]],
) -> bool:
    text = _normalized_text(new_question)
    for fact in resolutions:
        answer = _normalized_text(fact.get("answer_exact") or "").strip(" .!?？")
        if not answer:
            continue
        pattern = (
            r"\b(?:specifically|namely)\b[^?？]{0,60}" + re.escape(answer)
        )
        if re.search(pattern, text):
            return True
    return False


def _normalized_dataset(dataset: str) -> str:
    return str(dataset or "").strip().lower()


def _resolution_target_head(fact: dict[str, Any]) -> str:
    target = _normalized_text(fact.get("resolution_target") or "")
    head = target.split(" of ", 1)[0].strip() if " of " in target else target
    tokens = re.findall(r"[\w-]+", head)
    for token in tokens[:6]:
        if token in _SINGULAR_SLOT_HEADS:
            return token
    return tokens[-1] if len(tokens) == 1 else ""


def _classify_multiple_usage(question: str, fact: dict[str, Any]) -> str:
    if str(fact.get("cardinality") or "").upper() != "MULTIPLE":
        return MULTIPLE_NONE
    text = _normalized_text(question)
    if any(re.search(pattern, text) for pattern in _SAFE_SET_PATTERNS):
        return MULTIPLE_SAFE_SET
    head = _resolution_target_head(fact)
    if head in _SINGULAR_SLOT_HEADS and re.search(
        rf"\b(?:the|a|an)\s+(?:[\w-]+\s+){{0,4}}{re.escape(head)}\b", text
    ):
        return MULTIPLE_UNSAFE_BRANCH
    return MULTIPLE_AMBIGUOUS


def _is_parallel_operation(question: str) -> bool:
    text = _normalized_text(question)
    if _comparison_operation(question):
        return True
    return bool(
        re.search(r"\b(?:both|same|different|common|share)\b", text)
        or (re.search(r"\bwhich\b", text) and re.search(r"\bor\b", text))
    )


def _embedded_final_wh_slot(question: str) -> str:
    text = str(question or "").strip()
    if re.match(
        r"^(?:who|whom|what|which|where|when|how)\b",
        text,
        flags=re.IGNORECASE,
    ):
        return ""
    match = _FINAL_EMBEDDED_WH.search(text)
    if not match:
        match = _FINAL_WH_TAIL.search(text)
    return _normalized_text(match.group(1)) if match else ""


def _has_initial_wh_slot(question: str) -> bool:
    return bool(
        re.match(
            r"^(?:who|whom|what|which|where|when|how)\b",
            str(question or "").strip(),
            flags=re.IGNORECASE,
        )
    )


def _has_embedded_wh_slot(question: str) -> bool:
    return bool(_embedded_final_wh_slot(question))


def _validate_hotpot_embedded_wh_slot(
    *,
    old_question: str,
    new_question: str,
    resolutions: list[dict[str, Any]],
) -> list[str]:
    del resolutions
    old_slot = _embedded_final_wh_slot(old_question)
    if not old_slot:
        return []
    new_slot = _embedded_final_wh_slot(new_question)
    if new_slot or _has_initial_wh_slot(new_question):
        return []
    return ["EMBEDDED_FINAL_SLOT_FILLED"]


def _looks_like_nested_chain(question: str) -> bool:
    text = _normalized_text(question)
    patterns = (
        r"\bthe\b[^?]{0,80}\bwho\b",
        r"\bthe\b[^?]{0,80}\bthat\b",
        r"\bthe\b[^?]{0,80}\bwhich\b",
        r"\bof\s+the\b",
        r"\bcontaining\b",
        r"\blocated\b",
    )
    return sum(bool(re.search(pattern, text)) for pattern in patterns) >= 2


def _looks_like_kinship_case(
    question: str, resolutions: list[dict[str, Any]]
) -> bool:
    if not _KINSHIP_TERMS.search(question):
        return False
    return any(
        _KINSHIP_TERMS.search(str(fact.get("resolution_target") or ""))
        for fact in resolutions
    )


def collect_rewrite_guidance(
    *,
    dataset: str,
    current_question: str,
    resolutions: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    guidance: list[str] = []
    tags: list[str] = []
    if _is_parallel_operation(current_question):
        guidance.append(COMPARISON_GUIDANCE)
        tags.append("PARALLEL_OPERATION")
    multiple_modes = {
        _classify_multiple_usage(current_question, fact)
        for fact in resolutions
        if str(fact.get("cardinality") or "").upper() == "MULTIPLE"
    }
    if MULTIPLE_SAFE_SET in multiple_modes:
        guidance.append(MULTIPLE_SAFE_SET_GUIDANCE)
        tags.append("MULTIPLE_SAFE_SET")
    if MULTIPLE_UNSAFE_BRANCH in multiple_modes:
        guidance.append(MULTIPLE_UNSAFE_BRANCH_GUIDANCE)
        tags.append("MULTIPLE_UNSAFE_BRANCH")
    dataset_name = _normalized_dataset(dataset)
    if dataset_name == "hotpotqa" and _has_embedded_wh_slot(current_question):
        guidance.append(HOTPOT_EMBEDDED_WH_GUIDANCE)
        tags.append("HOTPOT_EMBEDDED_WH")
    if dataset_name == "2wikimultihopqa" and _looks_like_kinship_case(
        current_question, resolutions
    ):
        guidance.append(TWO_WIKI_KINSHIP_GUIDANCE)
        tags.append("2WIKI_KINSHIP")
    if dataset_name == "musique" and _looks_like_nested_chain(current_question):
        guidance.append(MUSIQUE_NESTED_GUIDANCE)
        tags.append("MUSIQUE_NESTED")
    return guidance, tags


def _requires_2wiki_step_keep(
    question: str, resolutions: list[dict[str, Any]]
) -> bool:
    text = _normalized_text(question)
    if not any(re.search(rf"\b{relation}\b", text) for relation in _STEP_RELATIONS):
        return False
    return any(
        re.search(
            r"\b(?:father|mother)\s+of\b",
            _normalized_text(fact.get("resolution_target") or ""),
        )
        for fact in resolutions
    )


def try_deterministic_2wiki_immediate_kinship(
    *,
    current_question: str,
    resolutions: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if len(resolutions) != 1:
        return None
    fact = resolutions[0]
    target = _normalized_text(fact.get("resolution_target") or "")
    target_match = re.match(
        r"^(father|mother|husband|wife|spouse)\s+of\s+(.+)$", target
    )
    answer = str(fact.get("answer_exact") or "").strip()
    if not target_match or not answer:
        return None
    resolved_relation = target_match.group(1)
    old = str(current_question or "").strip()

    direct_patterns = (
        rf"^who\s+is\s+(.+?)(?:'s|’s)\s+({_kinship_alternation()})\s*\?$",
        rf"^who\s+is\s+the\s+({_kinship_alternation()})\s+of\s+(.+?)\s*\?$",
        rf"^who\s+did\s+(.+?)\s+(?:marry|wed)\s*\?$",
    )
    direct_relation = ""
    for index, pattern in enumerate(direct_patterns):
        match = re.match(pattern, old, flags=re.IGNORECASE)
        if not match:
            continue
        if index == 0:
            direct_relation = _normalized_text(match.group(2))
        elif index == 1:
            direct_relation = _normalized_text(match.group(1))
        else:
            direct_relation = "spouse"
        break
    if direct_relation:
        if _kinship_relations_match(direct_relation, resolved_relation):
            return {
                "action": "KEEP",
                "updated_question": old,
                "used_query_ids": [],
                "reason": "DETERMINISTIC_2WIKI_IMMEDIATE_DIRECT_ANSWER_KEEP",
            }
        return {
            "action": "KEEP",
            "updated_question": old,
            "used_query_ids": [],
            "reason": "DETERMINISTIC_2WIKI_IMMEDIATE_RELATION_MISMATCH_KEEP",
        }

    subject = target_match.group(2).strip()
    possessive = rf"{re.escape(subject)}(?:'s|’s)\s+{re.escape(resolved_relation)}"
    of_form = rf"the\s+{re.escape(resolved_relation)}\s+of\s+{re.escape(subject)}"
    if not (re.search(possessive, _normalized_text(old)) or re.search(of_form, _normalized_text(old))):
        return None
    updated = re.sub(possessive, answer, old, count=1, flags=re.IGNORECASE)
    if updated == old:
        updated = re.sub(of_form, answer, old, count=1, flags=re.IGNORECASE)
    if updated == old:
        return None
    return {
        "action": "UPDATE",
        "updated_question": updated,
        "used_query_ids": [str(fact.get("query_id") or "")],
        "reason": "DETERMINISTIC_2WIKI_IMMEDIATE_OUTER_ATTRIBUTE",
    }


def _kinship_alternation() -> str:
    return "|".join(sorted(_IMMEDIATE_KINSHIP_RELATIONS))


def _kinship_relations_match(question_relation: str, target_relation: str) -> bool:
    if question_relation == target_relation:
        return True
    return "spouse" in {question_relation, target_relation} and {
        question_relation,
        target_relation,
    } <= {"husband", "wife", "spouse"}


def try_deterministic_location_container(
    *,
    current_question: str,
    resolutions: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if len(resolutions) != 1:
        return None
    fact = resolutions[0]
    target = _normalized_text(fact.get("resolution_target") or "")
    answer = str(fact.get("answer_exact") or "").strip()
    if not answer:
        return None
    relation = next(
        (
            head
            for head in sorted(_LOCATION_TARGET_HEADS, key=len, reverse=True)
            if target == head or target.startswith(head + " of ")
        ),
        "",
    )
    if not relation:
        return None
    old = str(current_question or "").strip()
    match = re.match(
        r"^in\s+(?:what|which)\s+"
        r"(district|county|state|province|region|country|continent)\s+"
        r"(?:was|were|is|are)\s+(.+?)\s+"
        r"(?:born|buried|located|headquartered|situated)\s*\?$",
        old,
        flags=re.IGNORECASE,
    )
    if not match or _normalized_text(match.group(1)) not in _LOCATION_CONTAINERS:
        return None
    event = _normalized_text(old)
    relation_matches = (
        relation in {"birthplace", "birth place"} and re.search(r"\bborn\s*\?$", event)
    ) or (
        relation == "burial location" and re.search(r"\bburied\s*\?$", event)
    ) or (
        relation == "headquarters location" and re.search(r"\bheadquartered\s*\?$", event)
    ) or (
        relation in {"location", "city", "town", "place"}
        and re.search(r"\b(?:located|situated)\s*\?$", event)
    )
    if not relation_matches:
        return None
    container = match.group(1)
    return {
        "updated_question": f"In which {container} is {answer} located?",
        "used_query_ids": [str(fact.get("query_id") or "")],
        "reason": "DETERMINISTIC_LOCATION_CONTAINER",
    }


def try_deterministic_2wiki_kinship(
    *,
    current_question: str,
    resolutions: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if len(resolutions) != 1:
        return None
    fact = resolutions[0]
    old = str(current_question or "").strip()
    old_norm = _normalized_text(old)
    target = _normalized_text(fact.get("resolution_target") or "")
    answer = str(fact.get("answer_exact") or "").strip()
    if not answer:
        return None
    for composite, (required_inner, remaining_outer) in _KINSHIP_COMPOSITION.items():
        if composite not in old_norm or not re.search(
            rf"\b{re.escape(required_inner)}\s+of\b", target
        ):
            continue
        possessive = re.match(
            rf"^who\s+is\s+(.+?)(?:'s|’s)\s+{re.escape(composite)}\s*\?$",
            old,
            flags=re.IGNORECASE,
        )
        of_form = re.match(
            rf"^who\s+is\s+the\s+{re.escape(composite)}\s+of\s+(.+?)\s*\?$",
            old,
            flags=re.IGNORECASE,
        )
        if possessive or of_form:
            return {
                "updated_question": f"Who is {answer}'s {remaining_outer}?",
                "used_query_ids": [str(fact.get("query_id") or "")],
                "reason": "DETERMINISTIC_2WIKI_KINSHIP",
            }
    return None


def _resolution_value_appears(fact: dict[str, Any], question: str) -> bool:
    if str(fact.get("answer_type") or "").upper() == "BOOLEAN":
        return True
    normalized = _normalized_text(question)
    answer_exact = _normalized_text(str(fact.get("answer_exact") or ""))
    if answer_exact and answer_exact in normalized:
        return True
    items = [_normalized_text(str(item)) for item in (fact.get("answer_items") or [])]
    if items and all(item in normalized for item in items):
        return True
    return False


def _final_interrogative_clause(question: str) -> str:
    text = str(question or "").strip()
    if not text:
        return ""
    q_end = max(text.rfind("?"), text.rfind("？"))
    if q_end >= 0:
        text = text[: q_end + 1]
    if _INTERROGATIVE_START.match(text):
        return text
    starters = re.compile(
        r"(?:^|[.;:。；：]\s+)"
        r"(who|whom|what|which|where|when|how|in|at|on|from|to|by|for|with|of|"
        r"are|is|was|were|do|does|did|"
        r"have|has|had|can|could|will|would)\b",
        flags=re.IGNORECASE,
    )
    matches = list(starters.finditer(text))
    return text[matches[-1].start(1) :].strip() if matches else text


def _text_before_final_interrogative(question: str) -> str:
    text = str(question or "").strip()
    final_clause = _final_interrogative_clause(text)
    start = text.rfind(final_clause) if final_clause else -1
    return text[:start].strip() if start > 0 else ""


def _answer_slot(question: str) -> str:
    text = _normalized_text(question).lstrip()
    if re.match(r"^how long\b", text):
        return "duration"
    if re.match(r"^how many\b", text):
        return "quantity"
    if re.match(r"^how\b", text):
        return "manner"
    if re.match(r"^(?:who|whom)\b", text):
        return "person"
    if re.match(r"^where\b", text):
        return "place"
    if re.match(r"^(?:when|what year|which year|in what year)\b", text):
        return "date"
    if re.match(r"^which\b", text) and re.search(r"\bor\b", text):
        return "choice"
    if re.match(
        r"^(?:are|is|was|were|do|does|did|have|has|had|can|could|will|would)\b",
        text,
    ):
        if re.search(r"\bor\b", text) and _comparison_operation(question):
            return "choice"
        return "boolean"
    if re.match(r"^(?:what|which)\b", text):
        return "attribute"
    return ""


def _answer_space_signature(question: str) -> str:
    text = _normalized_text(question).lstrip()
    match = re.match(r"^(?:what|which)\s+(.+)", text)
    if match:
        head = _wh_head(match.group(1))
        if head:
            return f"whnoun:{head}"
    return _answer_slot(question)


def _wh_head(text: str) -> str:
    tokens = re.findall(r"[\w-]+", text)
    predicate_boundaries = {
        "is",
        "was",
        "are",
        "were",
        "do",
        "does",
        "did",
        "has",
        "have",
        "had",
        "can",
        "could",
        "will",
        "would",
    }
    aliases = {"professionn": "profession"}
    for token in tokens[:5]:
        if token in predicate_boundaries:
            break
        normalized_token = aliases.get(token, token)
        if normalized_token in _ANSWER_HEADS:
            return normalized_token
    return ""


def _normalized_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value or "")).casefold().split())


def _normalize_answer_member(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(normalized.split())


def _call_client(
    client: PromptClient,
    prompt: str,
    response_format: dict[str, Any] | None,
) -> str:
    try:
        inspect.signature(client).bind(prompt, response_format)
    except (TypeError, ValueError):
        return client(prompt)
    return client(prompt, response_format)


def _require_exact_keys(value: dict[str, Any], expected: set[str]) -> None:
    if not isinstance(value, dict):
        raise ValueError("result must be an object")
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"schema keys mismatch: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))
