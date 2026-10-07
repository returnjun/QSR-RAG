from __future__ import annotations

import json
import re
import string
from dataclasses import asdict, dataclass, replace
from typing import Any, Callable

PromptClient = Callable[[str], str]


VALID_ANSWER_MODES = {
    "BOOLEAN",
    "CHOICE",
    "ENTITY",
    "ATTRIBUTE",
    "QUANTITY",
    "ORDINAL",
    "DATE",
    "LIST",
    "FREE_PHRASE",
}


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object from plain or fenced model output."""

    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            raise
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("Answer output must be a JSON object.")
    return parsed


@dataclass(slots=True)
class AnswerContract:
    mode: str
    answer_type: str
    answer_subtype: str
    expected_cardinality: int | None = None
    required_unit: str = ""
    allowed_answers: list[str] | None = None
    must_be_extractive: bool = True
    preserve_full_entity_name: bool = False
    final_relation: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["allowed_answers"] = list(self.allowed_answers or [])
        return value


NORMAL_RAG_PROMPT = """You are an evidence-grounded multi-hop question answering system.

Use only the supplied EVIDENCE.

The QUESTION is authoritative.

The ANSWER CONTRACT must never override the question.
Its `mode` and `answer_type` constrain the expected answer form.
Its `final_relation` is a semantic checksum derived from the question:
the answer must satisfy exactly that relation.

Never substitute a semantically related relation.
Examples:
- birth_place != nationality
- death_place != death_date
- workplace != work_country
- birth_country != nationality

Complete the fields in the exact order shown below. The final answer must be generated last.

Procedure:

1. Rewrite the main question as a precise unresolved information need in resolved_question.

2. Identify descriptive entities that must be resolved, such as:
   - the director of a film;
   - the instrument played by a person;
   - the venue that hosted an event;
   - the company an airline originally belonged to.

3. Record important bridge bindings supported by evidence.

4. Select the minimum evidence facts needed to answer.

5. Derive a candidate answer.

6. Check:
   - Does the candidate directly fill resolved_question?
   - Is it only an intermediate entity?
   - Is the direction of the requested relation correct?
   - For a comparison, were both values compared?
   - For a temporal question, was the relevant state at the requested time identified?
   - For a class question, is the answer the class rather than the instance?

7. Generate the final answer only after these checks.

Preserve the exact answer granularity required by the question.
Preserve full names, units, conjunctions, qualifiers, and all answer members supported by the evidence.

Return only valid JSON, using this exact field order:

{
  "resolved_question": "the precise final information need",
  "bridge_bindings": [
    {
      "mention": "descriptive phrase from the question",
      "value": "resolved entity",
      "evidence_id": "E1"
    }
  ],
  "evidence_facts": [
    {
      "evidence_id": "E1",
      "fact": "concise supported fact"
    }
  ],
  "derivation": "one concise derivation from the evidence facts",
  "candidate_answer": "candidate produced by the derivation",
  "answer_check": {
    "fills_requested_slot": true,
    "is_bridge_entity": false,
    "relation_direction_correct": true
  },
  "supporting_evidence_ids": ["E1", "E2"],
  "answer_type": "person | place | organization | work | date | number | boolean | category | list | other",
  "answer": "exact final answer"
}

Question:
__QUESTION__

Final answer contract:
__ANSWER_CONTRACT__

Evidence:
__EVIDENCE_CONTEXT__
"""

LOCAL_ANSWER_MODE_INSTRUCTION = """LOCAL ANSWER MODE:
Answer only the supplied local question.

Return the shortest direct answer supported by the retrieved evidence.

Do not solve an outer multi-hop question. Do not explain. Do not generate
follow-up questions.

If the local answer cannot be determined from the supplied evidence, return
UNKNOWN.

If multiple directly supported answer members are required, preserve all of
them."""

FINAL_ANSWER_MODE_INSTRUCTION = """FINAL ANSWER MODE:
The target question is the original question. The retrieval question is its
evidence-grounded concrete form. Answer the target question using the retrieved
evidence and the supplied known facts."""


ANSWER_CRITICAL_ERRORS = {
    "answer_empty",
    "boolean_must_be_yes_or_no",
    "answer_not_in_allowed_choices",
    "answer_type_mismatch",
    "answer_not_in_evidence",
    "answer_does_not_fill_slot",
    "answer_is_bridge_entity",
    "candidate_answer_mismatch",
    "relation_direction_wrong",
}
METADATA_ERRORS = {
    "unknown_supporting_evidence_id",
    "answer_not_in_evidence",
    "invalid_evidence_fact_id",
}
CONSISTENCY_ERRORS = {
    "answer_check_missing",
    "candidate_answer_mismatch",
    "answer_does_not_fill_slot",
    "answer_is_bridge_entity",
    "relation_direction_wrong",
}


NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}


def infer_answer_contract(question: str) -> AnswerContract:
    """Freeze a deterministic final-answer contract from the original question."""

    raw = normalize_space(str(question or ""))
    text = raw.casefold().rstrip(" ?")
    final_relation = infer_final_relation(text)

    # Determine the requested answer form before looking for comparison entities.
    # Otherwise a date question containing "A or B" is incorrectly frozen as CHOICE.
    if is_date_question(text):
        return AnswerContract(
            mode="DATE",
            answer_type="date",
            answer_subtype="date_or_year",
            preserve_full_entity_name=False,
            final_relation=final_relation,
        )
    if re.search(r"\bhow many\b", text):
        return AnswerContract(
            mode="QUANTITY",
            answer_type="number",
            answer_subtype="quantity_with_unit",
            required_unit=infer_required_unit(raw),
            preserve_full_entity_name=False,
            final_relation=final_relation,
        )
    if re.search(
        r"\b(?:what|which)\s+(?:rank|ranking|number|position)\b|\bwhat\s+ranking\b",
        text,
    ):
        return AnswerContract(
            mode="ORDINAL",
            answer_type="number",
            answer_subtype="rank_or_ordinal",
            preserve_full_entity_name=False,
            final_relation=final_relation,
        )
    choices = extract_question_choices(raw)
    if len(choices) == 2 and re.search(
        r"\b(?:first|earlier|later|older|younger|more|less|longer|shorter)\b",
        text,
    ):
        return AnswerContract(
            mode="CHOICE",
            answer_type="choice",
            answer_subtype="binary_choice",
            allowed_answers=choices,
            must_be_extractive=False,
            preserve_full_entity_name=False,
            final_relation=final_relation,
        )
    if re.match(
        r"^(?:are|is|was|were|do|does|did|have|has|had|can|could|will|would)\b",
        text,
    ):
        return AnswerContract(
            mode="BOOLEAN",
            answer_type="boolean",
            answer_subtype="yes_no",
            allowed_answers=["yes", "no"],
            must_be_extractive=False,
            preserve_full_entity_name=False,
            final_relation=final_relation,
        )
    list_match = re.search(
        r"\b(?:what|which|name|list)\s+"
        r"(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b",
        text,
    )
    if list_match:
        return AnswerContract(
            mode="LIST",
            answer_type="list",
            answer_subtype="exact_cardinality",
            expected_cardinality=NUMBER_WORDS[list_match.group(1)],
            preserve_full_entity_name=False,
            final_relation=final_relation,
        )
    if len(choices) == 2:
        return AnswerContract(
            mode="CHOICE",
            answer_type="choice",
            answer_subtype="binary_choice",
            allowed_answers=choices,
            must_be_extractive=False,
            preserve_full_entity_name=False,
            final_relation=final_relation,
        )

    entity_type = infer_entity_answer_type(text)
    if entity_type:
        return AnswerContract(
            mode="ENTITY",
            answer_type=entity_type,
            answer_subtype="canonical_entity",
            preserve_full_entity_name=True,
            final_relation=final_relation,
        )
    if re.match(r"^(?:who|whom)\b", text):
        return AnswerContract(
            mode="ENTITY",
            answer_type="unknown",
            answer_subtype="canonical_entity",
            preserve_full_entity_name=True,
            final_relation=final_relation,
        )
    return AnswerContract(
        mode="ATTRIBUTE",
        answer_type="unknown",
        answer_subtype="property_value",
        preserve_full_entity_name=False,
        final_relation=final_relation,
    )


def is_date_question(text: str) -> bool:
    normalized = normalize_space(str(text or "")).casefold().rstrip(" ?")
    return bool(
        re.match(
            r"^(?:when\b|what year\b|which year\b|"
            r"in what year\b|what date\b|which date\b)",
            normalized,
        )
        or re.search(r"\bwhen$", normalized)
    )


def infer_entity_answer_type(text: str) -> str:
    if re.search(
        r"^(?:who|whom)\b.*\b(?:direct(?:ed|or)|wrote|written|author|"
        r"father|mother|spouse|husband|wife|married)\b",
        text,
    ):
        return "person"
    if re.search(
        r"\b(?:what|which) (?:person|actor|actress|author|player|president|politician)\b",
        text,
    ):
        return "person"
    if re.search(
        r"^(?:where)\b|"
        r"\b(?:what|which)(?:\s+(?:is|was|are|were))?\s+(?:the\s+)?"
        r"(?:city|town|country|place|state|county|province|region|location|capital)\b",
        text,
    ):
        return "place"
    if re.search(r"\b(?:what|which) (?:company|organization|club|team|university)\b", text):
        return "organization"
    if re.search(r"\b(?:what|which) (?:film|movie|book|song|album|work|series)\b", text):
        return "work"
    return ""


def infer_final_relation(text: str) -> str:
    """Infer the semantic relation requested by a question.

    Patterns are deliberately ordered from specific relations to broad ones.
    In particular, a leading ``where`` determines the answer form but must not
    collapse birth place, death place, workplace, and generic location into the
    same relation.
    """

    normalized = normalize_space(str(text or "")).casefold().rstrip(" ?")
    for relation, pattern in (
        (
            "birth_country",
            r"\b(?:what|which)\s+country\b.*\bborn\b|"
            r"\bborn\b.*\b(?:what|which)\s+country\b|"
            r"\bcountry of birth\b",
        ),
        (
            "birth_place",
            r"^where\b.*\bborn\b|\bplace of birth\b|\bbirthplace\b",
        ),
        (
            "birth_date",
            r"^when\b.*\bborn\b|"
            r"^(?:in\s+)?(?:what|which)\s+(?:date|year)\b.*\bborn\b|"
            r"\b(?:date|year) of birth\b",
        ),
        (
            "death_country",
            r"\b(?:what|which)\s+country\b.*\b(?:die|died|death)\b|"
            r"\b(?:die|died|death)\b.*\b(?:what|which)\s+country\b|"
            r"\bcountry of death\b",
        ),
        (
            "death_place",
            r"^where\b.*\b(?:die|died)\b|"
            r"\b(?:place of death|death place)\b",
        ),
        (
            "death_date",
            r"^when\b.*\b(?:die|died)\b|"
            r"^(?:in\s+)?(?:what|which)\s+(?:date|year)\b.*\b(?:die|died)\b|"
            r"\b(?:date|year) of death\b",
        ),
        ("nationality", r"\bnationalit(?:y|ies)\b|\bcitizenship\b"),
        (
            "workplace",
            r"^where\b.*\bwork(?:ed|s|ing)?\b|"
            r"\bwork(?:ed|s|ing)?\s+(?:at|for)\b",
        ),
        ("director", r"\bwho directed\b|\bdirector of\b|\bdirected by whom\b"),
        ("author", r"\bwho wrote\b|\bauthor of\b|\bwritten by whom\b"),
        ("father", r"\bfather\b"),
        ("mother", r"\bmother\b"),
        ("spouse", r"\bspouse\b|\bhusband\b|\bwife\b|\bmarried to\b"),
        ("creator", r"\bcreator of\b|\bwho created\b"),
        ("nickname", r"\bnickname\b"),
        ("name", r"\b(?:former|english|birth|full) name\b"),
        ("reason", r"\bwhy\b|\breason\b"),
        ("location", r"^where\b|\blocated\b|\blocation\b|\bheadquarters\b"),
    ):
        if re.search(pattern, normalized):
            return relation
    return "property_value"


def infer_required_unit(question: str) -> str:
    match = re.search(r"\bhow many\s+([A-Za-z-]+)", str(question or ""), re.IGNORECASE)
    return match.group(1).casefold() if match else ""


def extract_question_choices(question: str) -> list[str]:
    """Conservatively identify explicit two-way choices; otherwise return none."""

    text = normalize_space(str(question or "")).rstrip(" ?")
    patterns = [
        r"^between\s+(.+?)\s+and\s+(.+?),\s*which\b",
        r"^which\s+[^,]+,\s*(?:the\s+)?(.+?)\s+or\s+(?:the\s+)?(.+?),",
        r",\s*([^,?]+?)\s+or\s+([^,?]+?)$",
        r"^(?:was|is|were|are)\s+(.+?)\s+or\s+(.+?)\s+"
        r"(?:born|created|released|founded|published|built|elected)\s+"
        r"(?:first|earlier|later)\b",
        r"\b(?:which|is)\b.*?\b(?:the\s+)?([^,?]+?)\s+or\s+([^,?]+?)(?:\s+(?:more|less|older|younger|larger|smaller)\b|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            continue
        choices = [match.group(1).strip(), match.group(2).strip()]
        if all(choice and len(choice.split()) <= 10 for choice in choices):
            return choices
    return []


@dataclass(slots=True)
class EvidenceItem:
    evidence_id: str
    title: str
    text: str
    focus_sentence: str = ""
    context_before: str = ""
    context_after: str = ""
    doc_id: str | None = None
    sentence_index: int | None = None
    covered_sentence_indices: list[int] | None = None
    score: float | None = None
    rerank_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ReaderEvidenceBinding:
    local_id: str
    global_id: str
    evidence: EvidenceItem


def build_reader_evidence_bindings(
    evidence: list[EvidenceItem],
) -> list[ReaderEvidenceBinding]:
    return [
        ReaderEvidenceBinding(
            local_id=f"E{index}",
            global_id=item.evidence_id,
            evidence=item,
        )
        for index, item in enumerate(evidence, start=1)
    ]


def localize_reader_evidence(
    bindings: list[ReaderEvidenceBinding],
) -> list[EvidenceItem]:
    return [
        replace(binding.evidence, evidence_id=binding.local_id)
        for binding in bindings
    ]


def build_evidence_id_aliases(
    evidence: list[EvidenceItem],
) -> dict[str, EvidenceItem]:
    """Build unambiguous full, local-position, and trailing-E aliases."""

    aliases: dict[str, EvidenceItem] = {}
    suffix_candidates: dict[str, list[EvidenceItem]] = {}
    for index, item in enumerate(evidence, start=1):
        aliases[item.evidence_id.casefold()] = item
        aliases.setdefault(f"e{index}", item)
        match = re.search(r"(E\d+)$", item.evidence_id, flags=re.IGNORECASE)
        if match:
            suffix_candidates.setdefault(match.group(1).casefold(), []).append(item)
    for suffix, candidates in suffix_candidates.items():
        if len(candidates) == 1:
            aliases.setdefault(suffix, candidates[0])
    return aliases


def normalize_reader_evidence_id(
    value: Any,
    bindings: list[ReaderEvidenceBinding],
) -> str:
    text = str(value or "").strip().strip("[](){}.,;:")
    if not text:
        return ""
    aliases: dict[str, str] = {}
    suffixes: dict[str, list[str]] = {}
    for binding in bindings:
        aliases[binding.local_id.casefold()] = binding.global_id
        aliases[binding.global_id.casefold()] = binding.global_id
        match = re.search(r"(E\d+)$", binding.global_id, flags=re.IGNORECASE)
        if match:
            suffixes.setdefault(match.group(1).casefold(), []).append(
                binding.global_id
            )
    direct = aliases.get(text.casefold())
    if direct:
        return direct
    matches = suffixes.get(text.casefold(), [])
    return matches[0] if len(matches) == 1 else text


def map_reader_output_to_global_ids(
    parsed: dict[str, Any],
    bindings: list[ReaderEvidenceBinding],
) -> dict[str, Any]:
    mapped = dict(parsed)
    supporting = parsed.get("supporting_evidence_ids") or []
    if not isinstance(supporting, list):
        supporting = []
    mapped["supporting_evidence_ids"] = list(
        dict.fromkeys(
            normalize_reader_evidence_id(item, bindings)
            for item in supporting
            if str(item or "").strip()
        )
    )
    mapped_facts: list[dict[str, Any]] = []
    evidence_facts = parsed.get("evidence_facts") or []
    if not isinstance(evidence_facts, list):
        evidence_facts = []
    for item in evidence_facts:
        if not isinstance(item, dict):
            continue
        mapped_fact = dict(item)
        mapped_fact["evidence_id"] = normalize_reader_evidence_id(
            item.get("evidence_id"),
            bindings,
        )
        mapped_facts.append(mapped_fact)
    mapped["evidence_facts"] = mapped_facts
    mapped_bindings: list[dict[str, Any]] = []
    bridge_bindings = parsed.get("bridge_bindings") or []
    if not isinstance(bridge_bindings, list):
        bridge_bindings = []
    for item in bridge_bindings:
        if not isinstance(item, dict):
            continue
        mapped_binding = dict(item)
        mapped_binding["evidence_id"] = normalize_reader_evidence_id(
            item.get("evidence_id"),
            bindings,
        )
        mapped_bindings.append(mapped_binding)
    mapped["bridge_bindings"] = mapped_bindings
    return mapped


def build_evidence_context(
    retrieved_sentences: list[Any],
    top_k: int = 20,
    *,
    evidence_id_prefix: str = "E",
) -> list[EvidenceItem]:
    evidence: list[EvidenceItem] = []
    seen: set[tuple[str, str]] = set()

    for item in retrieved_sentences:
        if len(evidence) >= top_k:
            break

        title = str(_item_get(item, "title", "") or "").strip()
        focus_sentence = normalize_space(
            str(_item_get(item, "sentence", "") or "")
        )
        text = _item_get(item, "window", None) or focus_sentence
        text = normalize_space(str(text or ""))
        if not title or not text:
            continue
        if not focus_sentence:
            focus_sentence = text
        context_before, context_after = split_focus_context(
            text,
            focus_sentence,
        )

        key = (title.casefold(), text.casefold())
        if key in seen:
            continue
        seen.add(key)

        score = _optional_float(_item_get(item, "score", None))
        rerank_score = _optional_float(_item_get(item, "rerank_score", None))
        evidence.append(
            EvidenceItem(
                evidence_id=f"{evidence_id_prefix}{len(evidence) + 1}",
                title=title,
                text=text,
                focus_sentence=focus_sentence,
                context_before=context_before,
                context_after=context_after,
                doc_id=str(_item_get(item, "doc_id", "") or "").strip() or None,
                sentence_index=_optional_int(_item_get(item, "sentence_index", None)),
                covered_sentence_indices=_int_list(_item_get(item, "covered_sentence_indices", None)),
                score=score,
                rerank_score=rerank_score if rerank_score is not None else score,
            )
        )

    return evidence


def format_evidence_for_prompt(evidence: list[EvidenceItem]) -> str:
    if not evidence:
        return "(no evidence)"
    blocks = []
    for item in evidence:
        lines = [
            f"[{item.evidence_id}]",
            f"Title: {item.title}",
            f"Focused evidence: {item.focus_sentence or item.text}",
        ]
        neighbor = normalize_space(
            " ".join(part for part in (item.context_before, item.context_after) if part)
        )
        if neighbor:
            lines.append(f"Neighbor context: {neighbor}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def estimate_text_tokens(text: str) -> int:
    """Deterministic tokenizer-independent estimate used for experiment logs."""

    return len(re.findall(r"[\w]+|[^\w\s]", str(text or ""), re.UNICODE))


def split_focus_context(window: str, focus_sentence: str) -> tuple[str, str]:
    window = normalize_space(window)
    focus = normalize_space(focus_sentence)
    if not window or not focus or normalize_space(window).casefold() == focus.casefold():
        return "", ""
    match = re.search(re.escape(focus), window, flags=re.IGNORECASE)
    if not match:
        return "", ""
    return window[: match.start()].strip(), window[match.end() :].strip()


def build_normal_rag_prompt(
    question: str,
    evidence: list[EvidenceItem],
    *,
    answer_contract: AnswerContract | None = None,
) -> str:
    contract = answer_contract or infer_answer_contract(question)
    return build_rag_prompt(
        question=question,
        evidence=evidence,
        contract=contract,
    )


def build_rag_prompt(
    *,
    question: str,
    evidence: list[EvidenceItem],
    contract: AnswerContract,
    mode_instruction: str = "",
) -> str:
    prompt = (
        NORMAL_RAG_PROMPT.replace("__QUESTION__", question.strip())
        .replace(
            "__ANSWER_CONTRACT__",
            json.dumps(contract.to_dict(), ensure_ascii=False, indent=2),
        )
        .replace("__EVIDENCE_CONTEXT__", format_evidence_for_prompt(evidence))
    )
    if mode_instruction.strip():
        prompt += "\n\n" + mode_instruction.strip()
    return prompt


def canonicalize_boolean(answer: str) -> str:
    key = normalize_space(str(answer or "").casefold())
    if key in {"true", "yes", "correct", "both"}:
        return "yes"
    if key in {"false", "no", "incorrect", "not both"}:
        return "no"
    return answer


def answer_types_compatible(expected: str, actual: str) -> bool:
    expected_key = str(expected or "unknown").casefold()
    actual_key = str(actual or "unknown").casefold()
    if "unknown" in {expected_key, actual_key} or not expected_key or not actual_key:
        return True
    groups = (
        {"person", "human"},
        {"place", "location", "city", "country", "state", "county"},
        {"organization", "company", "team", "club", "group", "university"},
        {"work", "film", "movie", "book", "song", "album", "series"},
        {"number", "quantity", "ordinal", "rank"},
        {"date", "year"},
        {"boolean", "yes_no"},
    )
    return expected_key == actual_key or any(
        expected_key in group and actual_key in group for group in groups
    )


def validate_answer_output(
    *,
    parsed: dict[str, Any],
    evidence: list[EvidenceItem],
    contract: AnswerContract,
) -> list[str]:
    errors: list[str] = []
    answer = str(parsed.get("answer") or "").strip()
    evidence_by_id = build_evidence_id_aliases(evidence)

    if not answer:
        errors.append("answer_empty")
    supporting_ids = parsed.get("supporting_evidence_ids") or []
    if not isinstance(supporting_ids, list):
        errors.append("supporting_evidence_ids_not_list")
        supporting_ids = []
    if any(
        str(item or "").strip().casefold() not in evidence_by_id
        for item in supporting_ids
        if str(item or "").strip()
    ):
        errors.append("unknown_supporting_evidence_id")
    evidence_facts = parsed.get("evidence_facts") or []
    if isinstance(evidence_facts, list):
        for item in evidence_facts:
            if not isinstance(item, dict):
                errors.append("invalid_evidence_fact_id")
                continue
            evidence_id = str(item.get("evidence_id") or "").strip()
            if not evidence_id or evidence_id.casefold() not in evidence_by_id:
                errors.append("invalid_evidence_fact_id")
    if contract.must_be_extractive and answer and not answer_in_evidence(answer, evidence):
        errors.append("answer_not_in_evidence")
    if contract.mode == "BOOLEAN" and normalize_space(answer.casefold()) not in {"yes", "no"}:
        errors.append("boolean_must_be_yes_or_no")
    allowed = list(contract.allowed_answers or [])
    if contract.mode == "CHOICE" and allowed and hotpot_normalize_answer(answer) not in {
        hotpot_normalize_answer(item) for item in allowed
    }:
        errors.append("answer_not_in_allowed_choices")
    actual_type = str(parsed.get("answer_type") or "unknown")
    if contract.mode != "CHOICE" and not answer_types_compatible(
        contract.answer_type,
        actual_type,
    ):
        errors.append("answer_type_mismatch")
    errors.extend(validate_answer_consistency(parsed))
    return errors


def validate_answer_consistency(parsed: dict[str, Any]) -> list[str]:
    """Record lightweight candidate/final-answer inconsistencies without repair."""

    errors: list[str] = []
    candidate = hotpot_normalize_answer(parsed.get("candidate_answer"))
    answer = hotpot_normalize_answer(parsed.get("answer"))

    answer_check = parsed.get("answer_check")
    if not isinstance(answer_check, dict):
        errors.append("answer_check_missing")
        return errors

    if candidate and answer and candidate != answer:
        errors.append("candidate_answer_mismatch")
    if answer_check.get("fills_requested_slot") is not True:
        errors.append("answer_does_not_fill_slot")
    if answer_check.get("is_bridge_entity") is True:
        errors.append("answer_is_bridge_entity")
    if answer_check.get("relation_direction_correct") is not True:
        errors.append("relation_direction_wrong")
    return errors


def evidence_search_text(item: EvidenceItem) -> str:
    return normalize_space(
        " ".join(
            part
            for part in (
                item.title,
                item.focus_sentence,
                item.context_before,
                item.context_after,
                item.text,
            )
            if part
        )
    )


def canonicalize_contract_answer(
    answer: str,
    *,
    contract: AnswerContract,
    evidence: list[EvidenceItem],
) -> str:
    result = normalize_answer_for_eval(answer)
    if contract.mode == "BOOLEAN":
        return canonicalize_boolean(result)
    if contract.mode in {"QUANTITY", "ORDINAL"}:
        return recover_quantity_span(
            answer=result,
            evidence=evidence,
            required_unit=contract.required_unit,
            ordinal=contract.mode == "ORDINAL",
        )
    return result


def recover_quantity_span(
    *,
    answer: str,
    evidence: list[EvidenceItem],
    required_unit: str,
    ordinal: bool = False,
) -> str:
    numeric = numeric_value(answer)
    if numeric is None:
        return answer
    patterns = []
    if ordinal:
        patterns.append(
            re.compile(rf"\b(?:number\s+)?{numeric}(?:st|nd|rd|th)?\b", re.IGNORECASE)
        )
    number_forms = [str(numeric), *(word for word, value in NUMBER_WORDS.items() if value == numeric)]
    escaped = "|".join(re.escape(item) for item in number_forms)
    if required_unit:
        patterns.append(
            re.compile(
                rf"\b(?:{escaped})\s+{re.escape(required_unit)}\b",
                re.IGNORECASE,
            )
        )
    patterns.append(re.compile(rf"\b(?:{escaped})\b", re.IGNORECASE))
    candidates: list[str] = []
    for item in evidence:
        source = evidence_search_text(item)
        for pattern in patterns:
            candidates.extend(match.group(0).strip() for match in pattern.finditer(source))
        if candidates:
            break
    if not candidates:
        return answer
    if required_unit:
        with_unit = [item for item in candidates if required_unit.casefold() in item.casefold()]
        if with_unit:
            return min(with_unit, key=len)
    if ordinal:
        with_marker = [
            item for item in candidates if re.search(r"\bnumber\b|(?:st|nd|rd|th)\b", item, re.IGNORECASE)
        ]
        if with_marker:
            return min(with_marker, key=len)
    return min(candidates, key=len)


def numeric_value(answer: str) -> int | None:
    match = re.search(r"\b\d+\b", str(answer or ""))
    if match:
        return int(match.group(0))
    tokens = re.findall(r"[a-z]+", str(answer or "").casefold())
    for token in tokens:
        if token in NUMBER_WORDS:
            return NUMBER_WORDS[token]
    return None


def critical_errors(errors: list[str]) -> set[str]:
    return {
        error
        for error in errors
        if error in ANSWER_CRITICAL_ERRORS
    }


def call_reader_with_one_repair(
    *,
    prompt: str,
    llm_client: PromptClient,
    evidence: list[EvidenceItem],
    contract: AnswerContract,
) -> tuple[dict[str, Any], str, str | None, list[str], bool]:
    raw_text = llm_client(prompt)
    parse_error: str | None = None
    try:
        parsed = parse_json_object(raw_text)
    except Exception as exc:  # noqa: BLE001
        parsed = {"answer": raw_text}
        parse_error = str(exc)

    parsed["answer"] = canonicalize_contract_answer(
        str(parsed.get("answer") or ""),
        contract=contract,
        evidence=evidence,
    )
    errors = validate_answer_output(parsed=parsed, evidence=evidence, contract=contract)
    repair_attempted = False
    original_critical = critical_errors(errors)
    if not original_critical:
        return parsed, raw_text, parse_error, errors, repair_attempted

    repair_attempted = True
    repair_prompt = (
        prompt
        + "\n\nThe previous answer violated these answer-critical constraints:\n- "
        + "\n- ".join(sorted(original_critical))
        + "\n\nPrevious output:\n"
        + json.dumps(parsed, ensure_ascii=False, indent=2)
        + "\n\nReturn corrected JSON only. Preserve the semantic target of the "
        "question and correct the final answer whenever it is unsupported, fills the "
        "wrong slot, returns a bridge entity, mismatches the candidate, reverses the "
        "relation, has the wrong type, violates Boolean format, or is outside the "
        "allowed choices. Keep the prescribed reasoning-before-answer field order. "
        "Do not rewrite an otherwise correct answer merely to polish metadata."
    )
    repaired_text = llm_client(repair_prompt)
    try:
        repaired = parse_json_object(repaired_text)
        repaired["answer"] = canonicalize_contract_answer(
            str(repaired.get("answer") or ""),
            contract=contract,
            evidence=evidence,
        )
        repaired_errors = validate_answer_output(
            parsed=repaired,
            evidence=evidence,
            contract=contract,
        )
        repaired_critical = critical_errors(repaired_errors)
        if len(repaired_critical) < len(original_critical):
            parsed, raw_text, errors, parse_error = (
                repaired,
                repaired_text,
                repaired_errors,
                None,
            )
    except Exception:  # noqa: BLE001
        pass
    return parsed, raw_text, parse_error, errors, repair_attempted


def normal_rag_answer(
    question: str,
    retrieval_result: Any,
    llm_client: PromptClient,
    *,
    top_k_sentences: int = 20,
    evidence_id_prefix: str = "E",
    answer_mode: str = "final",
) -> dict[str, Any]:
    if answer_mode not in {"local", "final"}:
        raise ValueError("answer_mode must be 'local' or 'final'")
    sentences = _item_get(retrieval_result, "sentences", []) or []
    evidence = build_evidence_context(
        list(sentences),
        top_k=top_k_sentences,
        evidence_id_prefix=evidence_id_prefix,
    )
    contract = infer_answer_contract(question)
    bindings = build_reader_evidence_bindings(evidence)
    reader_evidence = localize_reader_evidence(bindings)
    prompt = build_rag_prompt(
        question=question,
        evidence=reader_evidence,
        contract=contract,
        mode_instruction=(
            LOCAL_ANSWER_MODE_INSTRUCTION if answer_mode == "local" else ""
        ),
    )
    parsed, raw_output, parse_error, reader_validation_errors, repair_attempted = (
        call_reader_with_one_repair(
            prompt=prompt,
            llm_client=llm_client,
            evidence=reader_evidence,
            contract=contract,
        )
    )
    parsed = map_reader_output_to_global_ids(parsed, bindings)
    post_mapping_validation_errors = validate_answer_output(
        parsed=parsed,
        evidence=evidence,
        contract=contract,
    )
    validation_errors = list(post_mapping_validation_errors)

    answer = normalize_answer_for_eval(str(parsed.get("answer", "")))
    supporting_ids = parsed.get("supporting_evidence_ids", [])
    if not isinstance(supporting_ids, list):
        supporting_ids = []

    return {
        "question": question,
        "answer_mode": answer_mode,
        "answer": answer,
        "raw_answer": parsed.get("answer"),
        "resolved_question": str(parsed.get("resolved_question") or "").strip(),
        "bridge_bindings": (
            parsed.get("bridge_bindings")
            if isinstance(parsed.get("bridge_bindings"), list)
            else []
        ),
        "evidence_facts": (
            parsed.get("evidence_facts")
            if isinstance(parsed.get("evidence_facts"), list)
            else []
        ),
        "derivation": str(parsed.get("derivation") or "").strip(),
        "candidate_answer": str(parsed.get("candidate_answer") or "").strip(),
        "answer_check": (
            parsed.get("answer_check")
            if isinstance(parsed.get("answer_check"), dict)
            else {}
        ),
        "answer_type": str(parsed.get("answer_type") or "unknown"),
        "raw_output": parsed,
        "raw_text": raw_output,
        "supporting_evidence_ids": [str(item) for item in supporting_ids],
        "parse_error": parse_error,
        "validation_errors": validation_errors,
        "consistency_errors": [
            error for error in validation_errors if error in CONSISTENCY_ERRORS
        ],
        "reader_validation_errors": reader_validation_errors,
        "post_mapping_validation_errors": post_mapping_validation_errors,
        "final_validation_errors": validation_errors,
        "repair_attempted": repair_attempted,
        "answer_contract": contract.to_dict(),
        "evidence": [item.to_dict() for item in evidence],
        "top_k_sentences": top_k_sentences,
        "evidence_token_count": estimate_text_tokens(
            format_evidence_for_prompt(reader_evidence)
        ),
        "evidence_token_count_method": "lexical_token_estimate_v1",
        "prompt": prompt,
    }


def normal_rag_answer_from_evidence(
    question: str,
    evidence: list[EvidenceItem] | list[dict[str, Any]],
    llm_client: PromptClient,
    *,
    top_k_sentences: int = 30,
    evidence_id_prefix: str = "P-E",
    answer_contract: AnswerContract | None = None,
    answer_mode: str = "final",
    target_question: str | None = None,
    current_question: str | None = None,
    known_facts: list[str] | None = None,
) -> dict[str, Any]:
    """Answer from an accumulated evidence pool without retrieving again."""

    if answer_mode not in {"local", "final"}:
        raise ValueError("answer_mode must be 'local' or 'final'")
    normalized_evidence: list[EvidenceItem] = []
    for index, item in enumerate(evidence[:top_k_sentences], start=1):
        title = str(_item_get(item, "title", "") or "").strip()
        text = normalize_space(str(_item_get(item, "text", "") or ""))
        if not title or not text:
            continue
        normalized_evidence.append(
            EvidenceItem(
                evidence_id=(
                    str(_item_get(item, "evidence_id", "") or "").strip()
                    or f"{evidence_id_prefix}{index}"
                ),
                title=title,
                text=text,
                focus_sentence=normalize_space(
                    str(_item_get(item, "focus_sentence", "") or text)
                ),
                context_before=normalize_space(
                    str(_item_get(item, "context_before", "") or "")
                ),
                context_after=normalize_space(
                    str(_item_get(item, "context_after", "") or "")
                ),
                doc_id=str(_item_get(item, "doc_id", "") or "").strip() or None,
                sentence_index=_optional_int(
                    _item_get(item, "sentence_index", None)
                ),
                covered_sentence_indices=_int_list(
                    _item_get(item, "covered_sentence_indices", None)
                ),
                score=_optional_float(_item_get(item, "score", None)),
                rerank_score=_optional_float(
                    _item_get(item, "rerank_score", _item_get(item, "score", None))
                ),
            )
        )

    if answer_mode == "final":
        for fact_index, fact in enumerate(known_facts or [], start=1):
            fact_text = normalize_space(str(fact or ""))
            if fact_text:
                normalized_evidence.append(
                    EvidenceItem(
                        evidence_id=f"K{fact_index}",
                        title="Known facts",
                        text=fact_text,
                        focus_sentence=fact_text,
                    )
                )

    answer_question = str(
        question if answer_mode == "local" else target_question or question
    ).strip()
    contract = answer_contract or infer_answer_contract(answer_question)
    bindings = build_reader_evidence_bindings(normalized_evidence)
    reader_evidence = localize_reader_evidence(bindings)
    prompt = build_rag_prompt(
        question=answer_question,
        evidence=reader_evidence,
        contract=contract,
        mode_instruction=(
            LOCAL_ANSWER_MODE_INSTRUCTION
            if answer_mode == "local"
            else (
                FINAL_ANSWER_MODE_INSTRUCTION
                + "\n\nTARGET QUESTION:\n"
                + answer_question
                + "\n\nCURRENT RETRIEVAL QUESTION:\n"
                + str(current_question or question).strip()
            )
        ),
    )
    parsed, raw_output, parse_error, reader_validation_errors, repair_attempted = (
        call_reader_with_one_repair(
            prompt=prompt,
            llm_client=llm_client,
            evidence=reader_evidence,
            contract=contract,
        )
    )
    parsed = map_reader_output_to_global_ids(parsed, bindings)
    post_mapping_validation_errors = validate_answer_output(
        parsed=parsed,
        evidence=normalized_evidence,
        contract=contract,
    )
    validation_errors = list(post_mapping_validation_errors)

    supporting_ids = parsed.get("supporting_evidence_ids", [])
    if not isinstance(supporting_ids, list):
        supporting_ids = []
    return {
        "question": answer_question,
        "retrieval_question": str(current_question or question).strip(),
        "answer_mode": answer_mode,
        "answer": normalize_answer_for_eval(str(parsed.get("answer", ""))),
        "raw_answer": parsed.get("answer"),
        "resolved_question": str(parsed.get("resolved_question") or "").strip(),
        "bridge_bindings": (
            parsed.get("bridge_bindings")
            if isinstance(parsed.get("bridge_bindings"), list)
            else []
        ),
        "evidence_facts": (
            parsed.get("evidence_facts")
            if isinstance(parsed.get("evidence_facts"), list)
            else []
        ),
        "derivation": str(parsed.get("derivation") or "").strip(),
        "candidate_answer": str(parsed.get("candidate_answer") or "").strip(),
        "answer_check": (
            parsed.get("answer_check")
            if isinstance(parsed.get("answer_check"), dict)
            else {}
        ),
        "answer_type": str(parsed.get("answer_type") or "unknown"),
        "raw_output": parsed,
        "raw_text": raw_output,
        "supporting_evidence_ids": [str(item) for item in supporting_ids],
        "parse_error": parse_error,
        "validation_errors": validation_errors,
        "consistency_errors": [
            error for error in validation_errors if error in CONSISTENCY_ERRORS
        ],
        "reader_validation_errors": reader_validation_errors,
        "post_mapping_validation_errors": post_mapping_validation_errors,
        "final_validation_errors": validation_errors,
        "repair_attempted": repair_attempted,
        "answer_contract": contract.to_dict(),
        "evidence": [item.to_dict() for item in normalized_evidence],
        "top_k_sentences": top_k_sentences,
        "evidence_token_count": estimate_text_tokens(
            format_evidence_for_prompt(reader_evidence)
        ),
        "evidence_token_count_method": "lexical_token_estimate_v1",
        "prompt": prompt,
    }


class NormalRAG:
    """Small structured runner around the existing retriever and reader."""

    def __init__(
        self,
        *,
        retriever: Any,
        llm_client: PromptClient,
        top_k_sentences: int = 20,
        final_top_k_sentences: int = 30,
    ) -> None:
        self.retriever = retriever
        self.llm_client = llm_client
        self.top_k_sentences = top_k_sentences
        self.final_top_k_sentences = final_top_k_sentences

    def run(
        self,
        *,
        question: str,
        return_documents: bool = True,
        answer_mode: str = "local",
        target_question: str | None = None,
        known_facts: list[str] | None = None,
    ) -> dict[str, Any]:
        query = str(question or "").strip()
        if not query:
            raise ValueError("question must not be empty")
        if answer_mode not in {"local", "final"}:
            raise ValueError("answer_mode must be 'local' or 'final'")
        try:
            retrieval_result = self.retriever.retrieve(query)
            if answer_mode == "local":
                reader_result = normal_rag_answer(
                    query,
                    retrieval_result,
                    self.llm_client,
                    top_k_sentences=self.top_k_sentences,
                    answer_mode="local",
                )
            else:
                evidence = build_evidence_context(
                    list(_item_get(retrieval_result, "sentences", []) or []),
                    top_k=self.final_top_k_sentences,
                )
                reader_result = normal_rag_answer_from_evidence(
                    question=query,
                    evidence=evidence,
                    llm_client=self.llm_client,
                    top_k_sentences=self.final_top_k_sentences,
                    answer_mode="final",
                    target_question=target_question or query,
                    current_question=query,
                    known_facts=known_facts or [],
                )
            documents = _structured_rag_documents(reader_result)
            return {
                "question": query,
                "target_question": str(target_question or query).strip(),
                "answer_mode": answer_mode,
                "answer": str(reader_result.get("answer") or "").strip(),
                "answer_type": str(reader_result.get("answer_type") or "unknown"),
                "documents": documents if return_documents else [],
                "supporting_evidence_ids": list(
                    reader_result.get("supporting_evidence_ids") or []
                ),
                "reader_result": reader_result,
                "evidence_token_count": int(
                    reader_result.get("evidence_token_count") or 0
                ),
                "evidence_token_count_method": str(
                    reader_result.get("evidence_token_count_method")
                    or "lexical_token_estimate_v1"
                ),
                "retrieval_diagnostics": dict(
                    _item_get(retrieval_result, "diagnostics", {}) or {}
                ),
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "question": query,
                "target_question": str(target_question or query).strip(),
                "answer_mode": answer_mode,
                "answer": "",
                "answer_type": "unknown",
                "documents": [],
                "supporting_evidence_ids": [],
                "reader_result": None,
                "retrieval_diagnostics": {},
                "evidence_token_count": 0,
                "evidence_token_count_method": "lexical_token_estimate_v1",
                "error": str(exc),
            }

    def retrieve_documents(
        self,
        *,
        question: str,
        top_k_sentences: int | None = None,
        evidence_id_prefix: str = "E",
    ) -> dict[str, Any]:
        """Retrieve a ranked evidence list without making a reader LLM call."""

        query = str(question or "").strip()
        if not query:
            raise ValueError("question must not be empty")
        try:
            retrieval_result = self.retriever.retrieve(query)
            evidence = build_evidence_context(
                list(_item_get(retrieval_result, "sentences", []) or []),
                top_k=(
                    self.final_top_k_sentences
                    if top_k_sentences is None
                    else max(0, int(top_k_sentences))
                ),
                evidence_id_prefix=evidence_id_prefix,
            )
            return {
                "question": query,
                "documents": _structured_rag_documents(
                    {"evidence": [item.to_dict() for item in evidence]}
                ),
                "retrieval_diagnostics": dict(
                    _item_get(retrieval_result, "diagnostics", {}) or {}
                ),
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "question": query,
                "documents": [],
                "retrieval_diagnostics": {},
                "evidence_token_count": 0,
                "evidence_token_count_method": "lexical_token_estimate_v1",
                "error": str(exc),
            }

    def answer_from_evidence(
        self,
        *,
        target_question: str,
        reasoning_question: str,
        evidence: list[dict[str, Any]],
        known_facts: list[str] | None = None,
    ) -> dict[str, Any]:
        """Run the final reader directly over an already fused evidence pool."""

        try:
            reader_result = normal_rag_answer_from_evidence(
                question=reasoning_question,
                evidence=evidence,
                llm_client=self.llm_client,
                top_k_sentences=self.final_top_k_sentences,
                answer_mode="final",
                target_question=target_question,
                current_question=reasoning_question,
                known_facts=known_facts or [],
            )
        except Exception as exc:  # noqa: BLE001
            return {
                "question": reasoning_question,
                "target_question": target_question,
                "answer_mode": "final",
                "answer": "",
                "answer_type": "unknown",
                "documents": [],
                "supporting_evidence_ids": [],
                "reader_result": None,
                "retrieval_diagnostics": {},
                "evidence_token_count": 0,
                "evidence_token_count_method": "lexical_token_estimate_v1",
                "error": str(exc),
            }
        return {
            "question": reasoning_question,
            "target_question": target_question,
            "answer_mode": "final",
            "answer": str(reader_result.get("answer") or "").strip(),
            "answer_type": str(reader_result.get("answer_type") or "unknown"),
            "documents": _structured_rag_documents(reader_result),
            "supporting_evidence_ids": list(
                reader_result.get("supporting_evidence_ids") or []
            ),
            "reader_result": reader_result,
            "evidence_token_count": int(
                reader_result.get("evidence_token_count") or 0
            ),
            "evidence_token_count_method": str(
                reader_result.get("evidence_token_count_method")
                or "lexical_token_estimate_v1"
            ),
            "retrieval_diagnostics": {},
            "error": None,
        }


def _structured_rag_documents(reader_result: dict[str, Any]) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for index, item in enumerate(reader_result.get("evidence") or [], start=1):
        documents.append(
            {
                "document_id": str(
                    _item_get(item, "doc_id", "")
                    or _item_get(item, "evidence_id", "")
                    or f"D{index}"
                ),
                "evidence_id": str(_item_get(item, "evidence_id", "") or ""),
                "title": str(_item_get(item, "title", "") or ""),
                "text": str(_item_get(item, "text", "") or ""),
                "score": _optional_float(
                    _item_get(item, "rerank_score", _item_get(item, "score", None))
                ),
                "sentence_index": _optional_int(
                    _item_get(item, "sentence_index", None)
                ),
                "covered_sentence_indices": _int_list(
                    _item_get(item, "covered_sentence_indices", None)
                ),
            }
        )
    return documents


def normalize_answer_for_eval(answer: str) -> str:
    text = normalize_space(answer.strip().replace("\n", " "))
    prefixes = [
        "The answer is ",
        "Answer: ",
        "It is ",
        "It was ",
    ]
    lowered = text.casefold()
    for prefix in prefixes:
        if lowered.startswith(prefix.casefold()):
            text = text[len(prefix) :].strip()
            break
    return text.rstrip(" .")


def hotpot_normalize_answer(answer: Any) -> str:
    text = str(answer or "").casefold()
    text = "".join(" " if char in string.punctuation else char for char in text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return normalize_space(text)


def exact_match_score(prediction: Any, ground_truth: Any) -> int:
    return int(hotpot_normalize_answer(prediction) == hotpot_normalize_answer(ground_truth))


def f1_score(prediction: Any, ground_truth: Any) -> float:
    pred_tokens = hotpot_normalize_answer(prediction).split()
    gold_tokens = hotpot_normalize_answer(ground_truth).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common: dict[str, int] = {}
    for token in gold_tokens:
        common[token] = common.get(token, 0) + 1
    num_same = 0
    for token in pred_tokens:
        if common.get(token, 0) > 0:
            num_same += 1
            common[token] -= 1
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def answer_in_evidence(answer: Any, evidence: list[EvidenceItem] | list[dict[str, Any]]) -> bool:
    normalized_answer = hotpot_normalize_answer(answer)
    if not normalized_answer or normalized_answer in {"yes", "no", "unknown", "none"}:
        return False
    joined = " ".join(
        f"{_item_get(item, 'title', '')} {_item_get(item, 'text', '')}"
        for item in evidence
    )
    return normalized_answer in hotpot_normalize_answer(joined)


def normalize_space(text: str) -> str:
    return " ".join(text.split())


def _item_get(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_list(value: Any) -> list[int] | None:
    if not isinstance(value, list):
        return None
    items: list[int] = []
    for item in value:
        converted = _optional_int(item)
        if converted is not None:
            items.append(converted)
    return items


def dumps_jsonl_row(row: dict[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False)
