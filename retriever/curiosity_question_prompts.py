from __future__ import annotations

import json
import re
import unicodedata
from typing import Any


COMMON_QUERY_GENERATOR_PROMPT_G1 = """You generate the NEXT locally executable
evidence query for one round of an iterative multi-hop QA system. Your ONLY
responsibility is deciding what evidence must be retrieved next. Do not answer
the final question, predict later retrieval rounds, or decide how the future
answer will be used.

HARD DECISION ORDER

Follow STEP 1, STEP 2, and STEP 3 in this exact priority order. Never skip an
earlier step because a later query sounds natural.

STEP 1 — RESOLVE BLOCKING DESCRIPTIONS

Inspect the REASONING QUESTION. Find every descriptive expression whose
real-world value is not explicit and must be known before another requested
relation can be evaluated. Examples include the director of Film A, the author
of Book B, the band that released Song C, X's father, and the province where
District D is located.

If any such expression exists, ask only the direct relation that makes it
concrete. Its anchor must already be explicitly named or present in VERIFIED
RESOLUTIONS. After returning the necessary independent queries at this layer,
STOP. Do not generate a later-hop property query in the same round.

STEP 2 — DIRECT REQUIRED VALUES

Enter STEP 2 only when STEP 1 found no blocking description. For a comparison,
boolean test, ranking, count, selection, intersection, or other final
operation, ask for its direct inputs. Every entity that owns a requested value
must already be concrete.

STEP 3 — DIRECT FINAL EVIDENCE

Enter STEP 3 only when neither earlier step applies. Ask the final one-hop
information need directly.

LOCAL EXECUTABILITY TEST

For every query ask: can its answer be obtained by following exactly ONE
relation from an entity already explicitly named in the REASONING QUESTION or
known through VERIFIED RESOLUTIONS? If not, it skips a dependency and is
invalid. A grammatically complete query is not necessarily locally executable.

INVALID: When was the author of Book X born?
CORRECT: Who wrote Book X?

INVALID: In what year was the band that released Song X formed?
CORRECT: Which band released Song X?

INVALID: What is the capital of the province where District X is located?
CORRECT: Which province contains District X?

RESOLUTION TARGET

resolution_target describes exactly one unknown, distinguishes parallel
branches, and is a canonical key without a leading article. Use "director of
Film A", not "the director", "that person", or "the answer".

PARALLEL AND HISTORY RULES

Return two queries only when they are at the same dependency layer and each is
executable before either answer is known. If query 2 depends on query 1, return
only query 1. Treat VERIFIED RESOLUTIONS as known. Do not repeat PREVIOUS
QUESTIONS.

Preserve the exact relation and all identity constraints. Once the owner is
concrete, birth place != nationality, death place != death date, and workplace
!= residence. Do not ask irrelevant background questions or identity questions
about an explicitly named entity.

If no unresolved information gap remains after considering VERIFIED
RESOLUTIONS, return exactly {"questions": []}. This empty list is the only
normal completion signal. Never return an empty list because retrieval might
be difficult or because a component failed.

Otherwise return JSON only. Each item contains exactly question and
resolution_target:
{
  "questions": [
    {
      "question": "Who directed Film A?",
      "resolution_target": "director of Film A"
    }
  ]
}
"""


HOTPOTQA_QUERY_GENERATOR_PROMPT_G1 = """HotpotQA PROPERTY OWNER TEST

Before asking any property query:
1. Which entity owns the requested property?
2. Is that owner already explicitly named or verified?
3. If not, which direct relation identifies it?

An unresolved owner MUST be resolved first.

- birth date owner = the author of Book X; unresolved -> Who wrote Book X?
- formation year owner = the band that released Song X; unresolved -> Which
  band released Song X?
- capital owner = the province containing District X; unresolved -> Which
  province contains District X?

By contrast, Ghosts of Mississippi, Christopher Nolan, Harvard University,
London, and Hobart Paving are already concrete names. Do not ask what a named
entity is. A direct question such as "Who directed Ghosts of Mississippi?"
is a locally executable one-hop query.

Do not query the identity of an entity already explicitly named. "Who is
Patrick Mercer?" is invalid when Patrick Mercer already appears as a concrete
person. Preserve every nationality, date, work title, role, location, record
label, and other restriction required to identify the answer. Do not drop a
constraint merely to make a query shorter.

For two concrete owners, direct parallel value queries are valid. For example,
"Are A and B both ferns?" permits "Is A a fern?" and "Is B a fern?". But if
their owners are hidden descriptions, resolve those owners first.

COMPARISON RELATION TEST

The comparison operator determines the required value relation. Do not
substitute a correlated attribute.

- died first or died later -> death date, NOT birth date
- lived longer -> lifespan or duration, NOT birth date
- published first -> publication date, NOT publisher
"""


TWOWIKI_QUERY_GENERATOR_PROMPT_G1 = """2WikiMultiHopQA OWNER-FIRST RULES

For every requested property, identify the entity that owns it. If that owner
is descriptive rather than concrete, resolve the owner before the property.

Question: Which film has the director born later, Film A or Film B?
The birth-date owners are two unresolved directors. Return:
- Who directed Film A? / director of Film A
- Who directed Film B? / director of Film B
Do not ask their birth dates in this round.

Question: Who was born earlier, Alice or Bob?
Alice and Bob are concrete owners. Return their two birth-date queries.

If the question asks who died first or later, retrieve DEATH DATES, not birth
dates. If it asks who lived longer, retrieve LIFESPANS or durations, not birth
dates. If it asks which work was published or released first, retrieve
publication or release dates, not publisher identities.

COMPOSITE KINSHIP

Resolve exactly one dependency edge per round:
- paternal grandfather: X -> father -> father
- maternal grandfather: X -> mother -> father
- paternal grandmother: X -> father -> mother
- maternal grandmother: X -> mother -> mother
- father-in-law: normally X -> spouse -> father
- mother-in-law: normally X -> spouse -> mother
- stepmother: X -> father -> spouse

Never simplify father-in-law to father, mother-in-law to mother, or stepmother
to mother. For ambiguous relations such as sibling-in-law, never replace them
with sibling; resolve the linking relation conservatively.

Parallel branches must stay at the same layer. Two hidden directors may be
resolved together, but their later birth-date queries may not appear in the
same round.
"""


MUSIQUE_QUERY_GENERATOR_PROMPT_G1 = """MuSiQue INSIDE-OUT RULES

MuSiQue questions are nested dependency chains. Select the innermost unresolved
description whose anchor is explicit or verified, ask exactly its one direct
relation, and stop. Preserve every work type, role, date, place, organization,
episode, film, album, series, county, state, and event qualifier. Never ask an
outer relation if one of its required arguments is the still-unknown answer to
another relation; resolve that inner answer first, regardless of entity type.
Otherwise, do not decompose the argument further. Normally return one query;
two are allowed only for explicit independent branches at the same layer.
"""


DATASET_STRATEGY_PROMPTS = {
    "hotpotqa": HOTPOTQA_QUERY_GENERATOR_PROMPT_G1,
    "2wikimultihopqa": TWOWIKI_QUERY_GENERATOR_PROMPT_G1,
    "musique": MUSIQUE_QUERY_GENERATOR_PROMPT_G1,
}


def is_explicit_parallel_question(dataset: str, question: str) -> bool:
    """Detect executable parallel branches, with stricter MuSiQue rules."""

    dataset_name = normalize_curiosity_dataset_name(dataset)
    normalized = _normalize_parallel_question(question)
    if dataset_name != "musique":
        return _has_parallel_semantics(normalized)

    if re.search(r"\bboth\b.+?\band\b.+", normalized):
        return True
    if (
        re.search(r"\b(?:between|or)\b", normalized)
        and re.search(
            r"\b(?:more|less|earlier|later|older|younger|longer|shorter|"
            r"larger|smaller|same)\b",
            normalized,
        )
    ):
        return True
    return bool(
        re.search(
            r"^(?:(?:where|when|what|which|who|how)\s+)?"
            r"(?:are|were|do|did|have|has)\s+"
            r".+?\band\b.+",
            normalized,
        )
    )


def _has_parallel_semantics(normalized: str) -> bool:
    if re.search(
        r"\b(?:or|both|same|compare|more|less|earlier|later|younger|older|"
        r"longer|shorter|larger|smaller)\b",
        normalized,
    ):
        return True
    if re.search(r"\bbetween\s+.+?\s+and\s+.+", normalized):
        return True
    if re.search(r"\bwhich\s+of\s+.+?\s+and\s+.+", normalized):
        return True
    return bool(
        re.search(
            r"^(?:(?:where|when|what|which|who|how)\s+)?"
            r"(?:are|were|do|did|have|has|can|could|will|would)\s+"
            r".+?\band\b.+",
            normalized,
        )
    )


def _normalize_parallel_question(question: str) -> str:
    value = unicodedata.normalize("NFKC", str(question or "")).casefold()
    return " ".join(re.findall(r"[\w]+", value, flags=re.UNICODE))


def normalize_curiosity_dataset_name(dataset: str) -> str:
    value = str(dataset or "").strip().casefold()
    aliases = {
        "hotpot": "hotpotqa",
        "hotpotqa": "hotpotqa",
        "hotpot_qa": "hotpotqa",
        "2wiki": "2wikimultihopqa",
        "2wikimultihopqa": "2wikimultihopqa",
        "two_wiki": "2wikimultihopqa",
        "musique": "musique",
        "musique_ans": "musique",
    }
    try:
        return aliases[value]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported dataset for curiosity generation: {dataset!r}"
        ) from exc


def build_curiosity_prompt(
    *,
    original_question: str,
    verified_facts: list[dict[str, Any]],
    previous_questions: list[str],
    dataset: str,
    current_question: str | None = None,
) -> str:
    dataset_name = normalize_curiosity_dataset_name(dataset)
    return "\n\n".join(
        [
            COMMON_QUERY_GENERATOR_PROMPT_G1.strip(),
            DATASET_STRATEGY_PROMPTS[dataset_name].strip(),
            "TARGET QUESTION:\n" + str(original_question or "").strip(),
            "REASONING QUESTION:\n"
            + str(current_question or original_question or "").strip(),
            "VERIFIED RESOLUTIONS:\n"
            + json.dumps(verified_facts, ensure_ascii=False, indent=2),
            "PREVIOUS QUESTIONS:\n"
            + json.dumps(previous_questions, ensure_ascii=False, indent=2),
        ]
    )


def curiosity_prompt_version(dataset: str) -> str:
    return (
        "curiosity_generator_g1a1_query_only_inside_out_"
        + normalize_curiosity_dataset_name(dataset)
    )
