import re
import json
from typing import Tuple, get_args

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import ValidationError, create_model

from agents.state import (
    AgentState,
    DiscoveryScope,
    DiscoveryTopic,
    KnowledgeItem,
    KnowledgeState,
    TOPIC_KEY_MAP,
    TopicStatus,
)
from agents.semantic_validation import GapAnswer, GroundingResult, GroundingResponse, GAP_INSTRUCTION, ROLE_POLICY_INSTRUCTION, GROUNDING_INSTRUCTION, category_contradiction
from agents.discovery_fields import OVERLAP_RULES, field_contract
from agents.product_model import build_product_model
from agents.topic_lifecycle import invalidate_completed_topics
from agents.absence_supersession import matching_absences, can_replace_absence, supersession_record
from agents.answer_contract import interpret_closed_answer
from agents.evidence_spans import recover_evidence_span
from agents.knowledge_duplicates import FactComparison, compare_candidate
from agents.role_utils import roles_match, split_role_labels
from agents.extraction_passes import PASSES, RawPass, OwnedFact, GoalFact, normalize_fact, canonical_role, absence_label

# ──────────────────────────────────────────────
# Evidence Validation
# ──────────────────────────────────────────────

STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "and", "or",
    "but", "in", "on", "at", "to", "for", "of", "with", "by",
    "from", "that", "this", "it", "they", "their", "have", "has",
    "had", "be", "been", "being", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "must", "shall",
    "can", "not", "no", "yes", "so", "if", "as", "just", "also",
}

def get_content_words(text: str) -> set:
    """Extracts meaningful words, ignoring tiny stop words."""
    words = re.findall(r'\b[a-z]{3,}\b', text.lower())
    return {w for w in words if w not in STOP_WORDS}


def validate_extraction(
    item: KnowledgeItem,
    user_message: str,
    current_gap: str = None
) -> Tuple[bool, str]:
    if not item.evidence or not item.evidence.strip():
        return False, "Missing evidence field"
    evidence = recover_evidence_span(item.evidence, user_message)
    if evidence is None:
        return False, "Evidence is not an exact substring with a unique source span"
    item.evidence = evidence
    if item.evidence not in user_message:
        return False, "Evidence is not an exact substring of the user message"
    if not get_content_words(item.evidence) and not item.absence and not item_directly_answers_gap(item, current_gap):
        return False, "Evidence lacks meaningful content words"
    if item.confidence < 0.75:
        return False, "Confidence below threshold"
    if not item.value or not item.value.strip():
        return False, "Missing value"
    generic_roles = {"user", "users", "people", "person", "demand_side", "supply_side"}
    if item.key in ("primary_users", "secondary_users"):
        if not item.absence and (not item.roles or any(role.strip().lower() in generic_roles for role in item.roles)):
            return False, "Actor requires a functional canonical role"
    if item.key in ("responsibilities", "permissions", "primary_user_goals", "secondary_user_goals"):
        if not item.role or item.role.strip().lower() in generic_roles:
            return False, "Fact requires a functional role owner"
    return True, ""

def item_directly_answers_gap(item: KnowledgeItem, current_gap: str | None) -> bool:
    """Whether an item answers the exact field the PM asked about this turn."""
    if not current_gap:
        return False
    if "::" not in current_gap:
        return item.key == current_gap
    gap_key, gap_role = current_gap.split("::", 1)
    return item.key == gap_key and roles_match(item.role, gap_role)


def inferred_items_for_gap(state: AgentState) -> list[KnowledgeItem]:
    """Find unconfirmed evidence supporting the currently planned gap."""
    current_gap = state.get("current_gap")
    if not current_gap:
        return []
    scope = state.get("discovery_scope", DiscoveryScope.USER_APP)
    return [
        item for item in state.get("discovered_knowledge", [])
        if item.scope == scope
        and item.knowledge_state == KnowledgeState.INFERRED
        and item_directly_answers_gap(item, current_gap)
    ]


def normalize_role(role: str) -> str:
    return re.sub(r'[^\w\s]', '', role.lower()).strip()


# ──────────────────────────────────────────────
# Schema & LLM
# ──────────────────────────────────────────────

_extraction_models = None


def extraction_models():
    """Create one structured model per pass and reuse it across turns."""
    global _extraction_models
    if _extraction_models is None:
        # Definitions, structured schema and overlapping evidence must fit together.
        base = ChatOllama(model="qwen2.5:7b", temperature=0.0, timeout=60, num_ctx=8192)
        _extraction_models = {
            name: base.with_structured_output(
                create_model(f"{name.title()}RawPass", __base__=RawPass),
                include_raw=True)
            for name, *_ in PASSES
        }
        _extraction_models.update({
            "GAP_ANSWER": base.with_structured_output(GapAnswer, include_raw=True),
            "GROUNDING": base.with_structured_output(GroundingResponse, include_raw=True),
            "FACT_COMPARISON": base.with_structured_output(FactComparison, include_raw=True),
        })
    return _extraction_models


def group_audit_evidence(payload):
    """Put each candidate beside its own source; avoid model-side quote lookups."""
    quotes = payload["evidence_quotes"]
    candidates = payload["candidates"]
    return {**{key: value for key, value in payload.items() if key not in ("evidence_quotes", "candidates")},
            "evidence_groups": [dict(evidence_id=quote_id, evidence=quote,
          candidates=[{**{key: value for key, value in candidate.items() if key != "evidence_id"},
                       "field_definition": field_contract([(DiscoveryTopic(candidate["topic"]), candidate["key"])])}
                            for candidate in candidates if str(candidate["evidence_id"]) == quote_id])
                for quote_id, quote in quotes.items()]}


def semantic_decision(name, schema, instruction, payload, allow_repair=True):
    original_payload = payload
    original_instruction = instruction
    if name == "GROUNDING":
        instruction += "\nReturn a JSON object matching this schema:\n" + json.dumps(GroundingResponse.model_json_schema())
        if payload.get("active_gap_review"):
            instruction += (
                "\nThe active_gap_review is a separate semantic interpretation of the exact question/answer. "
                "Verify it against the question and quoted response. A short answer can negate the whole "
                "asked field without repeating its name. It supplies no evidence for OTHER fields. "
                "For a supported absence, include the candidate ID in BOTH supported_ids and confirmed_absence_ids. "
                "The evidence_categories entries must use the numeric evidence IDs "
                + json.dumps(list(payload["evidence_quotes"]))
                + "; TOPIC.key names belong only in each entry's categories list."
            )
        payload = group_audit_evidence(payload)
    result = extraction_models()[name].invoke([
        SystemMessage(content=instruction),
        HumanMessage(content=json.dumps(payload, default=str)),
    ])
    raw = result.get("parsed") if isinstance(result, dict) and "parsed" in result else result
    if name == "GROUNDING" and isinstance(raw, GroundingResponse):
        raw = raw.decision()
    elif name == "GROUNDING" and isinstance(raw, dict) and isinstance(raw.get("evidence_categories"), list):
        raw = GroundingResponse.model_validate(raw).decision()
    decision = raw if isinstance(raw, schema) else schema.model_validate(raw)
    if name == "GROUNDING" and allow_repair:
        # A support ID alone is not a complete verdict: its quote category and,
        # for absence, the independent polarity verdict must also be present.
        reviewed = {candidate["id"] for candidate in original_payload["candidates"]
                    if candidate["id"] in decision.supported_ids
                    and f'{candidate["topic"]}.{candidate["key"]}' in decision.evidence_categories.get(str(candidate["evidence_id"]), [])
                    and (not candidate.get("absence") or candidate["id"] in decision.confirmed_absence_ids)}
        reviewed.update(int(key) for key in decision.rejection_reasons if key.isdigit())
        missing = [candidate for candidate in original_payload["candidates"] if candidate["id"] not in reviewed]
        if missing:
            # Repair an incomplete protocol response once, not an explicit rejection.
            # Use local IDs so the repair cannot repeat the omitted/global index error.
            mapping = {index: candidate["id"] for index, candidate in enumerate(missing)}
            quote_ids = {str(candidate["evidence_id"]) for candidate in missing}
            repair_payload = {**original_payload,
                "candidates": [{**candidate, "id": index} for index, candidate in enumerate(missing)],
                "evidence_quotes": {key: value for key, value in original_payload["evidence_quotes"].items() if key in quote_ids}}
            try:
                repaired = semantic_decision(name, schema,
                    original_instruction + "\nThe previous audit omitted a complete verdict for these candidates. "
                    "Return either support or a rejection reason for EVERY supplied ID. "
                    "Supported facts need their evidence_id category; supported absences also need confirmed_absence_ids. Do not invent IDs.",
                    repair_payload, allow_repair=False)
            except Exception as exc:
                print(f"GROUNDING REPAIR FAILED: {exc}")
                return decision
            decision.supported_ids = [index for index in decision.supported_ids if index not in mapping.values()]
            decision.supported_ids.extend(mapping[index] for index in repaired.supported_ids if index in mapping)
            decision.confirmed_absence_ids.extend(mapping[index] for index in repaired.confirmed_absence_ids if index in mapping)
            for key, reason in repaired.rejection_reasons.items():
                if key.isdigit() and int(key) in mapping:
                    decision.rejection_reasons[str(mapping[int(key)])] = reason
            # A repair only supplements categories for quotes it actually reviewed.
            for quote_id, categories in repaired.evidence_categories.items():
                if quote_id in quote_ids:
                    decision.evidence_categories[quote_id] = list(dict.fromkeys(
                        decision.evidence_categories.get(quote_id, []) + categories))
    return decision

def answer_context(state):
    question = next((message.content for message in reversed(state.get("messages", [])[:-1])
                     if isinstance(message, AIMessage)), "")
    return dict(question=question, topic=state.get("current_topic"),
                gap=state.get("current_gap"), scope=state.get("discovery_scope"))


def extract_gap_absence(user_response, state, scope):
    gap = state.get("current_gap")
    topic = state.get("current_topic")
    if not gap or not topic:
        return None
    key, _, role = gap.partition("::")
    if key not in TOPIC_KEY_MAP.get(topic, set()):
        return None
    # Owner-specific answers must never become unowned blanket denials.
    from agents.interview_planner import PER_ROLE_TASKS, get_roles_in_discovery_order
    if key in PER_ROLE_TASKS.get(topic, set()) and not role:
        return None
    if role and key not in PER_ROLE_TASKS.get(topic, set()):
        return None
    if role and role not in get_roles_in_discovery_order(state, DiscoveryTopic.USER_ROLES):
        return None
    existing = [item.model_dump() for item in state.get("discovered_knowledge", [])
                if item.scope == scope and item.topic == topic and item.key == key
                and item.role == (role or None)]
    policy_field = topic == DiscoveryTopic.USER_ROLES and key in ("multiple_roles", "role_transitions")
    try:
        decision = semantic_decision("GAP_ANSWER", GapAnswer,
            (ROLE_POLICY_INSTRUCTION if policy_field else GAP_INSTRUCTION)
            + "\nActive field definition:\n" + field_contract([(topic, key)]),
            dict(**answer_context(state), latest_response=user_response, existing=existing))
        evidence = recover_evidence_span(decision.evidence, user_response)
        if (decision.resolution == "unresolved" or decision.confidence < 0.75
                or evidence is None):
            return None
        decision.evidence = evidence
        if decision.resolution == "policy":
            if not policy_field or not decision.value or not decision.value.strip() or absence_label(decision.value):
                return None
            return KnowledgeItem(topic=topic, scope=scope, key=key,
                value=decision.value.strip(), evidence=decision.evidence,
                confidence=decision.confidence, knowledge_state=KnowledgeState.CONFIRMED,
                source_turn=state.get("turn_count", 0))
        return KnowledgeItem(topic=topic, scope=scope, key=key, role=role or None,
            roles=[] if key in ("primary_users", "secondary_users") else None,
            value="none" if decision.resolution == "none" else "not applicable",
            absence=decision.resolution, evidence=decision.evidence,
            confidence=decision.confidence, knowledge_state=KnowledgeState.CONFIRMED,
            source_turn=state.get("turn_count", 0))
    except Exception as exc:
        print(f"GAP ANSWER FAILED: {exc}")
        return None


def confirmed_actor_context(state, scope):
    """Retain the source of actor identities, not just their canonical IDs.

    Descriptions can explain that several names are capacities of one actor.
    They are identity context only, never evidence of this turn's new actions.
    """
    return [dict(key=item.key, roles=item.roles, aliases=item.aliases,
                 value=item.value, evidence=item.evidence, source_turn=item.source_turn)
            for item in state.get("discovered_knowledge", [])
            if item.scope == scope and item.topic == DiscoveryTopic.USER_ROLES
            and item.key in ("primary_users", "secondary_users", "multiple_roles", "role_transitions")
            and item.knowledge_state == KnowledgeState.CONFIRMED and not item.absence]


def ground_items(items, user_response, state, active_gap_review=None):
    """Audit the asked-for answer independently of incidental extracted claims.

    A malformed/omitted verdict in a large cross-topic batch must not erase a
    valid answer to the current question. Both batches still need grounding.
    """
    focused = [item for item in items if item.topic == state.get("current_topic")
               and item_directly_answers_gap(item, state.get("current_gap"))]
    remaining = [item for item in items if item not in focused]
    if not focused or not remaining:
        return ground_batch(items, user_response, state, active_gap_review)
    print(f"GROUNDING BATCHES: active_gap={state.get('current_gap')} "
          f"direct_candidates={len(focused)} incidental_candidates={len(remaining)}")
    accepted = ground_batch(focused, user_response, state, active_gap_review)
    # Only grounded actor declarations may provide new identity context to the
    # remaining batch. A rejected actor proposal is not an established owner.
    context_state = {**state, "discovered_knowledge": [
        *state.get("discovered_knowledge", []),
        *(item for item in accepted if item.topic == DiscoveryTopic.USER_ROLES
          and item.key in ("primary_users", "secondary_users"))]}
    accepted += ground_batch(remaining, user_response, context_state)
    return [item for item in items if item in accepted]


def ground_batch(items, user_response, state, active_gap_review=None):
    eligible = []
    for item in items:
        reason = category_contradiction(item.key, item.evidence)
        if reason:
            print(f"CATEGORY REJECTED: {item.topic.value}.{item.key} owner={item.role} | {reason}")
        else:
            eligible.append(item)
    items = eligible
    if not items:
        return []
    # Intern repeated quotes instead of serializing the entire source sentence
    # in every candidate. Long overlap batches otherwise crowd out audit rules.
    # Preserve source order independently of which extractor emitted a quote first.
    # Actor declarations may quote later capability sentences; that must not reorder
    # the source narrative presented to the semantic audit.
    quotes = sorted({item.evidence for item in items}, key=lambda quote: (user_response.find(quote), len(quote), quote))
    candidates = []
    for index, item in enumerate(items):
        candidate = dict(id=index, topic=item.topic.value, key=item.key,
                         value=item.value, scope=item.scope.value,
                         evidence_id=quotes.index(item.evidence))
        if item.key in ("primary_users", "secondary_users"):
            candidate.update(kind="actor_declaration", roles=item.roles or [])
            if item.aliases:
                candidate["aliases"] = item.aliases
        elif item.role:
            candidate["role"] = item.role
        absence = item.absence or absence_label(item.value)
        if absence:
            candidate["absence"] = absence
        if item.knowledge_state == KnowledgeState.INFERRED:
            candidate["knowledge_state"] = item.knowledge_state.value
        candidates.append(candidate)
    scope = state.get("discovery_scope", DiscoveryScope.USER_APP)
    actors = [item for item in [*state.get("discovered_knowledge", []), *items]
              if item.scope == scope and item.knowledge_state == KnowledgeState.CONFIRMED
              and item.key in ("primary_users", "secondary_users") and not item.absence]
    actor_context = {key: sorted({r for item in actors if item.key == key for r in item.roles or []})
                     for key in ("primary_users", "secondary_users")}
    try:
        decision = semantic_decision("GROUNDING", GroundingResult,
            GROUNDING_INSTRUCTION + "\n" + OVERLAP_RULES + "\nField definitions:\n"
            + field_contract((item.topic, item.key) for item in items),
              dict(**answer_context(state), latest_response=user_response,
                   **({"active_gap_review": active_gap_review.model_dump(mode="json")} if active_gap_review else {}),
                   actor_classification=actor_context,
                   confirmed_actor_context=confirmed_actor_context(state, scope),
                   candidates=candidates, evidence_quotes={str(i): quote for i, quote in enumerate(quotes)}))
        supported = set(decision.supported_ids)
        confirmed_absences = set(decision.confirmed_absence_ids)
        accepted = []
        for index, item in enumerate(items):
            absence = item.absence or absence_label(item.value)
            categories = decision.evidence_categories.get(str(candidates[index]["evidence_id"]), [])
            category_supported = f"{item.topic.value}.{item.key}" in categories
            if index not in supported or not category_supported or (absence and index not in confirmed_absences):
                identity = f"actors={item.roles}" if item.key in ("primary_users", "secondary_users") else f"owner={item.role or 'not required'}"
                reason = decision.rejection_reasons.get(str(index),
                    "No explicit whole-field absence verified" if absence else
                    "Category not supported by own quote" if not category_supported else
                    "Auditor did not support this candidate")
                print(f"GROUNDING REJECTED: id={index} {item.topic.value}.{item.key} {identity} | {reason} | evidence={item.evidence!r}")
                continue
            if absence and not item.absence:
                item = item.model_copy(update={"absence": absence,
                    "value": "none" if absence == "none" else "not applicable"})
            accepted.append(item)
        return accepted
    except Exception as exc:
        # Fail closed: a validator outage must not persist unsupported facts.
        print(f"GROUNDING FAILED: {exc}")
        return []


def extract_passes(user_response: str, state: AgentState,
                   scope: DiscoveryScope) -> list[KnowledgeItem]:
    accepted = []
    stored_actors = [item for item in state.get("discovered_knowledge", [])
                     if item.scope == scope and item.knowledge_state == KnowledgeState.CONFIRMED
                     and item.key in ("primary_users", "secondary_users")]
    for name, schema, topic, instruction in PASSES:
        actors = stored_actors + [item for item in accepted
                                  if item.key in ("primary_users", "secondary_users")
                                  and item.knowledge_state == KnowledgeState.CONFIRMED]
        primary = [canonical_role(r) for item in actors if item.key == "primary_users" for r in item.roles or []]
        secondary = [canonical_role(r) for item in actors if item.key == "secondary_users" for r in item.roles or []]
        allowed_remaining_keys = {
            remaining.value: sorted(TOPIC_KEY_MAP[remaining])
            for remaining in (DiscoveryTopic.BUSINESS_RULES, DiscoveryTopic.CONSTRAINTS,
                              DiscoveryTopic.MVP_SCOPE, DiscoveryTopic.EXCEPTIONS,
                              DiscoveryTopic.EDGE_CASES)
        }
        definitions = field_contract(
            [(topic, key) for key in get_args(schema.model_fields["key"].annotation)]
            if topic else [(DiscoveryTopic(t), key) for t, keys in allowed_remaining_keys.items() for key in keys]
        )
        prompt = f"""Extract {name} facts from the latest user response using JSON with an items array.
Each item must match {schema.__name__}: {json.dumps(schema.model_json_schema(), default=str)}
Current scope: {scope.value}. Extract only facts about this scope; do not import
facts explicitly assigned to another app/dashboard into this scope.
Current topic: {state.get('current_topic').value if state.get('current_topic') else 'NONE - INITIAL DISCOVERY'}
Current gap: {state.get('current_gap') or 'none'} (interview focus only; never determines semantic provenance)
Last question (context for short answers/pronouns only, never evidence): {answer_context(state)['question']}
CONFIRMED means explicitly stated by the user. INFERRED means genuinely deduced.
Category admission for this {name} pass happens BEFORE generating candidates:
For each exact statement ask: "Does this exact statement explicitly express a
fact belonging to MY category?" If not, emit no candidate for that statement.
First identify the meaning explicitly expressed, then check this pass's field
definitions; do not reinterpret the sentence to fit an available field.
Current topic, current gap, question wording, product domain, and previously
inferred semantics cannot justify category membership, even for INFERRED items.
Question context may resolve references or the proposition explicitly affirmed
or denied by a short answer such as yes/no. It cannot supply an action, outcome,
boundary, process, or rule that the response does not assert. A question listing
several possibilities does not make an ambiguous answer confirm all of them.
Actor context resolves identity only; mentioning a known actor does not restate
its membership or supply its actions. A motivation alone does not imply duties,
authorization, process stages, governing rules, exceptions, or new actor membership.
Reuse a quote across passes only when it explicitly expresses each category's
meaning independently; related concepts and plausible implications do not qualify.
Do not make every statement belong somewhere. Return {{"items": []}} when this
pass's category is unsupported. Do not emit speculative candidates for the final
auditor to sort out; its later checks do not replace this admission decision.
EVIDENCE MUST BE copied directly from the latest user response as one exact contiguous, case-sensitive substring sufficient to support the entire fact.
The candidate's OWN quote must support its value and category. A fact stated in
another sentence cannot rescue a wrong quote. Select the sentence that actually
contains this actor's action/outcome; use a longer contiguous quote for pronouns.
Do not reconstruct, summarize, remove words from the middle, append punctuation that changes the substring, or quote prompt examples.
If a short exact quote is unavailable, copy the entire supporting sentence.
Prompt examples and actor context are not new facts. Actor-only answers contain
no workflow or goal. Never infer actions from instructions or earlier turns.
Return an empty array when the latest response does not support this category.
Confirmed primary roles: {', '.join(primary) or 'none'}
Confirmed secondary roles: {', '.join(secondary) or 'none'}
Canonical actor IDs: {json.dumps(primary + secondary)}
Confirmed actor declarations and role relationships (identity context only):
{json.dumps(confirmed_actor_context({**state, "discovered_knowledge": [*state.get("discovered_knowledge", []), *accepted]}, scope), default=str)}
Use these declarations to resolve actor names and transaction-specific capacities.
Preserve the capacity and its conditions in the value; do not turn it into a new
actor when the user has already identified it as a capacity of an existing actor.
Do not extract old actions from this context. New facts still need their OWN quote
from the latest response; do not assign ownership solely from the active gap.
Actor declarations introduce IDs in roles (for example patient); their key is the
classification (for example primary_users), never an actor ID. They need no role owner.
Only actor-owned facts use role, matching a listed canonical ID exactly.
{instruction}
FIELD DEFINITIONS:
{definitions}
{OVERLAP_RULES}
Explicit whole-field absence may use absence="none", value="none" (or
absence="not_applicable", value="not applicable"). Never infer absence from silence.
An explicit prohibition or excluded feature is substantive knowledge, not absence
of rules or exclusions. Preserve it in value with its original meaning.
"""
        if name in ("WORKFLOW", "RULES"):
            prompt += (
                "\nScope boundary: An explicitly stated dependency, handoff, or business rule "
                "linking the current application's process to another surface is knowledge "
                "about the current process. Preserve its participant and stated surface "
                "without declaring that participant a current-app user. No registered "
                "current-app actor is required for an external process participant. "
                "Do not invent its surface when unspecified, and do not import unrelated "
                "details of another application. Apply the existing field definitions.\n"
            )
        if name == "RULES":
            prompt += ("\nYou may ONLY use the following topic/key combinations:\n"
                       + json.dumps(allowed_remaining_keys, indent=2) + "\n")
        prompt += "\nFinal check: Apply each field definition independently. Reuse evidence where supported; never fill a field merely because it exists.\n"
        print(f"========== {name} EXTRACTION RAW ==========")
        try:
            result = extraction_models()[name].invoke([
                SystemMessage(content=prompt), HumanMessage(content=user_response)])
            raw = result.get("parsed") if isinstance(result, dict) and "parsed" in result else result
            if raw is None:
                raise ValueError(str(result.get("parsing_error", "No parsed output")))
            payload = raw if isinstance(raw, RawPass) else RawPass.model_validate(raw)
            print(payload.model_dump())
        except Exception as exc:
            print(f"{name} EXTRACTION FAILED: {exc}")
            continue
        pass_accepted = []
        repair_candidates = []
        for raw_fact in payload.items:
            try:
                fact = schema.model_validate(raw_fact)
                item = normalize_fact(fact, topic, scope, state.get("turn_count", 0))
                if name == "ACTOR" and fact.key in ("multiple_roles", "role_transitions") and fact.absence == "none":
                    # Reinterpret absence-shaped role policies instead of losing
                    # their conditions. The result still requires normal grounding.
                    item = extract_gap_absence(user_response, {
                        **state, "current_topic": DiscoveryTopic.USER_ROLES,
                        "current_gap": fact.key}, scope)
                    if item is None:
                        raise ValueError("Role-policy absence lacks a supported interpretation")
                valid, reason = validate_extraction(item, user_response, state.get("current_gap"))
                if not valid:
                    raise ValueError(reason)
                if isinstance(fact, (OwnedFact, GoalFact)) and item.role:
                    if item.role not in primary + secondary:
                        raise ValueError("Owner is not a confirmed actor")
                if isinstance(fact, GoalFact) and item.key in ("primary_user_goals", "secondary_user_goals"):
                    expected = primary if item.key == "primary_user_goals" else secondary
                    if item.role not in expected:
                        raise ValueError("Goal owner does not match confirmed actor classification")
                pass_accepted.append(item)
            except (ValidationError, ValueError) as exc:
                print(f"{name} REJECTED: {exc}")
                if name == "GOAL":
                    repair_candidates.append({"item": raw_fact, "error": str(exc)})
        if repair_candidates:
            # One repair call, only for rejected goals. Never guess an owner from
            # the active gap or discard valid siblings. Repaired items still go
            # through the same validation and the downstream grounding audit.
            repair_prompt = prompt + "\nRepair ONLY these rejected goal candidates:\n" + json.dumps(repair_candidates)
            repair_prompt += (
                "\nReturn corrected items only. Include role for every primary/secondary goal, "
                "using the confirmed actor registry and source/question to resolve ownership. "
                "Do not assign the active gap's owner automatically. Copy evidence exactly "
                "from the latest response. Omit candidates whose outcome or owner is unsupported."
            )
            try:
                result = extraction_models()[name].invoke([
                    SystemMessage(content=repair_prompt), HumanMessage(content=user_response)])
                raw = result.get("parsed") if isinstance(result, dict) and "parsed" in result else result
                repaired = raw if isinstance(raw, RawPass) else RawPass.model_validate(raw)
                for raw_fact in repaired.items:
                    try:
                        fact = schema.model_validate(raw_fact)
                        item = normalize_fact(fact, topic, scope, state.get("turn_count", 0))
                        valid, reason = validate_extraction(item, user_response, state.get("current_gap"))
                        if not valid:
                            raise ValueError(reason)
                        if item.role and item.role not in primary + secondary:
                            raise ValueError("Owner is not a confirmed actor")
                        if item.key in ("primary_user_goals", "secondary_user_goals"):
                            expected = primary if item.key == "primary_user_goals" else secondary
                            if item.role not in expected:
                                raise ValueError("Goal owner does not match confirmed actor classification")
                        if item not in pass_accepted:
                            pass_accepted.append(item)
                    except (ValidationError, ValueError) as exc:
                        print(f"GOAL REPAIR REJECTED: {exc}")
            except Exception as exc:
                print(f"GOAL REPAIR FAILED: {exc}")
        print(f"{name} ACCEPTED {pass_accepted}")
        accepted.extend(pass_accepted)
    return accepted

# ──────────────────────────────────────────────
# Node
# ──────────────────────────────────────────────

def knowledge_tracker_node(state: AgentState) -> dict:
    """
    Extracts structured product knowledge from the latest user response.
    """
    print(">>> EXTRACT")

    print("\n===== KNOWLEDGE TRACKER INPUT =====")
    print("Current topic:", state.get("current_topic"))
    print("Turn:", state.get("turn_count"))
    print("Incoming topic status:")
    for topic in DiscoveryTopic:
        print(f"  {topic.value}: {state.get('topic_status', {}).get(topic)}")
    print("Knowledge count:", len(state.get("discovered_knowledge", [])))
    print("===================================\n")

    messages = state["messages"]

    # Nothing to process
    if not messages:
        return {}

    # Only extract after a HUMAN response
    if not isinstance(messages[-1], HumanMessage):
        return {}

    user_response = messages[-1].content
    current_topic = state.get("current_topic")
    current_gap = state.get("current_gap")
    current_scope = state.get("discovery_scope", DiscoveryScope.USER_APP)
    superseded_knowledge = list(state.get("superseded_knowledge", []))

    # A concise confirmation is meaningful only when the planner presented
    # inferred evidence for the active gap. Promote that evidence instead of
    # asking the generic question again.
    if (
        state.get("conversation_intent") == "confirmation"
        and state.get("next_discovery_move") == "confirm_inference"
    ):
        inferred = inferred_items_for_gap(state)
        if inferred:
            promoted, committed_promotions, replaced_absences = [], [], []
            for item in state.get("discovered_knowledge", []):
                if item not in inferred:
                    promoted.append(item)
                    continue
                replacement = item.model_copy(update={"knowledge_state": KnowledgeState.CONFIRMED})
                previous = matching_absences(item, state.get("discovered_knowledge", []))
                if previous:
                    replacement = replacement.model_copy(update={
                        "evidence": user_response, "source_turn": state.get("turn_count", 0),
                        "source_question": answer_context(state)["question"] or None})
                    if not can_replace_absence(replacement, previous, user_response,
                                               answer_context(state), semantic_decision):
                        promoted.append(item)
                        continue
                    replaced_absences.extend(previous)
                    superseded_knowledge.extend(supersession_record(old, replacement) for old in previous)
                promoted.append(replacement)
                committed_promotions.append(replacement)
            promoted = [item for item in promoted if item not in replaced_absences]
            print("\n===== TOPIC STATUS MERGE =====")
            topic_status = dict(state.get("topic_status", {}))
            if current_topic and topic_status.get(current_topic) != TopicStatus.COMPLETED:
                topic_status[current_topic] = TopicStatus.PARTIAL
            topic_status = invalidate_completed_topics(
                state, promoted,
                committed_promotions, topic_status)
            return {
                "discovered_knowledge": promoted,
                "superseded_knowledge": superseded_knowledge,
                "topic_status": topic_status,
                "product_model": build_product_model(promoted, current_scope),
            }

    # When the user says a question was already answered, recover the answer
    # from earlier human turns rather than pretending the gap is still blank.
    # We only ask the extractor to recover the current gap, and retain the
    # original quoted text as evidence for the normal validation pipeline.
    recovering_prior_answer = state.get("conversation_intent") == "objection"
    if recovering_prior_answer:
        earlier_answers = [
            message.content for message in messages[:-1]
            if isinstance(message, HumanMessage)
        ]
        if earlier_answers:
            user_response = "\n".join(earlier_answers)

    print(f"Extracting knowledge for topic: {current_topic}, user response: {user_response}")
    closed_answer = None if recovering_prior_answer else interpret_closed_answer(state)
    answer_followup = None
    if closed_answer is not None:
        # The exact generated question defines the choice's meaning. No model
        # inference is involved, and no free-form answer takes this path.
        extracted_items, followup = closed_answer
        print(f"CONTROLLED ANSWER: {current_gap} | "
              f"{'additional actor names needed' if followup else 'explicit absence recorded'}")
        if followup:
            answer_followup = dict(gap=current_gap, scope=current_scope, question=followup)
    else:
        extracted_items = extract_passes(user_response, state, current_scope)
        # Never reinterpret historical denials as this turn's answer.
        absence = None
        if not recovering_prior_answer:
            absence = extract_gap_absence(user_response, state, current_scope)
            if absence:
                extracted_items.append(absence)
        extracted_items = ground_items(extracted_items, user_response, state, absence)
    discovered_knowledge = list(state.get("discovered_knowledge", []))
    if current_gap:
        print(f"ACTIVE ANSWER: gap={current_gap} accepted_facts="
              f"{sum(item.topic == current_topic and item_directly_answers_gap(item, current_gap) for item in extracted_items)}")
    accepted_topics = set()
    committed_items = []
    blocked_actor_roles = set()
    for item in extracted_items:
        if not recovering_prior_answer and not item.source_question:
            item = item.model_copy(update={"source_question": answer_context(state)["question"] or None})
        print("STATE TRACE:", item.topic, item.key, "| LLM state:",
              item.knowledge_state, "| current_gap:", current_gap,
              "| directly_answers_gap:", item_directly_answers_gap(item, current_gap))
        if (item.scope, item.role) in blocked_actor_roles and not any(
            old.scope == item.scope and old.knowledge_state == KnowledgeState.CONFIRMED
            and old.key in ("primary_users", "secondary_users")
            and item.role in (old.roles or []) for old in discovered_knowledge
        ):
            continue  # Do not commit owned facts through a rejected absence replacement.
        # Review each candidate against the turn-start decisions, even if an
        # earlier sibling has already replaced one during this same merge.
        previous_absences = matching_absences(item, state.get("discovered_knowledge", []))
        if previous_absences and item.knowledge_state == KnowledgeState.CONFIRMED:
            label = item.absence or absence_label(item.value)
            if label and all((old.absence or absence_label(old.value)) == label for old in previous_absences):
                continue  # A repeated absence keeps its original provenance.
            if not can_replace_absence(item, previous_absences, messages[-1].content,
                                       answer_context(state), semantic_decision):
                print(f"ABSENCE PRESERVED: {item.topic.value}.{item.key} scope={item.scope.value} owner={item.role}")
                if item.key in ("primary_users", "secondary_users"):
                    blocked_actor_roles.update((item.scope, role) for role in item.roles or [])
                continue
        relation, matched = compare_candidate(item, discovered_knowledge, semantic_decision)
        print(f"KNOWLEDGE COMPARISON: {item.topic.value}.{item.key} | {relation}")
        if relation in ("exact_duplicate", "semantic_duplicate", "already_refined"):
            continue
        if relation == "refinement":
            discovered_knowledge.remove(matched)
            superseded_knowledge.append(supersession_record(matched, item))
        # Absence replaces old values; new positive facts replace absence.
        discovered_knowledge = [existing for existing in discovered_knowledge
            if not (item.knowledge_state == KnowledgeState.CONFIRMED
                    and existing.scope == item.scope and existing.topic == item.topic
                    and existing.key == item.key and existing.role == item.role
                    and (item.absence or existing.absence or existing in previous_absences))]
        if state.get("is_correction"):
            discovered_knowledge = [existing for existing in discovered_knowledge
                if not (existing.scope == item.scope and existing.topic == item.topic
                        and existing.key == item.key and existing.role == item.role
                        and not (existing.absence or absence_label(existing.value)))]
        if item.knowledge_state == KnowledgeState.CONFIRMED:
            discovered_knowledge = [existing for existing in discovered_knowledge
                if not (existing.knowledge_state == KnowledgeState.INFERRED
                        and existing.scope == item.scope and existing.topic == item.topic
                        and existing.key == item.key and existing.role == item.role)]
        discovered_knowledge.append(item)
        if item.knowledge_state == KnowledgeState.CONFIRMED:
            superseded_knowledge.extend(supersession_record(old, item) for old in previous_absences)
        committed_items.append(item)
        accepted_topics.add(item.topic)
    # -----------------------------
    # Merge topic status
    # -----------------------------
    print("\n===== TOPIC STATUS MERGE =====")

    topic_status = dict(state.get("topic_status", {}))

    for topic in accepted_topics:
        old = topic_status.get(topic)

        print(f"\nTopic: {topic.value}")
        print(f"  Old status : {old}")

        if old != TopicStatus.COMPLETED:
            topic_status[topic] = TopicStatus.PARTIAL

        print(f"  New status : {topic_status.get(topic)}")

    topic_status = invalidate_completed_topics(
        state, discovered_knowledge, committed_items, topic_status)

    print("================================\n")

    print("\n===== KNOWLEDGE TRACKER OUTPUT =====")
    print("Current topic:", current_topic)
    print("Outgoing topic status:")

    for topic in DiscoveryTopic:
        print(f"  {topic.value}: {topic_status.get(topic)}")

    print("Knowledge count:", len(discovered_knowledge))
    print("====================================\n")

    return {
        "discovered_knowledge": discovered_knowledge,
        "superseded_knowledge": superseded_knowledge,
        "answer_followup": answer_followup,
        "topic_status": topic_status,
        "product_model": build_product_model(discovered_knowledge, current_scope),
    }
