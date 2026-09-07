import re
from difflib import SequenceMatcher
from typing import List, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel

from agents.state import (
    AgentState,
    DiscoveryScope,
    DiscoveryTopic,
    KnowledgeItem,
    KnowledgeState,
    TOPIC_KEY_MAP,
    TopicStatus,
)
from agents.product_model import build_product_model
from agents.role_utils import roles_match, split_role_labels

# ──────────────────────────────────────────────
# Evidence Validation
# ──────────────────────────────────────────────

GENERIC_VALUES = {
    "user", "users", "the user", "app", "the app",
    "system", "admin", "administrator", "they", "them",
}

STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "and", "or",
    "but", "in", "on", "at", "to", "for", "of", "with", "by",
    "from", "that", "this", "it", "they", "their", "have", "has",
    "had", "be", "been", "being", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "must", "shall",
    "can", "not", "no", "yes", "so", "if", "as", "just", "also",
}

ADMIN_KEYWORDS = ["admin", "customer care", "customer support", "moderator"]

# A model often normalizes a user-written role ("administrators") to its
# singular shorthand ("admin").  This is a safe role-name normalization, not
# an inference about product behaviour.
ROLE_ALIASES = {
    "admin": {"admin", "administrator"},
    "administrator": {"admin", "administrator"},
    "moderator": {"moderator"},
}


def get_content_words(text: str) -> set:
    """Extracts meaningful words, ignoring tiny stop words."""
    words = re.findall(r'\b[a-z]{3,}\b', text.lower())
    return {w for w in words if w not in STOP_WORDS}


def _normalise_token(token: str) -> str:
    """Small, deterministic normalization for inflection and common typos."""
    token = re.sub(r"[^a-z]", "", token.lower())
    for suffix in ("ing", "ied", "ed", "es", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            return token[:-len(suffix)] + ("y" if suffix == "ied" else "")
    return token


def _value_word_is_grounded(word: str, message_words: set[str]) -> bool:
    """Allow harmless normalization without accepting unrelated concepts."""
    normalized = _normalise_token(word)
    aliases = ROLE_ALIASES.get(normalized, {normalized})
    normalized_message_words = {_normalise_token(candidate) for candidate in message_words}
    if any(alias in normalized_message_words for alias in aliases):
        return True
    # Typo tolerance is deliberately limited to longer words and a high
    # similarity threshold. Evidence still has to be a verbatim source quote.
    return len(word) >= 5 and any(
        SequenceMatcher(None, normalized, _normalise_token(candidate)).ratio() >= 0.80
        for candidate in message_words
    )


def _role_is_grounded(role: str, message_words: set[str]) -> bool:
    # Multi-word labels such as "support staff" must be grounded word by
    # word; normalising the whole phrase to "supportstaff" loses evidence.
    return all(
        _value_word_is_grounded(word, message_words)
        for word in get_content_words(role)
    )


def validate_extraction(
    item: KnowledgeItem,
    user_message: str,
    current_gap: str = None
) -> Tuple[bool, str]:
    if not item.evidence:
        return False, "Missing evidence field"

    normalized_msg = user_message.lower().strip()
    normalized_evidence = item.evidence.lower().strip()

    # ---------------------------------------------------------
    # HEURISTIC 1: Gap-Aware Key & Role Repair
    # ---------------------------------------------------------
    if current_gap and "::" in current_gap:
        gap_key, gap_role = current_gap.split("::", 1)
        gap_role_lower = gap_role.lower()
        active_roles = split_role_labels([gap_role])
        role_is_grounded = any(
            _role_is_grounded(role, get_content_words(normalized_msg))
            for role in active_roles
        )

        # Every role-scoped gap—not only responsibilities and permissions—must
        # persist the active role. Without this, a direct answer to e.g.
        # ``primary_user_goals::buyer`` is stored with no role, classified as
        # incidental knowledge, and never satisfies the buyer-specific gap.
        if item.key == gap_key:
            if item.role and any(roles_match(item.role, role) for role in active_roles):
                item.role = next(role for role in active_roles if roles_match(item.role, role))
            elif not item.role and role_is_grounded:
                item.role = active_roles[0]

        if item.key in ("responsibilities", "permissions") and gap_key in ("responsibilities", "permissions"):
            role_match = False
            if item.role and any(roles_match(item.role, role) for role in active_roles):
                role_match = True
            elif role_is_grounded or gap_role_lower in normalized_msg:
                role_match = True
                if not item.role:
                    item.role = active_roles[0]

            if role_match and item.key != gap_key:
                item.key = gap_key

        # A response to a role-responsibility/permission question may be
        # mislabelled by the model as a repeated secondary-user fact.  If that
        # fact names the role currently being discussed, preserve its grounded
        # content under the active gap instead of leaving the gap unresolved.
        if (
            item.key in ("primary_users", "secondary_users")
            and gap_key in ("responsibilities", "permissions")
            and any(
                roles_match(role, active_role)
                for role in split_role_labels(item.roles)
                for active_role in active_roles
            )
        ):
            item.key = gap_key
            item.role = next(
                active_role
                for active_role in active_roles
                if any(roles_match(role, active_role) for role in split_role_labels(item.roles))
            )
            item.roles = None
    elif item.key == "permissions" and "responsibilit" in normalized_msg:
        item.key = "responsibilities"

    # Evidence is the grounding anchor. It must remain a verbatim span from
    # the user, while the extracted value may be a normalized paraphrase.
    if normalized_evidence not in normalized_msg:
        return False, "Evidence is not a verbatim substring of the user message"

    evidence_content_words = get_content_words(normalized_evidence)
    if not evidence_content_words:
        return False, "Evidence lacks meaningful content words"

    # ---------------------------------------------------------
    # Layer 4: Value Grounding (Substring Method - Bulletproof)
    # ---------------------------------------------------------
    value_words = get_content_words(item.value)

    if not value_words:
        return False, "Value contains no content words after removing stop words"

    message_words = get_content_words(normalized_msg)
    for word in value_words:
        if not _value_word_is_grounded(word, message_words):
            return False, f"Value contains word not in user message: '{word}'"

    # ---------------------------------------------------------
    # Layer 5: Block generic values
    # ---------------------------------------------------------
    if item.value.lower().strip() in GENERIC_VALUES:
        return False, f"Generic value '{item.value}' blocked"

    # ---------------------------------------------------------
    # Layer 6: Role grounding
    # ---------------------------------------------------------
    if item.key in ("primary_users", "secondary_users") and item.roles:
        for role in item.roles:
            # Preserve separators while grounding.  ``get_content_words``
            # correctly compares a compound label such as "super-admin" as
            # the grounded words "super" and "admin"; stripping the hyphen
            # first turned it into the nonexistent token "superadmin".
            if not _role_is_grounded(role, message_words):
                return False, f"Role '{role}' not found in user message"

    # ---------------------------------------------------------
    # HEURISTIC 3: Prevent Admin -> Primary role mapping
    # ---------------------------------------------------------
    if item.key == "primary_users" and item.roles:
        for role in item.roles:
            if any(kw in role.lower() for kw in ADMIN_KEYWORDS):
                return False, f"Admin role '{role}' cannot be mapped to primary_users"

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

class ExtractedKnowledge(BaseModel):
    items: List[KnowledgeItem]


# ──────────────────────────────────────────────
# Prompt
# ──────────────────────────────────────────────

KNOWLEDGE_EXTRACTION_PROMPT = """
You are an expert Product Discovery Analyst.

Your job is to extract structured product knowledge from a user's latest response
and determine the progress of the CURRENT discovery topic.

Current topic being explored:
{current_topic}

The specific piece of information currently being asked about:
{current_gap}

If the user's response answers that specific gap, classify it under that
exact key — do not default to a different key in the same topic just
because the wording overlaps with another key's theme.

If the Current gap is written as key::role, set the item's "role" to that
one atomic role for a direct answer. Never merge several entities into one
role-specific item.

Topics and what they represent:
{topic_definitions}

Previously discovered knowledge:
{existing_knowledge}

Latest user response:
{user_response}

Correction of an earlier statement: {is_correction}

Follow these instructions carefully.

========================================
STEP 1 — Extract Knowledge
========================================

Extract ONLY facts the user EXPLICITLY stated in their latest response.

Do NOT infer facts from general domain knowledge, even if they seem
obviously true for this type of product. For example, if the user says
"I want to build an escrow app" without naming any roles, do NOT extract
primary_users as "buyer, seller" — that is a guess, not something the
user said. Wait for the user to state it.

If the user's response does not explicitly state a fact for a given key,
DO NOT include that key in your output at all. Do not write hedged,
speculative, or placeholder values like "likely X", "probably Y", or
"we'll say Z for now". Omitting the key entirely is correct and expected
when information is genuinely not yet known. An empty items list is a
valid and often correct output.

A "value" field must be a direct, concrete statement of what the user
said — never a note about uncertainty, confidence, or what might be
assumed later.

Do NOT repeat, paraphrase, or slightly reword facts that already exist in
Previously discovered knowledge.

If this is a correction, extract the corrected fact. It will replace the
previous fact with the same topic, key, and role.

Each extracted fact must contain:
- topic
- key
- value
- confidence (0.5–1.0)

A single response may contain knowledge belonging to multiple topics.
Extract all relevant facts.

Do not limit extraction to the Current gap. Extract grounded facts for other
valid keys too. Those incidental facts will be marked unconfirmed by the
system and revisited later; never omit them merely because they were not the
question asked this turn.

CRITICAL RULES FOR ROLES:
- 'admins', 'super admins', and 'customer support' are ALWAYS secondary_users.
  NEVER extract them as primary_users.
- If the user explicitly states a role or entity does NOT exist (e.g., "No,
  there are no other users"), extract the missing key with the value
  "None specified". Do not omit it.

========================================
STEP 2 — Assign Topics
========================================

Assign each fact according to its PRIMARY meaning.

Use these principles:

USER_ROLES
Answers:
"Who participates?"

You MUST use ONLY the following keys:

- primary_users
- secondary_users
- responsibilities
- permissions
- multiple_roles
- role_transitions

Do not invent any other key names.

SPECIAL RULE for primary_users and secondary_users:
In addition to "value" (a natural-language sentence), you MUST also
populate a "roles" field: a list of short role names, lowercase, each
1-2 words (e.g. ["buyer", "seller"], ["admin", "customer support"]).
Only include a role in "roles" if the user explicitly named it.
Each list entry MUST be exactly one role. For example, use
["administrator", "support staff"], never ["administrator and support staff"].

SPECIAL RULE for responsibilities and permissions:
You MUST populate a "role" field (singular) naming exactly which role
this specific fact concerns (e.g. "buyer"). If the user describes
responsibilities or permissions for multiple roles in one response,
emit ONE separate item per role — do not merge multiple roles into a
single item's value.

USER_GOALS
Answers:
"Why are they participating?"

You MUST use ONLY the following keys:

- primary_user_goals: the concrete outcome the primary user wants (WHAT they
  want to achieve — e.g. "receive the goods as agreed").
- secondary_user_goals: same, for secondary users, if any exist.
- success_criteria: the signal or condition that tells the user THEY GOT what
  they wanted (HOW they know it worked — e.g. "delivery is confirmed in the app").
- motivations: WHY they'd pick this product over alternatives — a comparative,
  competitive reason (e.g. "because it protects their money, unlike paying
  upfront with no protection").

If the user's answer explains why the product is better/safer/preferable
compared to not using it or using something else, that is motivations, even
if it also mentions security, protection, or trust.

Do not invent other key names.

CORE_WORKFLOW
Answers:
"What happens?"

You MUST use ONLY the following keys:

- trigger
- workflow_steps
- completion_condition
- downstream_dependency
- end_state

Do not invent other key names.

BUSINESS_RULES
Answers:
"What governs what happens?"

You MUST use ONLY the following keys:

- validation_rules
- approval_rules
- eligibility_rules
- limits
- ownership_rules
- visibility_rules

Do not invent other key names.

CONSTRAINTS
Answers:
"What limits the solution?"

You MUST use ONLY the following keys:

- legal_constraints
- business_constraints
- operational_constraints
- geographic_constraints
- time_constraints

Do not invent other key names.

MVP_SCOPE
Answers:
"What must exist first?"

You MUST use ONLY the following keys:

- must_have_features
- nice_to_have_features
- out_of_scope
- success_metrics

Do not invent other key names.

EXCEPTIONS
Answers:
"How should expected failures be handled?"

You MUST use ONLY the following keys:

- user_cancellations
- timeouts
- invalid_actions
- recovery

Do not invent other key names.

EDGE_CASES
Answers:
"What unusual situations should be considered?"

You MUST use ONLY the following keys:

- duplicate_actions
- boundary_conditions
- simultaneous_actions
- rare_scenarios

Do not invent other key names.

Key names are STRICT.

Never create new key names.

Always choose one of the allowed keys for the topic.

If none fit perfectly, choose the closest allowed key.

Example:
User response: "I want to build an escrow app."

Valid extraction:

{{
  "items": []
}}

(Nothing extracted — the user described a product category but did not
explicitly state who the users are, what the workflow is, or any other
fact yet.)

Example:
User response: "Buyers deposit funds, and sellers receive them after delivery is confirmed."

Valid extraction:

{{
  "items": [
    {{
      "topic": "USER_ROLES",
      "key": "primary_users",
      "value": "buyer, seller",
      "roles": ["buyer", "seller"]
    }},
    {{
      "topic": "CORE_WORKFLOW",
      "key": "workflow_steps",
      "value": "buyer deposits funds; funds released after delivery confirmed"
    }}
  ]
}}

Example:
User response: "Buyers can deposit funds and raise disputes. Sellers can deliver goods and withdraw funds."

Valid extraction:

{{
  "items": [
    {{
      "topic": "USER_ROLES",
      "key": "permissions",
      "value": "deposit funds, raise disputes",
      "role": "buyer"
    }},
    {{
      "topic": "USER_ROLES",
      "key": "permissions",
      "value": "deliver goods, withdraw funds",
      "role": "seller"
    }}
  ]
}}

========================================
STEP 3 — Important
========================================

Your only job is to extract new knowledge.

Do NOT determine whether a topic is none,
partial, or complete.

Do NOT return topic progress.

Only return the extracted knowledge items.

========================================
EVIDENCE REQUIREMENT (MANDATORY)
========================================

For EACH extracted item, you MUST include an "evidence" field containing
a VERBATIM, WORD-FOR-WORD quote from the user's message that supports
this extraction.

Rules:
1. The evidence must be an EXACT substring of the user's message
2. The evidence must actually contain the information you're extracting
3. Do NOT paraphrase, summarize, or modify the quote in any way
4. The value may normalize spelling, grammar, or tense only when the
   evidence clearly supports the same meaning.
5. If you cannot find a supporting quote, DO NOT extract that item

Example of VALID evidence:
User: "The buyers will deposit money and sellers will receive it"
Evidence for primary_users: "buyers will deposit money and sellers will receive it"

Example of INVALID evidence:
User: "I want to build an escrow app"
Evidence for primary_users: "I"  ← TOO SHORT, DOESN'T SUPPORT THE CLAIM
Evidence for primary_users: "escrow app"  ← DOESN'T NAME ANY USERS

========================================
OUTPUT
========================================

Return ONLY valid JSON matching the ExtractedKnowledge schema.

Do not include explanations.

Do not include markdown.

Do not return any text outside the JSON.
"""

extraction_llm = ChatOllama(
    model="qwen2.5:7b",
    temperature=0.0,
    timeout=30,
).with_structured_output(ExtractedKnowledge)

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

    # A concise confirmation is meaningful only when the planner presented
    # inferred evidence for the active gap. Promote that evidence instead of
    # asking the generic question again.
    if (
        state.get("conversation_intent") == "confirmation"
        and state.get("next_discovery_move") == "confirm_inference"
    ):
        inferred = inferred_items_for_gap(state)
        if inferred:
            promoted = [
                item.model_copy(update={"knowledge_state": KnowledgeState.CONFIRMED})
                if item in inferred else item
                for item in state.get("discovered_knowledge", [])
            ]
            topic_status = dict(state.get("topic_status", {}))
            if current_topic and topic_status.get(current_topic) != TopicStatus.COMPLETED:
                topic_status[current_topic] = TopicStatus.PARTIAL
            return {
                "discovered_knowledge": promoted,
                "topic_status": topic_status,
                "product_model": build_product_model(promoted, current_scope),
            }

    # -------------------------------------------------------------
    # HANDLE EXPLICIT "NO" ANSWERS TO UNBLOCK THE PLANNER
    # -------------------------------------------------------------
    cleaned_response = user_response.lower().strip().rstrip('.,')
    is_negative = (
        cleaned_response in {"no", "none", "nope", "n/a"}
        or "no other" in cleaned_response
        or "none besides" in cleaned_response
        or bool(re.fullmatch(
            r"(?:none|nothing|nope|not really)(?:\s+(?:else|more))?"
            r"(?:\s+(?:that )?i can think of)?", cleaned_response
        ))
    )

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

    if is_negative and current_gap:
        if "::" in current_gap:
            gap_key, gap_role = current_gap.split("::", 1)
        else:
            gap_key, gap_role = current_gap, None

        # Never manufacture a pseudo-key such as "workflow". The planner now
        # emits canonical keys, but this protects persisted/older sessions.
        allowed_keys = TOPIC_KEY_MAP.get(current_topic, set())
        if gap_key not in allowed_keys:
            fallback_keys = {
                DiscoveryTopic.CORE_WORKFLOW: "workflow_steps",
            }
            gap_key = fallback_keys.get(current_topic)
        if not gap_key:
            print(f">>> NEGATIVE HANDLER SKIPPED: invalid gap '{current_gap}'")
            return {}

        try:
            dummy_item = KnowledgeItem(
                topic=current_topic or DiscoveryTopic.USER_ROLES,
                scope=current_scope,
                key=gap_key,
                value="None specified",
                evidence=user_response,
                confidence=1.0,
                role=gap_role,
            )
            dummy_item.source_turn = state.get("turn_count", 0)

            discovered_knowledge = list(state.get("discovered_knowledge", []))
            discovered_knowledge.append(dummy_item)

            topic_status = dict(state.get("topic_status", {}))
            if topic_status.get(dummy_item.topic) != TopicStatus.COMPLETED:
                topic_status[dummy_item.topic] = TopicStatus.PARTIAL

            print(f"\n>>> NEGATIVE HANDLER: Marked {current_gap} as 'None specified'")
            return {
                "discovered_knowledge": discovered_knowledge,
                "topic_status": topic_status,
                "product_model": build_product_model(discovered_knowledge, current_scope),
            }
        except Exception as e:
            print(f">>> NEGATIVE HANDLER FAILED: {e}")
            # fall through to normal extraction

    print(
        f"Extracting knowledge for topic: {current_topic}, "
        f"user response: {user_response}"
    )

    topic_definitions = {
        DiscoveryTopic.USER_ROLES:
            "Who are the users? What roles exist? What permissions do they need?",
        DiscoveryTopic.USER_GOALS:
            "What are users trying to accomplish? What are their success criteria?",
        DiscoveryTopic.CORE_WORKFLOW:
            "What is the complete happy path from start to finish?",
        DiscoveryTopic.BUSINESS_RULES:
            "What rules govern the workflow? What validations exist?",
        DiscoveryTopic.CONSTRAINTS:
            "Performance, scale, legal, business, cost, or operational constraints.",
        DiscoveryTopic.MVP_SCOPE:
            "What is required for MVP? What is intentionally out of scope?",
        DiscoveryTopic.EXCEPTIONS:
            "How should expected failures or error situations behave?",
        DiscoveryTopic.EDGE_CASES:
            "Boundary conditions, unusual situations, duplicates, empty states.",
    }

    existing_knowledge_keys = [
        f"- [{item.topic.value}] {item.key}" + (f" (for: {item.role})" if item.role else "")
        for item in state.get("discovered_knowledge", [])
    ]

    prompt = KNOWLEDGE_EXTRACTION_PROMPT.format(
        current_topic=current_topic.value if current_topic else DiscoveryTopic.USER_ROLES.value,
        current_gap=current_gap or "none",
        topic_definitions="\n".join(
            f"{topic.value}: {description}"
            for topic, description in topic_definitions.items()
        ),
        existing_knowledge="\n".join(existing_knowledge_keys) or "None",
        user_response=user_response,
        is_correction="yes" if state.get("is_correction") else "no",
    )
    if recovering_prior_answer:
        prompt += """

This is a repair pass after the user said the current question was already
answered. Search the earlier answers above and extract ONLY the fact that
answers the current gap. If no earlier answer supports that gap, return an
empty items list. Do not invent a placeholder answer.
"""

    try:
        extracted = extraction_llm.invoke([SystemMessage(content=prompt)])
        print("\n========== KNOWLEDGE EXTRACTION ==========")
        for item in extracted.items:
            print(f"- {item.topic} | {item.key} | {item.confidence:.2f}")
    except Exception as e:
        print(f"Knowledge extraction failed: {e}")
        return {}

    # -----------------------------
    # Merge newly extracted knowledge
    # -----------------------------
    discovered_knowledge = list(state.get("discovered_knowledge", []))
    accepted_topics = set()

    CONFIDENCE_THRESHOLD = 0.75

    for item in extracted.items:
        # Filter 1: Confidence
        if item.confidence < CONFIDENCE_THRESHOLD:
            print(f"Dropping low-confidence item: {item.topic} | {item.key} | {item.confidence:.2f}")
            continue

        # Filter 2: Evidence validation (now gap-aware)
        is_valid, reason = validate_extraction(item, user_response, current_gap)
        if not is_valid:
            print(f"REJECTED (evidence): {item.topic} | {item.key} | {reason}")
            continue

        item.source_turn = state.get("turn_count", 0)
        item.scope = current_scope  # stamp scope for this discovery phase
        item.knowledge_state = (
            KnowledgeState.CONFIRMED
            if item_directly_answers_gap(item, current_gap)
            else KnowledgeState.INFERRED
        )
        if item.key in ("primary_users", "secondary_users") and item.roles:
            item.roles = split_role_labels(item.roles)
        if state.get("is_correction"):
            discovered_knowledge = [
                existing for existing in discovered_knowledge
                if not (
                    existing.scope == item.scope
                    and existing.topic == item.topic
                    and existing.key == item.key
                    and existing.role == item.role
                )
            ]

        # A direct answer replaces prior incidental evidence for the exact
        # same decision. This is a promotion, not a duplicate fact.
        if item.knowledge_state == KnowledgeState.CONFIRMED:
            discovered_knowledge = [
                existing for existing in discovered_knowledge
                if not (
                    existing.knowledge_state == KnowledgeState.INFERRED
                    and existing.scope == item.scope
                    and existing.topic == item.topic
                    and existing.key == item.key
                    and existing.role == item.role
                )
            ]

        if any(
            existing.scope == item.scope
            and existing.topic == item.topic
            and existing.key == item.key
            and existing.role == item.role
            and existing.value.lower() == item.value.lower()
            and existing.knowledge_state == item.knowledge_state
            for existing in discovered_knowledge
        ):
            continue
        discovered_knowledge.append(item)
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
        "topic_status": topic_status,
        "product_model": build_product_model(discovered_knowledge, current_scope),
    }
