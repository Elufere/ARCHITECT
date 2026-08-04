import re
from typing import List, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel

from agents.state import (
    AgentState,
    DiscoveryScope,
    DiscoveryTopic,
    KnowledgeItem,
    TopicStatus,
)

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


def get_content_words(text: str) -> set:
    """Extracts meaningful words, ignoring tiny stop words."""
    words = re.findall(r'\b[a-z]{3,}\b', text.lower())
    return {w for w in words if w not in STOP_WORDS}


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

        if item.key in ("responsibilities", "permissions") and gap_key in ("responsibilities", "permissions"):
            role_match = False
            if item.role and normalize_role(item.role) == normalize_role(gap_role):
                role_match = True
            elif gap_role_lower in normalized_msg:
                role_match = True
                if not item.role:
                    item.role = gap_role

            if role_match and item.key != gap_key:
                item.key = gap_key
    elif item.key == "permissions" and "responsibilit" in normalized_msg:
        item.key = "responsibilities"

    # ---------------------------------------------------------
    # Layer 1: Evidence Check (Soft fallback)
    # ---------------------------------------------------------
    evidence_is_exact = normalized_evidence in normalized_msg

    if evidence_is_exact:
        if len(normalized_evidence) < 20:
            return False, f"Evidence too short ({len(normalized_evidence)} chars, min 20)"
        evidence_content_words = re.findall(r'\b[a-z]{4,}\b', normalized_evidence)
        if len(evidence_content_words) < 2:
            return False, "Evidence lacks sufficient content words"

    # ---------------------------------------------------------
    # Layer 4: Value Grounding (Substring Method - Bulletproof)
    # ---------------------------------------------------------
    value_words = get_content_words(item.value)

    if not value_words:
        return False, "Value contains no content words after removing stop words"

    for word in value_words:
        if word not in normalized_msg:
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
            clean_role = re.sub(r'[^\w\s]', '', role.lower())
            if clean_role not in normalized_msg:
                return False, f"Role '{role}' not found in user message"

    # ---------------------------------------------------------
    # HEURISTIC 3: Prevent Admin -> Primary role mapping
    # ---------------------------------------------------------
    if item.key == "primary_users" and item.roles:
        for role in item.roles:
            if any(kw in role.lower() for kw in ADMIN_KEYWORDS):
                return False, f"Admin role '{role}' cannot be mapped to primary_users"

    return True, ""


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

Topics and what they represent:
{topic_definitions}

Previously discovered knowledge:
{existing_knowledge}

Latest user response:
{user_response}

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

Each extracted fact must contain:
- topic
- key
- value
- confidence (0.5–1.0)

A single response may contain knowledge belonging to multiple topics.
Extract all relevant facts.

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

- primary_user_goals
- secondary_user_goals
- success_criteria
- motivations

Do not invent other key names.

CORE_WORKFLOW
Answers:
"What happens?"

You MUST use ONLY the following keys:

- trigger
- workflow_steps
- completion_condition

Do not invent other key names.

BUSINESS_RULES
Answers:
"What governs what happens?"

You MUST use ONLY the following keys:

- validations
- conditions
- policies

Do not invent other key names.

CONSTRAINTS
Answers:
"What limits the solution?"

You MUST use ONLY the following keys:

- legal
- technical
- operational
- cost
- performance

Do not invent other key names.

MVP_SCOPE
Answers:
"What must exist first?"

You MUST use ONLY the following keys:

- required_mvp_functionality
- out_of_scope_functionality

Do not invent other key names.

EXCEPTIONS
Answers:
"How should expected failures be handled?"

You MUST use ONLY the following keys:

- expected_error_scenarios
- failure_handling
- recovery_behavior

Do not invent other key names.

EDGE_CASES
Answers:
"What unusual situations should be considered?"

You MUST use ONLY the following keys:

- rare_scenarios
- unusual_inputs
- boundary_conditions
- duplicates
- empty_states

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
2. The evidence must be at least 15-20 characters long
3. The evidence must actually contain the information you're extracting
4. Do NOT paraphrase, summarize, or modify the quote in any way
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

    # -------------------------------------------------------------
    # HANDLE EXPLICIT "NO" ANSWERS TO UNBLOCK THE PLANNER
    # -------------------------------------------------------------
    cleaned_response = user_response.lower().strip().rstrip('.,')
    is_negative = (
        cleaned_response in {"no", "none", "nope", "n/a"}
        or "no other" in cleaned_response
        or "none besides" in cleaned_response
    )

    if is_negative and current_gap:
        if "::" in current_gap:
            gap_key, gap_role = current_gap.split("::", 1)
        else:
            gap_key, gap_role = current_gap, None

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
        topic_definitions="\n".join(
            f"{topic.value}: {description}"
            for topic, description in topic_definitions.items()
        ),
        existing_knowledge="\n".join(existing_knowledge_keys) or "None",
        user_response=user_response,
    )

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
    }