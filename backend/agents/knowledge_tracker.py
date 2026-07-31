# agents/knowledge_tracker.py

from typing import List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel

from agents.state import (
    AgentState,
    DiscoveryTopic,
    KnowledgeItem,
    TopicStatus,
)

class ExtractedKnowledge(BaseModel):
    items: List[KnowledgeItem]

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

Do NOT repeat, paraphrase, or slightly reword facts that already exist in
Previously discovered knowledge.

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

When extracting "responsibilities" or "permissions" facts for USER_ROLES,
you MUST include a "role" field naming exactly which role the fact
concerns (e.g. "buyer", "seller", "admin"). If the user describes
responsibilities for multiple roles in one response, emit ONE separate
item per role — do not merge multiple roles into a single value.

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
- business logic
- decision rules

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
- compliance constraints

MVP_SCOPE
Answers:
"What must exist first?"

You MUST use ONLY the following keys:

- required MVP functionality
- out-of-scope functionality

EXCEPTIONS
Answers:
"How should expected failures be handled?"

You MUST use ONLY the following keys:

- expected error scenarios
- failure handling
- recovery behavior

EDGE_CASES
Answers:
"What unusual situations should be considered?"

You MUST use ONLY the following keys:

- rare scenarios
- unusual inputs
- boundary conditions
- duplicates
- empty states

Key names are STRICT.

Never create new key names.

Always choose one of the allowed keys for the topic.

If none fit perfectly, choose the closest allowed key.

Example:
User response: "I want to build an escrow app."

Valid extraction:
[]

(Nothing extracted — the user described a product category but did not
state who the users are, what the workflow is, or any other fact yet.)

Example:
User response: "Buyers deposit funds, and sellers receive them after delivery is confirmed."

Valid extraction:
[
  {{"topic": "USER_ROLES", "key": "primary_users", "value": "buyer, seller"}},
  {{"topic": "CORE_WORKFLOW", "key": "workflow_steps", "value": "buyer deposits funds; funds released after delivery confirmed"}}
]

========================================
STEP 3 — Important
========================================

Your only job is to extract new knowledge.

Do NOT determine whether a topic is none,
partial, or complete.

Do NOT return topic progress.

Only return the extracted knowledge items.

========================================
OUTPUT
========================================

Return ONLY valid JSON matching the ExtractedKnowledge schema.

Do not include explanations.

Do not include markdown.

Do not return any text outside the JSON.
"""

extraction_llm = ChatOllama(
    model="llama3.1",
    temperature=0.0,
).with_structured_output(ExtractedKnowledge)

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

    existing_knowledge = "\n".join(
        [
            f"- [{item.topic.value}] {item.key}: {item.value}"
            for item in state.get("discovered_knowledge", [])
        ]
    )

    prompt = KNOWLEDGE_EXTRACTION_PROMPT.format(
        current_topic=current_topic.value if current_topic else DiscoveryTopic.USER_ROLES.value,
        topic_definitions="\n".join(
            f"{topic.value}: {description}"
            for topic, description in topic_definitions.items()
        ),
        existing_knowledge=existing_knowledge or "None",
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
    
    # Merge newly extracted knowledge
    discovered_knowledge = list(state.get("discovered_knowledge", []))

    # Track which topics received new information
    new_topics = set()

    CONFIDENCE_THRESHOLD = 0.75

    for item in extracted.items:
        if item.confidence < CONFIDENCE_THRESHOLD:
            print(f"Dropping low-confidence item: {item.topic} | {item.key} | {item.confidence:.2f}")
            continue
        item.source_turn = state.get("turn_count", 0)
        discovered_knowledge.append(item)
        new_topics.add(item.topic)

    # -----------------------------
    # Merge topic status
    # -----------------------------
    print("\n===== TOPIC STATUS MERGE =====")

    topic_status = dict(state.get("topic_status", {}))

    # Determine which topics received NEW facts
    new_topics = {
        item.topic
        for item in extracted.items
        if item.confidence >= CONFIDENCE_THRESHOLD
    }

    for topic in new_topics:

        old = topic_status.get(topic)

        print(f"\nTopic: {topic.value}")
        print(f"  Old status : {old}")

        # If we've learned anything new about a topic,
        # it is at least PARTIAL.
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