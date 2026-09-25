from agents.state import AgentState, DiscoveryScope, DiscoveryTopic, KnowledgeState, TopicMaturity, TopicStatus
from agents.role_utils import role_identity, roles_match, split_role_labels
from agents.discovery_coverage import coverage_key, gap_resolved, facts_for_gap, fact_id, active_question_matches
from agents.question_candidates import QuestionCandidate
from agents.requirements import RequirementStatus
from agents.consistency_validation import DiscoveryValidationIssue, ValidationIssueKind, ValidationResolution

# Topic ordering
TOPIC_PREREQUISITES = {
    DiscoveryTopic.USER_ROLES: [],
    DiscoveryTopic.USER_GOALS: [DiscoveryTopic.USER_ROLES],
    DiscoveryTopic.CORE_WORKFLOW: [DiscoveryTopic.USER_ROLES, DiscoveryTopic.USER_GOALS],
    DiscoveryTopic.BUSINESS_RULES: [DiscoveryTopic.CORE_WORKFLOW],
    DiscoveryTopic.CONSTRAINTS: [DiscoveryTopic.BUSINESS_RULES],
    DiscoveryTopic.MVP_SCOPE: [DiscoveryTopic.BUSINESS_RULES],
    DiscoveryTopic.EXCEPTIONS: [DiscoveryTopic.CORE_WORKFLOW, DiscoveryTopic.BUSINESS_RULES],
    DiscoveryTopic.EDGE_CASES: [DiscoveryTopic.EXCEPTIONS],
}

# Required knowledge for each topic
DISCOVERY_TASKS = {
    DiscoveryTopic.USER_ROLES: {
        "primary_users": {
            "objective": "Identify the primary users of the product.",
            "question_hint": "Ask who the primary users are."
        },
        "secondary_users": {
            "objective": "Determine whether anyone besides the primary users interacts with the product.",
            "question_hint": "Ask whether there are additional users besides the primary users."
        },
        "responsibilities": {
            "objective": "Understand the significant actions, capabilities, duties, and processes each role performs or manages in the product.",
            "question_hint": "Ask what the role does or can do in the product, including any duties or processes it manages.",
            "role_source": "all_confirmed_roles",
        },
        "permissions": {
            "objective": "Understand explicit authorization and access boundaries for each role.",
            "question_hint": "Ask about access restrictions, conditional authority, forbidden actions, or role-exclusive actions; do not simply repeat the capability question.",
            "role_source": "all_confirmed_roles",
        },
        "multiple_roles": {
            "objective": "Determine whether one person can have multiple roles.",
            "question_hint": "Ask whether a user can perform multiple roles."
        },
        "role_transitions": {
            "objective": "Determine whether users can change roles over time.",
            "question_hint": "Ask whether users can move from one role to another."
        },
    },

    DiscoveryTopic.USER_GOALS: {
        "primary_user_goals": {
            "objective": "Understand what the primary users are trying to achieve.",
            "question_hint": "Ask what the primary users want to accomplish when using the product.",
            "role_source": "primary_users",
        },
        "secondary_user_goals": {
            "objective": "Understand what secondary users (if any) are trying to achieve.",
            "question_hint": "Ask what secondary users, if any exist, want to accomplish when using the product.",
            "role_source": "secondary_users",
        },
        "success_criteria": {
            "objective": "Understand what successful completion means.",
            "question_hint": "Ask how users know they have successfully completed their goal."
        },
        "motivations": {
            "objective": "Understand why users choose this product.",
            "question_hint": "Ask why users would use this product instead of alternatives."
        },
    },

    DiscoveryTopic.CORE_WORKFLOW: {
        "trigger": {
            "objective": "Identify what starts the workflow.",
            "question_hint": "Ask what event starts the process."
        },
        "workflow_steps": {
            "objective": "Understand the stated normal interaction steps and their sequence.",
            "question_hint": "Ask the user to describe the workflow from start to finish."
        },
        "completion_condition": {  
            "objective": "Determine when the workflow is considered complete.",
            "question_hint": "Ask what conditions mark successful completion."
        },
        "downstream_dependency": {
            "objective": "Determine whether this workflow depends on something happening outside "
                        "the current app/phase before it can be considered complete (e.g. an "
                        "approval, review, or action taken elsewhere in the system).",
            "question_hint": "Ask whether anything needs to happen outside of what the user "
                            "controls — such as a review or approval — before this workflow is "
                            "considered fully complete."
        },
        "end_state": {
            "objective": "Understand the final outcome after completion.",
            "question_hint": "Ask what the final state of the workflow is."
        },
    },

    DiscoveryTopic.BUSINESS_RULES: {
        "validation_rules": {
            "objective": "Understand validation requirements.",
            "question_hint": "Ask what validations must be enforced."
        },
        "approval_rules": {
            "objective": "Understand approval requirements.",
            "question_hint": "Ask whether any actions require approval."
        },
        "eligibility_rules": {
            "objective": "Understand who is allowed to use certain functionality.",
            "question_hint": "Ask whether any eligibility requirements exist."
        },
        "limits": {
            "objective": "Understand operational limits.",
            "question_hint": "Ask whether there are limits such as amounts, quantities or durations."
        },
        "ownership_rules": {
            "objective": "Understand ownership of data and actions.",
            "question_hint": "Ask who owns or controls important resources."
        },
        "visibility_rules": {
            "objective": "Understand what each role can see.",
            "question_hint": "Ask who can view different information."
        },
    },

    DiscoveryTopic.CONSTRAINTS: {
        "legal_constraints": {
            "objective": "Identify legal constraints.",
            "question_hint": "Ask whether legal or regulatory requirements affect the product."
        },
        "business_constraints": {
            "objective": "Identify business limitations.",
            "question_hint": "Ask whether business policies limit the product."
        },
        "operational_constraints": {
            "objective": "Identify operational limitations.",
            "question_hint": "Ask whether operational limitations affect the product."
        },
        "geographic_constraints": {
            "objective": "Identify geographic limitations.",
            "question_hint": "Ask whether the product is limited to certain countries or regions."
        },
        "time_constraints": {
            "objective": "Identify time-based restrictions.",
            "question_hint": "Ask whether any deadlines or expiry periods exist."
        },
    },

    DiscoveryTopic.MVP_SCOPE: {
        "must_have_features": {
            "objective": "Identify essential MVP functionality.",
            "question_hint": "Ask which features must exist in version one."
        },
        "nice_to_have_features": {
            "objective": "Identify features that are desirable but not essential.",
            "question_hint": "Ask which features could wait until later."
        },
        "out_of_scope": {
            "objective": "Identify functionality intentionally excluded from MVP.",
            "question_hint": "Ask what should definitely not be included in version one."
        },
        "success_metrics": {
            "objective": "Understand how MVP success will be measured.",
            "question_hint": "Ask how success of the MVP will be evaluated."
        },
    },

    DiscoveryTopic.EXCEPTIONS: {
        "user_cancellations": {
            "objective": "Understand what happens when users cancel.",
            "question_hint": "Ask what should happen if a user cancels."
        },
        "timeouts": {
            "objective": "Understand timeout behaviour.",
            "question_hint": "Ask what happens if users take too long."
        },
        "invalid_actions": {
            "objective": "Understand how invalid actions are handled.",
            "question_hint": "Ask what should happen when users perform invalid actions."
        },
        "recovery": {
            "objective": "Understand how interrupted workflows recover.",
            "question_hint": "Ask how users continue after interruptions."
        },
    },

    DiscoveryTopic.EDGE_CASES: {
        "duplicate_actions": {
            "objective": "Understand duplicate action handling.",
            "question_hint": "Ask what happens if users perform the same action twice."
        },
        "boundary_conditions": {
            "objective": "Understand behavior at or beyond limits and empty/zero/capacity boundaries.",
            "question_hint": "Ask what happens at or beyond minimum, maximum, empty, or capacity boundaries, rather than re-asking the limit itself."
        },
        "simultaneous_actions": {
            "objective": "Understand concurrent user scenarios.",
            "question_hint": "Ask what happens if multiple users act at the same time."
        },
        "rare_scenarios": {
            "objective": "Identify unusual but important situations.",
            "question_hint": "Ask whether there are uncommon scenarios the product should support."
        },
    },
}

PER_ROLE_TASKS = {
    DiscoveryTopic.USER_ROLES: {"responsibilities", "permissions"},
    DiscoveryTopic.USER_GOALS: {"primary_user_goals", "secondary_user_goals"},
}

INTERNAL_ROLE_TERMS = {
    "admin", "administrator", "administrators", "support", "moderator",
    "moderators", "back office", "back-office", "internal staff",
}



def assess_topic_maturity(state: AgentState, topic: DiscoveryTopic) -> TopicMaturity:
    """Assess whether a concept can safely unlock dependent discovery."""
    known = get_known_keys(state, topic)
    if not known:
        return TopicMaturity.UNSEEN
    # Empty role sets can legitimately waive per-role fields. Completed coverage
    # must still unlock dependent topics without manufacturing placeholder facts.
    if not build_gap_info(state, topic)["missing_keys"]:
        return TopicMaturity.DECISION_READY

    if topic == DiscoveryTopic.CORE_WORKFLOW:
        required = {"workflow_steps", "completion_condition"}
        return TopicMaturity.COHERENT if required.issubset(known) else TopicMaturity.SKETCHED

    coherence_requirements = {
        DiscoveryTopic.USER_ROLES: {"primary_users", "responsibilities"},
        DiscoveryTopic.USER_GOALS: {"primary_user_goals", "success_criteria"},
        DiscoveryTopic.BUSINESS_RULES: {"validation_rules", "approval_rules"},
        DiscoveryTopic.CONSTRAINTS: {"legal_constraints"},
        DiscoveryTopic.MVP_SCOPE: {"must_have_features"},
        DiscoveryTopic.EXCEPTIONS: {"user_cancellations", "recovery"},
        DiscoveryTopic.EDGE_CASES: {"boundary_conditions"},
    }
    required = coherence_requirements.get(topic, set())
    if required and required.issubset(known):
        return TopicMaturity.COHERENT
    return TopicMaturity.MENTIONED if len(known) == 1 else TopicMaturity.SKETCHED

def get_roles_in_discovery_order(state: AgentState, topic: DiscoveryTopic) -> list[str]:
    """Keep the founder's role order and defer confirmed internal roles.

    A set plus ``sorted()`` made an administrator the first role solely because
    "administrator" alphabetically precedes "buyer".  Product discovery should
    first understand the customer journey, in the order the founder described
    its users; internal operations follow afterward.
    """
    scope = state.get("discovery_scope", DiscoveryScope.USER_APP)
    customer_roles = []
    internal_roles = []
    seen = set()
    for item in state.get("discovered_knowledge", []):
        if (
            item.topic == topic and item.scope == scope
            and item.knowledge_state == KnowledgeState.CONFIRMED
            and item.key in ("primary_users", "secondary_users")
            and not item.absence
        ):
            extracted_roles = split_role_labels(item.roles if item.roles is not None else item.value.split(","))
            for role in extracted_roles:
                role = role.strip().lower()
                if role in {"none", "none specified", "n/a"}:
                    continue
                identity = role_identity(role)
                if not role or identity in seen:
                    continue
                seen.add(identity)
                if role in INTERNAL_ROLE_TERMS or identity in INTERNAL_ROLE_TERMS:
                    internal_roles.append(role)
                else:
                    customer_roles.append(role)
    return customer_roles + internal_roles


def get_known_roles(state: AgentState, topic: DiscoveryTopic) -> set[str]:
    """Compatibility helper for callers that only need membership."""
    return set(get_roles_in_discovery_order(state, topic))


def get_confirmed_roles_for_source(state: AgentState, source_key: str) -> list[str]:
    """Return atomic confirmed USER_ROLES labels from one source key."""
    scope = state.get("discovery_scope", DiscoveryScope.USER_APP)
    roles, seen = [], set()
    for item in state.get("discovered_knowledge", []):
        if (
            item.topic != DiscoveryTopic.USER_ROLES or item.scope != scope
            or item.absence
            or item.knowledge_state != KnowledgeState.CONFIRMED or item.key != source_key
        ):
            continue
        for role in split_role_labels(item.roles if item.roles is not None else item.value.split(",")):
            if role in {"none", "none specified", "n/a"}:
                continue
            identity = role_identity(role)
            if identity and identity not in seen:
                seen.add(identity)
                roles.append(role)
    return roles

def get_known_keys(state: AgentState, topic: DiscoveryTopic):
    scope = state.get("discovery_scope", DiscoveryScope.USER_APP)
    keys = {
        item.key
        for item in state.get("discovered_knowledge", [])
        if item.topic == topic and item.scope == scope
        and item.knowledge_state == KnowledgeState.CONFIRMED
    }
    return keys


def inferred_evidence_for_gap(state: AgentState, topic: DiscoveryTopic, gap: str) -> list[str]:
    """Evidence that supports a gap but was not gathered as its direct answer."""
    scope = state.get("discovery_scope", DiscoveryScope.USER_APP)
    if "::" in gap:
        key, role = gap.split("::", 1)
        matches = lambda item: item.key == key and roles_match(item.role, role)
    else:
        matches = lambda item: item.key == gap
    return [
        item.value for item in state.get("discovered_knowledge", [])
        if item.topic == topic and item.scope == scope
        and item.knowledge_state == KnowledgeState.INFERRED
        and matches(item)
    ]


def relevant_confirmed_context(state: AgentState, topic: DiscoveryTopic, gap: str) -> list[str]:
    """Select concise confirmed context without including the target itself."""
    scope = state.get("discovery_scope", DiscoveryScope.USER_APP)
    if "::" in gap:
        target_key, target_role = gap.split("::", 1)
    else:
        target_key, target_role = gap, None
    context = []
    for item in state.get("discovered_knowledge", []):
        if item.scope != scope or item.knowledge_state != KnowledgeState.CONFIRMED:
            continue
        if item.topic == topic and item.key == target_key and (
            target_role is None or roles_match(item.role, target_role)
        ):
            continue
        is_actor_fact = item.topic == DiscoveryTopic.USER_ROLES and item.key in {"primary_users", "secondary_users"}
        is_current_topic = item.topic == topic
        is_active_role_fact = target_role is not None and roles_match(item.role, target_role)
        if is_actor_fact or is_current_topic or is_active_role_fact:
            label = f"{item.topic.value}.{item.key}"
            if item.role:
                label += f"[{item.role}]"
            context.append(f"{label}: {item.value}")
    return context[:12]


def build_gap_info(state: AgentState, topic: DiscoveryTopic):
    scope = state.get("discovery_scope", DiscoveryScope.USER_APP)
    known_keys = get_known_keys(state, topic)
    required = DISCOVERY_TASKS[topic]
    per_role_tasks = PER_ROLE_TASKS.get(topic, set())

    missing_keys = []

    for key in required.keys():
        if key in per_role_tasks:
            source_key = required[key].get("role_source")
            roles = (
                get_roles_in_discovery_order(state, DiscoveryTopic.USER_ROLES)
                if source_key == "all_confirmed_roles"
                else get_confirmed_roles_for_source(state, source_key)
            )
            if not roles:
                actor_keys = {actor_key for actor_key in ("primary_users", "secondary_users")
                              if gap_resolved(state, DiscoveryTopic.USER_ROLES, actor_key)}
                if (source_key and source_key in actor_keys) or (
                    source_key == "all_confirmed_roles"
                    and {"primary_users", "secondary_users"}.issubset(actor_keys)
                ):
                    continue
                missing_keys.append(key)
                continue
            for role in roles:
                if not gap_resolved(state, topic, f"{key}::{role}"):
                    missing_keys.append(f"{key}::{role}")
        elif not gap_resolved(state, topic, key):
            missing_keys.append(key)

    if not missing_keys:
        return {
            "current_gap": None, "current_objective": None,
            "question_hint": None, "current_role": None,
            "known_keys": known_keys, "missing_keys": [], "inferred_gap_evidence": [],
            "relevant_context": [], "known_gap_evidence": [],
        }

    gap = missing_keys[0]
    inferred_evidence = inferred_evidence_for_gap(state, topic, gap)
    known_evidence = [item.value for item in facts_for_gap(state, topic, gap)]
    context = relevant_confirmed_context(state, topic, gap)

    if "::" in gap:
        base_key, role = gap.split("::", 1)
        task = required[base_key]
        return {
            "current_gap": gap,
            "current_objective": f"{task['objective']} (specifically for the '{role}' role)",
            "question_hint": f"{task['question_hint']} Ask about the '{role}' role only — do not ask about other roles in this question.",
            "current_role": role,
            "known_keys": known_keys,
            "missing_keys": missing_keys,
            "inferred_gap_evidence": inferred_evidence,
            "known_gap_evidence": known_evidence,
            "relevant_context": context,
        }

    task = required[gap]
    return {
        "current_gap": gap,
        "current_objective": task["objective"],
        "question_hint": task["question_hint"],
        "current_role": None,
        "known_keys": known_keys,
        "missing_keys": missing_keys,
        "inferred_gap_evidence": inferred_evidence,
        "known_gap_evidence": known_evidence,
        "relevant_context": context,
    }




def _validation_plan(state: AgentState, issue: DiscoveryValidationIssue) -> dict:
    fact_index = {
        fact_id(item): item
        for item in state.get("discovered_knowledge", [])
    }
    conflicting = [
        fact_index[identity]
        for identity in issue.fact_ids
        if identity in fact_index
    ]
    if not conflicting:
        raise RuntimeError(f"Validation issue has no live conflicting facts: {issue.id}")

    anchor = conflicting[0]
    role = anchor.role
    gap = anchor.key + (f"::{role}" if role else "")
    context = []
    for item in conflicting:
        label = f"{item.topic.value}.{item.key}"
        if item.role:
            label += f"[{item.role}]"
        context.append(
            f"{label}: {item.value} (turn {item.source_turn}; evidence: {item.evidence})"
        )

    return {
        "planner_source": "validation",
        "selected_validation_issue": issue.model_dump(mode="json"),
        "selected_requirement_candidate": None,
        "selected_requirement_priority": None,
        "current_topic": anchor.topic,
        "current_gap": gap,
        "current_objective": (
            "Resolve a contradiction in confirmed product requirements by establishing "
            "which rule or statement should currently apply."
        ),
        "question_hint": (
            "Briefly present the incompatible confirmed statements and ask the user to "
            "state the current rule/decision. Do not choose a side, merge incompatible "
            "rules, or treat the clarification as complete until the user resolves it."
        ),
        "current_role": role,
        "known_keys": list(get_known_keys(state, anchor.topic)),
        "missing_keys": [],
        "inferred_gap_evidence": [],
        "known_gap_evidence": [item.value for item in conflicting],
        "relevant_context": context,
        "next_discovery_move": "resolve_contradiction",
        "awaiting_confirmation": False,
    }


def consistency_resolved(state: AgentState) -> bool:
    return not state.get("validation_blocking", False)


def _requirement_topic_unlocked(
    state: AgentState,
    candidate: QuestionCandidate,
    topic_maturity: dict,
) -> bool:
    prerequisites = TOPIC_PREREQUISITES.get(candidate.topic, [])
    for dep in prerequisites:
        maturity = topic_maturity.get(dep)
        if maturity is None:
            maturity = assess_topic_maturity(state, dep)
            topic_maturity[dep] = maturity
        if maturity not in {TopicMaturity.COHERENT, TopicMaturity.DECISION_READY}:
            return False
    return True


def _requirement_context(state: AgentState, candidate: QuestionCandidate) -> list[str]:
    known_ids = set(candidate.known_fact_ids)
    context = []
    for item in state.get("discovered_knowledge", []):
        if fact_id(item) in known_ids:
            label = f"{item.topic.value}.{item.key}"
            if item.role:
                label += f"[{item.role}]"
            context.append(f"{label}: {item.value}")
    return context[:12]


def _requirement_plan(state: AgentState, candidate: QuestionCandidate) -> dict:
    requirement = state.get("active_requirements", {}).get(candidate.requirement_key)
    if requirement is None or requirement.status != RequirementStatus.ACTIVE:
        raise RuntimeError(f"Ranked requirement candidate is stale: {candidate.requirement_key}")

    facet_map = {facet.id: facet for facet in requirement.facets}
    target_descriptions = [
        facet_map[facet_id].description
        for facet_id in candidate.target_facets
        if facet_id in facet_map
    ]
    target_labels = [
        facet_map[facet_id].label
        for facet_id in candidate.target_facets
        if facet_id in facet_map
    ]
    objective = requirement.description or requirement.label
    if target_descriptions:
        objective += " Focus on: " + "; ".join(target_descriptions)

    parent_gap = requirement.parent_gap
    return {
        "planner_source": "requirement",
        "selected_requirement_candidate": candidate.model_dump(mode="json"),
        "selected_requirement_priority": state.get("question_candidate_priority", {}).get(candidate.id),
        "selected_validation_issue": None,
        "current_topic": candidate.topic,
        "current_gap": parent_gap,
        "current_objective": objective,
        "question_hint": (
            "Ask one natural product question about this requirement. "
            + ("Cover these unresolved aspects together where natural: " + ", ".join(target_labels) + ". "
               if target_labels else "")
            + "Use existing evidence as context, not as proof that the requirement is complete."
        ),
        "current_role": None,
        "known_keys": list(get_known_keys(state, candidate.topic)),
        "missing_keys": [],
        "inferred_gap_evidence": [],
        "known_gap_evidence": [],
        "relevant_context": _requirement_context(state, candidate),
        "next_discovery_move": (
            "requirement_expansion"
            if candidate.coverage_status.value in {"KNOWN_SHALLOW", "NEEDS_EXPANSION"}
            else "requirement_discovery"
        ),
        "awaiting_confirmation": False,
    }


def active_requirements_resolved(state: AgentState) -> bool:
    scope = state.get("discovery_scope", DiscoveryScope.USER_APP)
    return not any(
        requirement.scope == scope and requirement.status == RequirementStatus.ACTIVE
        for requirement in state.get("active_requirements", {}).values()
    )


def all_required_gaps_resolved(state):
    """Legacy diagnostic only.

    Schema coverage remains useful for reporting/tests, but it no longer controls
    whether the interview continues or completes.
    """
    return all(not build_gap_info(state, topic)["missing_keys"] for topic in DiscoveryTopic)


def all_discovery_resolved(state: AgentState) -> bool:
    return (
        not state.get("open_inquiries", [])
        and active_requirements_resolved(state)
        and consistency_resolved(state)
    )


def _candidate_context(state: AgentState, candidate: QuestionCandidate) -> list[str]:
    known_ids = set(candidate.known_fact_ids)
    context = []
    for item in state.get("discovered_knowledge", []):
        if fact_id(item) not in known_ids:
            continue
        label = f"{item.topic.value}.{item.key}"
        if item.role:
            label += f"[{item.role}]"
        context.append(f"{label}: {item.value}")
    return context[:12]


def _model_plan(state: AgentState, candidate: QuestionCandidate) -> dict:
    return {
        "planner_source": "model",
        "selected_inquiry": candidate.model_dump(mode="json"),
        "selected_requirement_candidate": None,
        "selected_requirement_priority": state.get("question_candidate_priority", {}).get(candidate.id),
        "selected_validation_issue": None,
        "current_topic": candidate.topic,
        "current_gap": candidate.anchor_gap,
        "current_objective": candidate.objective,
        "question_hint": candidate.question_hint,
        "current_role": candidate.role,
        "known_keys": list(get_known_keys(state, candidate.topic)),
        "missing_keys": [],
        "inferred_gap_evidence": [],
        "known_gap_evidence": [],
        "relevant_context": _candidate_context(state, candidate),
        "next_discovery_move": "resolve_model_uncertainty",
        "awaiting_confirmation": False,
    }


def _requirement_candidate_plan(state: AgentState, candidate: QuestionCandidate) -> dict:
    plan = _requirement_plan(state, candidate)
    return {
        **plan,
        "selected_inquiry": candidate.model_dump(mode="json"),
    }


def _validation_candidate_plan(state: AgentState, candidate: QuestionCandidate) -> dict:
    issues = [
        DiscoveryValidationIssue.model_validate(item)
        for item in state.get("validation_issues", [])
    ]
    issue = next(
        (
            item for item in issues
            if item.severity.value == "BLOCKING"
            and item.resolution == ValidationResolution.USER_CLARIFICATION
        ),
        None,
    )
    if issue is None:
        raise RuntimeError("Validation inquiry is stale; no blocking clarification remains")
    plan = _validation_plan(state, issue)
    return {
        **plan,
        "selected_inquiry": candidate.model_dump(mode="json"),
    }


def _plan_candidate(state: AgentState, candidate: QuestionCandidate) -> dict:
    source = getattr(candidate.source, "value", candidate.source)
    if source == "REQUIREMENT":
        return _requirement_candidate_plan(state, candidate)
    if source == "VALIDATION":
        return _validation_candidate_plan(state, candidate)
    return _model_plan(state, candidate)


def _refresh_inquiry_frontier(state: AgentState) -> tuple[AgentState, dict]:
    """Rebuild the derived inquiry frontier when planning is invoked directly.

    The graph normally materializes these nodes before the planner. Recomputing
    here keeps resumed legacy checkpoints and focused callers safe without
    reintroducing schema traversal.
    """
    from agents.inquiries import identify_open_inquiries
    from agents.question_candidates import build_question_candidates, filter_question_candidates
    from agents.question_priority import prioritize_question_candidates

    inquiries = identify_open_inquiries(state)
    with_inquiries = {
        **state,
        "open_inquiries": [item.model_dump(mode="json") for item in inquiries],
    }
    candidates = build_question_candidates(with_inquiries)
    with_candidates = {
        **with_inquiries,
        "question_candidates": [item.model_dump(mode="json") for item in candidates],
    }
    eligible, decisions = filter_question_candidates(with_candidates, candidates)
    with_eligible = {
        **with_candidates,
        "eligible_question_candidates": [
            item.model_dump(mode="json") for item in eligible
        ],
        "question_candidate_eligibility": {
            candidate_id: decision.model_dump(mode="json")
            for candidate_id, decision in decisions.items()
        },
    }
    ranked, scores = prioritize_question_candidates(with_eligible, eligible)
    updates = {
        "open_inquiries": with_inquiries["open_inquiries"],
        "question_candidates": with_candidates["question_candidates"],
        "eligible_question_candidates": with_eligible["eligible_question_candidates"],
        "question_candidate_eligibility": with_eligible["question_candidate_eligibility"],
        "ranked_question_candidates": [
            item.model_dump(mode="json") for item in ranked
        ],
        "question_candidate_priority": {
            candidate_id: score.model_dump(mode="json")
            for candidate_id, score in scores.items()
        },
    }
    return {**with_eligible, **updates}, updates


def interview_planner_node(state: AgentState) -> dict:
    # Deliberate answer receipts are retained as diagnostics for model-level
    # inquiries, but schema coverage no longer drives routing or completion.
    coverage = dict(state.get("gap_coverage", {}))
    receipt = state.get("active_answer_result")
    if state.get("planner_source", "model") in {"model", "schema"} and receipt and active_question_matches(state):
        scope = state.get("discovery_scope", DiscoveryScope.USER_APP)
        topic, gap = state.get("current_topic"), state.get("current_gap")
        if topic is not None and gap and (
            receipt["scope"] == scope.value
            and receipt["topic"] == topic.value
            and receipt["gap"] == gap
            and receipt["source_turn"] == state.get("turn_count", 0)
        ):
            active_ids = {fact_id(item) for item in facts_for_gap(state, topic, gap)}
            if active_ids.intersection(receipt["fact_ids"]):
                coverage[coverage_key(scope, topic, gap)] = {**receipt, "status": "RESOLVED"}
                print(
                    f"MODEL INQUIRY RESOLVED: {scope.value}.{topic.value}.{gap} "
                    f"| {receipt['resolution']}"
                )

    state = {**state, "gap_coverage": coverage, "active_answer_result": None}
    scope = state.get("discovery_scope", DiscoveryScope.USER_APP)

    print("\n=== INTERVIEW PLANNER ===")
    print("Scope:", scope.value)
    print("Mode: model-driven inquiry frontier")

    validation_issues = [
        DiscoveryValidationIssue.model_validate(item)
        for item in state.get("validation_issues", [])
    ]
    structural = [
        issue for issue in validation_issues
        if issue.severity.value == "BLOCKING"
        and issue.resolution != ValidationResolution.USER_CLARIFICATION
        and issue.kind != ValidationIssueKind.RESOLVED_REQUIREMENT_BLOCKED
    ]
    if structural:
        details = "; ".join(f"{issue.kind.value}: {issue.message}" for issue in structural)
        raise RuntimeError("Discovery consistency validation failed: " + details)

    clarification = next(
        (
            issue for issue in validation_issues
            if issue.severity.value == "BLOCKING"
            and issue.resolution == ValidationResolution.USER_CLARIFICATION
        ),
        None,
    )
    if clarification is not None:
        print("Selected validation issue:", clarification.id)
        return {
            "gap_coverage": coverage,
            "active_answer_result": None,
            "selected_inquiry": None,
            **_validation_plan(state, clarification),
        }

    frontier_updates = {}
    ranked = [
        QuestionCandidate.model_validate(item)
        for item in state.get("ranked_question_candidates", [])
    ]
    if not ranked:
        state, frontier_updates = _refresh_inquiry_frontier(state)
        ranked = [
            QuestionCandidate.model_validate(item)
            for item in state.get("ranked_question_candidates", [])
        ]

    if ranked:
        selected = ranked[0]
        print("Selected inquiry:", selected.inquiry_id or selected.id)
        print("Source:", getattr(selected.source, "value", selected.source))
        if selected.requirement_id:
            print("Requirement:", selected.requirement_id)
            print("Target facets:", selected.target_facets)
        return {
            **frontier_updates,
            "gap_coverage": coverage,
            "active_answer_result": None,
            **_plan_candidate(state, selected),
        }

    open_inquiries = state.get("open_inquiries", [])
    if open_inquiries:
        raise RuntimeError(
            "Open product inquiries exist but none survived candidate eligibility/prioritization"
        )

    if not active_requirements_resolved(state):
        blocked = [
            requirement.id
            for requirement in state.get("active_requirements", {}).values()
            if requirement.scope == scope and requirement.status == RequirementStatus.ACTIVE
        ]
        raise RuntimeError(
            "Discovery has unresolved active requirements but no askable inquiry: "
            + ", ".join(blocked)
        )

    if not consistency_resolved(state):
        details = "; ".join(
            f"{issue.kind.value}: {issue.message}"
            for issue in validation_issues
            if issue.severity.value == "BLOCKING"
        )
        raise RuntimeError(
            "Discovery consistency is still unresolved; compilation is blocked: " + details
        )

    print("No material inquiry remains. Discovery is ready to compile.")
    return {
        **frontier_updates,
        "gap_coverage": coverage,
        "active_answer_result": None,
        "planner_source": "model",
        "selected_inquiry": None,
        "selected_requirement_candidate": None,
        "selected_requirement_priority": None,
        "selected_validation_issue": None,
        "current_topic": None,
        "current_gap": None,
        "current_objective": None,
        "question_hint": None,
        "current_role": None,
        "known_keys": [],
        "missing_keys": [],
        "known_gap_evidence": [],
        "inferred_gap_evidence": [],
        "relevant_context": [],
        "next_discovery_move": None,
        "awaiting_confirmation": True,
    }
