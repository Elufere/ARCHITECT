from agents.state import AgentState, DiscoveryScope, DiscoveryTopic, TopicStatus

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
            "objective": "Understand what responsibilities each user has.",
            "question_hint": "Ask what each user is responsible for."
        },
        "permissions": {
            "objective": "Understand what actions each role is allowed to perform.",
            "question_hint": "Ask what each user is allowed to do."
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
            "question_hint": "Ask what the primary users want to accomplish when using the product."
        },
        "secondary_user_goals": {
            "objective": "Understand what secondary users (if any) are trying to achieve.",
            "question_hint": "Ask what secondary users, if any exist, want to accomplish when using the product."
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
            "objective": "Understand the complete happy path.",
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
            "objective": "Understand minimum and maximum limits.",
            "question_hint": "Ask about minimum and maximum supported values."
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
}

def get_known_roles(state: AgentState, topic: DiscoveryTopic) -> set[str]:
    scope = state.get("discovery_scope", DiscoveryScope.USER_APP)
    roles = set()
    for item in state.get("discovered_knowledge", []):
        if item.topic == topic and item.scope == scope and item.key in ("primary_users", "secondary_users"):
            if item.roles:
                for r in item.roles:
                    r = r.strip().lower()
                    if r:
                        roles.add(r)
    return roles


def get_known_keys(state: AgentState, topic: DiscoveryTopic):
    scope = state.get("discovery_scope", DiscoveryScope.USER_APP)
    return {
        item.key
        for item in state.get("discovered_knowledge", [])
        if item.topic == topic and item.scope == scope
    }

def build_gap_info(state: AgentState, topic: DiscoveryTopic):
    scope = state.get("discovery_scope", DiscoveryScope.USER_APP)
    known_keys = get_known_keys(state, topic)
    required = DISCOVERY_TASKS[topic]
    per_role_tasks = PER_ROLE_TASKS.get(topic, set())
    roles = get_known_roles(state, topic) if per_role_tasks else set()

    missing_keys = []

    for key in required.keys():
        if key in per_role_tasks:
            if not roles:
                missing_keys.append(key)
                continue
            known_roles_for_key = {
                item.role
                for item in state.get("discovered_knowledge", [])
                if item.topic == topic and item.key == key
                and item.role and item.scope == scope   
            }
            for role in sorted(roles - known_roles_for_key):
                missing_keys.append(f"{key}::{role}")
        elif key not in known_keys:
            missing_keys.append(key)

    if not missing_keys:
        return {
            "current_gap": None, "current_objective": None,
            "question_hint": None, "current_role": None,
            "known_keys": known_keys, "missing_keys": [],
        }

    gap = missing_keys[0]

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
        }

    task = required[gap]
    return {
        "current_gap": gap,
        "current_objective": task["objective"],
        "question_hint": task["question_hint"],
        "current_role": None,
        "known_keys": known_keys,
        "missing_keys": missing_keys,
    }

def interview_planner_node(state: AgentState) -> dict:
    topic_status = dict(state.get("topic_status", {}))
    current_topic = state.get("current_topic")
    scope = state.get("discovery_scope", DiscoveryScope.USER_APP)

    print("\n=== INTERVIEW PLANNER ===")
    print("Scope:", scope.value)
    print("Current topic:", current_topic)

    # --------------------------------------------------------
    # Stay on current topic, but only if it still has a real gap
    # --------------------------------------------------------
    if (
        current_topic is not None
        and topic_status.get(current_topic) != TopicStatus.COMPLETED
    ):
        gap = build_gap_info(state, current_topic)

        print("Known:", gap["known_keys"])
        print("Missing:", gap["missing_keys"])
        print("Current gap:", gap["current_gap"])

        if gap["current_gap"] is not None:
            return {
                "current_topic": current_topic,
                **gap,
            }

        # No gaps left — mark complete and fall through to pick the next topic
        # instead of returning a null objective/hint to the generator.
        topic_status[current_topic] = TopicStatus.COMPLETED
        current_topic = None

    # --------------------------------------------------------
    # Find next topic
    # --------------------------------------------------------
    for topic in DiscoveryTopic:
        status = topic_status.get(topic)

        if status == TopicStatus.COMPLETED:
            continue

        deps = TOPIC_PREREQUISITES.get(topic, [])
        deps_met = all(
            topic_status.get(dep) == TopicStatus.COMPLETED
            for dep in deps
        )
        if not deps_met:
            continue

        updated = dict(topic_status)
        if status is None:
            updated[topic] = TopicStatus.IN_PROGRESS

        gap = build_gap_info(state, topic)

        print(f"\nSelected topic: {topic.value}")
        print("Known:", gap["known_keys"])
        print("Missing:", gap["missing_keys"])
        print("Current gap:", gap["current_gap"])

        return {
            "current_topic": topic,
            "topic_status": updated,
            **gap,
        }

    # All topics exhausted — persist the completion we just marked above
    return {
        "current_topic": None,
        "topic_status": topic_status,
        "awaiting_confirmation": True,
    }