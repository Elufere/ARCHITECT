"""Plain-language questions/clarifications, separate from semantic field definitions.

These templates explain the requested information; they never decide whether a
user answer satisfies a gap or invent product actors/capabilities.
"""

import re


def final_question_text(content: str) -> str:
    """Return the final interview question from a PM response.

    Most turns contain only a question. Advice-with-continuation turns may contain
    short PM suggestions first; short answers such as "yes" must still be resolved
    against the actual trailing question rather than the advisory prose.
    """
    text = (content or "").strip()
    if not text:
        return ""
    matches = list(re.finditer(r"(?:^|\n)([^\n?]*\?)\s*$", text, re.M))
    if matches:
        return matches[-1].group(1).strip()
    question_positions = [match.start() for match in re.finditer(r"[^?]+\?", text)]
    if question_positions:
        segment = text[question_positions[-1]:].strip()
        return segment
    return text


def _join(labels):
    if len(labels) < 2:
        return "".join(labels)
    return ", ".join(labels[:-1]) + " and " + labels[-1]


def confirmed_actor_labels(state, source="primary_users"):
    scope = state.get("discovery_scope", "USER_APP")
    labels, seen = [], set()
    for item in state.get("discovered_knowledge", []):
        if (item.scope != scope or item.topic != "USER_ROLES" or item.key != source
                or item.knowledge_state != "CONFIRMED" or item.absence):
            continue
        for role in item.roles or []:
            identity = role.replace("_", " ").strip().lower()
            if identity in seen:
                continue
            seen.add(identity)
            value = item.value.replace("_", " ").strip()
            # Prefer a short declared display label; use the actor ID if the
            # stored value is a sentence/compound declaration instead of a name.
            label = value if len(item.roles) == 1 and len(value.split()) <= 4 and not any(c in value for c in ".!?;:") else identity
            labels.append(label.lower())
    return labels


def secondary_users_question(state):
    labels = confirmed_actor_labels(state)
    app = "dashboard" if state.get("discovery_scope") == "ADMIN_DASHBOARD" else "user app"
    if labels:
        return f"Besides {_join(labels)}, will anyone else use the {app}?"
    return f"Will anyone else need to use the {app}?"


# Explicit user-facing wording for all current gaps. Never interpolate a schema
# definition or planner objective into a reply to "what do you mean?".
CLARIFICATION_QUESTIONS = {
    "primary_users": "Who will actually use the app, and what will they use it for?",
    "responsibilities": "What should {person} be able to do in the app?",
    "permissions": "Are there things {person} must not be allowed to see or do, or things that require approval first?",
    "multiple_roles": "Could the same person use one account for more than one kind of user?",
    "role_transitions": "Could someone change from one kind of user to another later?",
    "primary_user_goals": "What result does {person} want to achieve by using the app?",
    "secondary_user_goals": "What result does {person} want to achieve by using the app?",
    "success_criteria": "How would someone know they had achieved what they came to do?",
    "motivations": "What problem makes people want to use this app?",
    "trigger": "What happens that makes someone start using this part of the app?",
    "workflow_steps": "What does someone do first, next, and after that?",
    "completion_condition": "What must happen before you would say the process is finished?",
    "downstream_dependency": "Does anything outside this part of the app need to happen before it can finish, such as approval from someone else?",
    "end_state": "Once the process is finished, what should the result look like?",
    "validation_rules": "What should the app check before accepting someone's information or request?",
    "approval_rules": "Does anyone need to approve an action before it can go ahead?",
    "eligibility_rules": "Does someone need to meet any conditions before they can use the service?",
    "limits": "Are there limits on how much someone can do, such as a maximum number or amount?",
    "ownership_rules": "Who owns or controls the information and resources people create?",
    "visibility_rules": "Who should be able to see which information?",
    "legal_constraints": "Are there any laws or regulations the app must follow?",
    "business_constraints": "Are there business policies or budget limits the app must work within?",
    "operational_constraints": "Are there practical limits or outside services the app needs to work with?",
    "geographic_constraints": "Is the service available everywhere, or only in certain places?",
    "time_constraints": "Are there deadlines, expiry times, or hours when the service is available?",
    "must_have_features": "What absolutely needs to be available when the first version launches?",
    "nice_to_have_features": "What would be useful but could wait until a later version?",
    "out_of_scope": "What have you decided to leave out of the first version?",
    "success_metrics": "What would you measure to tell whether the first version is doing well?",
    "user_cancellations": "What should happen if someone cancels?",
    "timeouts": "What should happen if someone takes too long to finish?",
    "invalid_actions": "What should the app do when someone tries something invalid or not allowed?",
    "recovery": "If the process is interrupted or fails, how should someone continue or try again?",
    "duplicate_actions": "What should happen if someone submits the same request twice?",
    "boundary_conditions": "What should happen when a limit is reached, or there is nothing available?",
    "simultaneous_actions": "What should happen if two people try to use the same thing at the same time?",
    "rare_scenarios": "Is there an unusual situation we should decide how to handle?",
}


def clarification_question(state):
    if state.get("planner_source") == "validation":
        issue = state.get("selected_validation_issue") or {}
        values = [str(value).strip() for value in issue.get("fact_values", []) if str(value).strip()]
        if len(values) >= 2:
            return (
                f"I have two conflicting rules recorded: '{values[0]}' and '{values[1]}'. "
                "Which one should apply now, or what is the current rule?"
            )
        return "I have conflicting product information recorded. What is the current rule that should apply?"

    key, _, role = (state.get("current_gap") or "").partition("::")
    if key == "secondary_users":
        return secondary_users_question(state)
    question = CLARIFICATION_QUESTIONS.get(key)
    if not question:
        return "Which part of my last question would you like me to explain?"
    person = "someone using the app" if not role else "the " + role.replace("_", " ")
    return question.format(person=person)


def clarification_reply(state):
    question = clarification_question(state)
    return "I mean, " + question[0].lower() + question[1:]
