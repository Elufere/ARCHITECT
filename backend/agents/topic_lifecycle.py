"""Explicit completion invalidation at the knowledge commit boundary."""
from agents.interview_planner import build_gap_info, get_roles_in_discovery_order
from agents.role_utils import role_identity
from agents.state import DiscoveryScope, DiscoveryTopic, KnowledgeState, TopicStatus


def _gap_identity(gap):
    key, separator, role = gap.partition("::")
    return (key, role_identity(role)) if separator else (key, None)


def invalidate_completed_topics(state, knowledge, committed, topic_status):
    """Reopen only coverage newly invalidated by committed same-scope facts.

    Existing gaps are not an invalidation event. Compare the entire committed
    batch so a new actor supplied with complete coverage need not reopen anything.
    """
    scope = state.get("discovery_scope", DiscoveryScope.USER_APP)
    confirmed = [item for item in committed
                 if item in knowledge and item.scope == scope
                 and item.knowledge_state == KnowledgeState.CONFIRMED]
    if not confirmed:
        return topic_status
    after = {**state, "discovered_knowledge": knowledge}
    previous_roles = {role_identity(role) for role in
                      get_roles_in_discovery_order(state, DiscoveryTopic.USER_ROLES)}
    new_roles = [role for role in get_roles_in_discovery_order(after, DiscoveryTopic.USER_ROLES)
                 if role_identity(role) not in previous_roles]
    actor_changes = [item for item in confirmed if item.topic == DiscoveryTopic.USER_ROLES
                     and item.key in ("primary_users", "secondary_users")]
    updated = dict(topic_status)
    for topic, status in topic_status.items():
        if status != TopicStatus.COMPLETED:
            continue
        topic = DiscoveryTopic(topic)
        # Actor declarations can change role-dependent coverage across topics;
        # other committed facts can invalidate only their own topic's coverage.
        if not actor_changes and not any(item.topic == topic for item in confirmed):
            continue
        previous_gaps = {_gap_identity(gap) for gap in build_gap_info(state, topic)["missing_keys"]}
        new_gaps = [gap for gap in build_gap_info(after, topic)["missing_keys"]
                    if _gap_identity(gap) not in previous_gaps]
        if not new_gaps:
            continue
        affected_roles = [role for role in new_roles if any(
            "::" in gap and role_identity(gap.split("::", 1)[1]) == role_identity(role)
            for gap in new_gaps)]
        if affected_roles and actor_changes:
            reason = "new confirmed same-scope actor " + ", ".join(affected_roles)
        elif state.get("is_correction"):
            reason = "explicit correction changed confirmed completion coverage"
        else:
            reason = "committed confirmed fact changed required completion coverage"
        updated[topic] = TopicStatus.PARTIAL
        print(f"TOPIC INVALIDATION\n{topic.value}: COMPLETED -> PARTIAL\n"
              f"scope: {scope.value}\nreason: {reason}\nnew gaps: {', '.join(new_gaps)}")
    return updated
