"""Standalone Ollama probe for the five extraction pass prompts (stdlib only)."""
import json
import urllib.request
from pathlib import Path


CARE = ("I want to build CareConnect, a platform that helps patients find and book appointments with healthcare professionals such as general doctors, dermatologists, dentists, physiotherapists, and nutritionists. Patients should be able to describe their health concern, search for suitable healthcare providers, compare their profiles and availability, book appointments, communicate with the provider, receive appointment reminders, and pay through the platform. Healthcare providers should be able to manage their profiles, specialties, schedules, appointments, and payments.")
FIX = ("I want to build FixMate, a platform that helps people find and book trusted local artisans like electricians, plumbers, cleaners, painters, and appliance repair technicians. Users should be able to describe what they need, find suitable service providers nearby, compare them, book a service, communicate with the provider, and pay through the platform.")

PASSES = {
    "ACTOR": ("Identify functional product actors only. Primary users include demand and supply sides in the normal core value proposition. Secondary users mainly administer, moderate, support, supervise or audit, but emit them only when explicitly described. Never infer an admin because providers manage their own profiles. Professions are categories, not separate roles. Keep named functions: patients -> patient, healthcare providers -> healthcare provider. Normalize vague people booking artisans to customer. Do not extract behavior.", ["primary_users", "secondary_users"], "roles"),
    "RESPONSIBILITY": ("Extract only actor responsibilities and permissions. Responsibilities are normal expected behavior. Permissions are explicit authorization, prohibition or restriction. Do not label every action a permission. Give each item an owner role.", ["responsibilities", "permissions"], "role"),
    "GOAL": ("Extract only user goals, success criteria and motivations. Goals are desired outcomes, not action lists. Do not extract actors or workflow.", ["primary_user_goals", "secondary_user_goals", "success_criteria", "motivations"], None),
    "WORKFLOW": ("Extract only explicitly described normal workflow. Preserve stated order; an action list can be workflow_steps even without a complete process. Do not invent sequence.", ["trigger", "workflow_steps", "completion_condition", "downstream_dependency", "end_state"], None),
    "RULES": ("Extract only explicit BUSINESS_RULES (validation_rules, approval_rules, eligibility_rules, limits, ownership_rules, visibility_rules), CONSTRAINTS (legal_constraints, business_constraints, operational_constraints, geographic_constraints, time_constraints), MVP_SCOPE (must_have_features, nice_to_have_features, out_of_scope, success_metrics), EXCEPTIONS (user_cancellations, timeouts, invalid_actions, recovery), EDGE_CASES (duplicate_actions, boundary_conditions, simultaneous_actions, rare_scenarios). Do not re-extract other knowledge and do not assume all features are MVP.", [], "topic"),
}


def call(name, message, actor_context="", post_context=""):
    purpose, keys, special = PASSES[name]
    props = {"key": {"type": "string"}, "value": {"type": "string"},
             "evidence": {"type": "string"}, "confidence": {"type": "number"},
             "knowledge_state": {"type": "string", "enum": ["CONFIRMED", "INFERRED"]}}
    if keys:
        props["key"]["enum"] = keys
    required = list(props)
    if special == "roles":
        props["roles"] = {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 1}
        props["aliases"] = {"type": "array", "items": {"type": "string"}}
        required.append("roles")
    elif special == "role":
        props["role"] = {"type": "string"}
        required.append("role")
    elif special == "topic":
        props["topic"] = {"type": "string", "enum": ["BUSINESS_RULES", "CONSTRAINTS", "MVP_SCOPE", "EXCEPTIONS", "EDGE_CASES"]}
        required.append("topic")
    if name == "GOAL":
        props["role"] = {"type": "string"}
        required.append("role")
    fmt = {"type": "object", "properties": {"items": {"type": "array", "items": {
        "type": "object", "properties": props, "required": required, "additionalProperties": False}}},
        "required": ["items"], "additionalProperties": False}
    prompt = (f"Extract facts from the latest USER_APP response. {purpose} Return JSON items. "
              "Evidence must be an exact case-sensitive substring of the response supporting the whole item. "
              "Use a short sufficient quote. Every explicit fact has knowledge_state CONFIRMED. "
              f"{actor_context}Latest user response:\n{message}\n{post_context}")
    body = json.dumps({"model": "qwen2.5:7b", "stream": False, "format": fmt,
                       "messages": [{"role": "system", "content": prompt}]}).encode()
    request = urllib.request.Request("http://127.0.0.1:11434/api/chat", body,
                                     {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.loads(response.read())["message"]["content"]


if __name__ == "__main__":
    output = Path(__file__).with_name("live_pass_probe_output.json")
    results = {}
    for label, message in (("CareConnect", CARE), ("FixMate", FIX)):
        results[label] = {}
        for name in PASSES:
            try:
                results[label][name] = call(name, message)
            except Exception as exc:
                results[label][name] = f"ERROR: {exc}"
            output.write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(label, name, results[label][name], flush=True)
