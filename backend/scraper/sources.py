"""
Scraper source configuration.

Each source maps to a category folder under knowledge_base/.okf/
Categories: 'standards' | 'security'

To add new sources: append to the SOURCES list.
To add new categories: create the folder under .okf/ and use the category name here.
"""

from __future__ import annotations

SOURCES: list[dict] = [
    # ========================================================================
    # PRODUCT REQUIREMENT & SPECIFICATION STANDARDS
    # ========================================================================
    {
        "url": "https://www.rfc-editor.org/rfc/rfc2119.html",
        "category": "standards",
        "title": "RFC 2119 — Key Words for Use in RFCs",
        "slug": "rfc-2119-keywords",
        "description": "Canonical MUST/SHOULD/MAY/SHOULD NOT/MUST NOT requirement-level keywords for unambiguous spec language",
    },
    {
        "url": "https://www.rfc-editor.org/rfc/rfc8174.html",
        "category": "standards",
        "title": "RFC 8174 — Clarifying RFC 2119 Keywords",
        "slug": "rfc-8174-2119-clarification",
        "description": "Clarifies that RFC 2119 keywords only carry requirement force when in ALL CAPS; closes a real ambiguity",
    },
    {
        "url": "https://www.industrialempathy.com/posts/design-docs-at-google/",
        "category": "standards",
        "title": "Design Docs at Google — Structure and Process",
        "slug": "google-design-docs-practice",
        "description": "Widely-cited breakdown of Google's design-doc practice — structure, review process, when a design doc is warranted",
    },
    {
        "url": "https://abseil.io/resources/swe-book/html/ch10.html",
        "category": "standards",
        "title": "Software Engineering at Google — Chapter 10: Documentation",
        "slug": "google-swe-book-documentation",
        "description": "Documentation types, design docs, audience-separation principles from the O'Reilly book hosted free by Abseil",
    },
    {
        "url": "https://www.gov.uk/service-manual/agile-delivery/writing-user-stories",
        "category": "standards",
        "title": "Gov.uk — Writing User Stories",
        "slug": "govuk-writing-user-stories",
        "description": "UK Government Digital Service formal standard for user stories, acceptance criteria, epics vs. stories",
    },
    {
        "url": "https://www.gov.uk/service-manual/agile-delivery/agile-tools-techniques",
        "category": "standards",
        "title": "Gov.uk — Agile Tools and Techniques",
        "slug": "govuk-agile-tools-techniques",
        "description": "GDS manual on backlog, sprint planning, and related artifacts that surround requirement documents",
    },
    {
        "url": "https://adr.github.io/",
        "category": "standards",
        "title": "Architecture Decision Records — Canonical Reference",
        "slug": "adr-canonical-reference",
        "description": "ADR reference site — formats, templates, history, and how ADRs relate to specs and RFCs",
    },
    {
        "url": "https://gds-way.digital.cabinet-office.gov.uk/standards/architecture-decisions.html",
        "category": "standards",
        "title": "GDS — Documenting Architecture Decisions",
        "slug": "gds-architecture-decisions",
        "description": "UK government engineering standard on ADR format, status lifecycle, and when to write one",
    },
    {
        "url": "https://github.com/joelparkerhenderson/architecture-decision-record/blob/main/locales/en/templates/decision-record-template-by-michael-nygard/index.md",
        "category": "standards",
        "title": "Michael Nygard's Original ADR Template",
        "slug": "nygard-adr-template",
        "description": "The template that started the ADR practice — title, context, decision, status, consequences",
    },
    {
        "url": "https://raw.githubusercontent.com/phuryn/pm-skills/main/pm-execution/skills/create-prd/SKILL.md",
        "category": "standards",
        "title": "Comprehensive 8-Section PRD Engineering Spec",
        "slug": "eight-section-prd-spec",
        "description": "Structural guide: background, SMART OKR objectives, solution architecture, and scoping constraints",
    },
    {
        "url": "https://raw.githubusercontent.com/mattpocock/skills/main/skills/engineering/to-prd/SKILL.md",
        "category": "standards",
        "title": "Technical Feature Synthesis and PRD Boundary Mapping",
        "slug": "tech-feature-prd-mapping",
        "description": "Rule-set mapping raw user intent to modular implementation choices, schema variations, and API bounds",
    },
    {
        "url": "https://raw.githubusercontent.com/jamesrochabrun/skills/main/skills/prd-generator/SKILL.md",
        "category": "standards",
        "title": "Automated Functional PRD Generator Protocol",
        "slug": "functional-prd-generator-protocol",
        "description": "Step-by-step constraint logic for user stories, acceptance criteria, metric frameworks (HEART/AARRR), and edge cases",
    },
        {
        "url": "https://www.aboutamazon.com/news/workplace/an-insider-look-at-amazons-culture-and-processes",
        "category": "standards",
        "title": "Amazon — Working Backwards / PR-FAQ Method",
        "slug": "amazon-working-backwards-pr-faq",
        "description": "Amazon's official explanation of the PR/FAQ narrative spec process — writing the press release before building",
    },
    {
        "url": "https://google.github.io/eng-practices/",
        "category": "standards",
        "title": "Google Engineering Practices",
        "slug": "google-engineering-practices",
        "description": "Google's public engineering practices — code review, documentation, and spec review culture",
    },

    # ========================================================================
    # DATABASE DESIGN & SCALING STANDARDS
    # ========================================================================
    {
        "url": "https://www.postgresql.org/docs/current/indexes.html",
        "category": "standards",
        "title": "PostgreSQL Index Types and When to Use Them",
        "slug": "postgres-index-types",
        "description": "B-tree, GiST, GIN, BRIN, hash indexes — when each helps or hurts query performance",
    },
    {
        "url": "https://www.postgresql.org/docs/current/indexes-multicolumn.html",
        "category": "standards",
        "title": "PostgreSQL Multicolumn Indexes",
        "slug": "postgres-multicolumn-indexes",
        "description": "Composite index behavior, leading-column rules, and column order selection",
    },
    {
        "url": "https://www.postgresql.org/docs/current/indexes-partial.html",
        "category": "standards",
        "title": "PostgreSQL Partial Indexes",
        "slug": "postgres-partial-indexes",
        "description": "Partial index design patterns — excluding common or uninteresting values to shrink index size",
    },
    {
        "url": "https://www.postgresql.org/docs/current/ddl-constraints.html",
        "category": "standards",
        "title": "PostgreSQL Constraints — Foreign Keys, Check, Exclusion",
        "slug": "postgres-constraints",
        "description": "Foreign keys, check constraints, ON DELETE/UPDATE actions, exclusion constraints",
    },
    {
        "url": "https://www.postgresql.org/docs/current/transaction-iso.html",
        "category": "standards",
        "title": "PostgreSQL Transaction Isolation Levels",
        "slug": "postgres-transaction-isolation",
        "description": "Read committed through serializable — concurrency anomalies and what each level prevents",
    },
    {
        "url": "https://wiki.postgresql.org/wiki/Don't_Do_This",
        "category": "standards",
        "title": "PostgreSQL Anti-Patterns — Don't Do This",
        "slug": "postgres-dont-do-this",
        "description": "Community-maintained list of common PostgreSQL anti-patterns: NOT IN pitfalls, char(n), serial types",
    },
    {
        "url": "https://use-the-index-luke.com/",
        "category": "standards",
        "title": "Use The Index, Luke — Indexing Deep Dive",
        "slug": "use-the-index-luke",
        "description": "Markus Winand's vendor-agnostic guide to B-tree internals, index scans, and query optimization",
    },
    {
        "url": "https://brandur.org/idempotency-keys",
        "category": "standards",
        "title": "Implementing Idempotency Keys in Postgres",
        "slug": "brandur-idempotency-keys",
        "description": "Stripe-style idempotency keys — schema design and race-condition handling",
    },
    {
        "url": "https://www.ibm.com/think/topics/database-normalization",
        "category": "standards",
        "title": "IBM Database Normalization — 1NF through 4NF",
        "slug": "ibm-database-normalization",
        "description": "Formal normalization forms with functional dependency definitions and worked examples",
    },
    {
        "url": "https://brandur.org/postgres-connections",
        "category": "standards",
        "title": "Postgres Connection Pooling and Limits",
        "slug": "brandur-postgres-connections",
        "description": "Connection pooling patterns, performance boundaries, and transaction overhead in Postgres",
    },

    # ========================================================================
    # API DESIGN CONVENTIONS
    # ========================================================================
    {
        "url": "https://google.aip.dev/158",
        "category": "standards",
        "title": "Google AIP-158 — Pagination",
        "slug": "google-aip-158-pagination",
        "description": "Page tokens, opaque cursors, backward-compatibility rules for paginated APIs",
    },
    {
        "url": "https://google.aip.dev/132",
        "category": "standards",
        "title": "Google AIP-132 — Standard List Method",
        "slug": "google-aip-132-list-method",
        "description": "List method conventions: page_size, page_token, filtering semantics",
    },
    {
        "url": "https://google.aip.dev/121",
        "category": "standards",
        "title": "Google AIP-121 — Resource-Oriented Design",
        "slug": "google-aip-121-resource-design",
        "description": "Resource-oriented API design principles — nouns over verbs, standard methods",
    },
    {
        "url": "https://docs.stripe.com/api/idempotent_requests",
        "category": "standards",
        "title": "Stripe — Idempotent Requests",
        "slug": "stripe-idempotent-requests",
        "description": "Idempotency key specification for POST requests — error propagation and response caching",
    },
    {
        "url": "https://docs.stripe.com/pagination",
        "category": "standards",
        "title": "Stripe — Cursor-Based Pagination",
        "slug": "stripe-pagination",
        "description": "Cursor-based pagination with limit, has_more, and auto-pagination patterns",
    },
    {
        "url": "https://docs.stripe.com/error-low-level",
        "category": "standards",
        "title": "Stripe — Error Handling and Retries",
        "slug": "stripe-error-handling",
        "description": "Error handling and retry semantics for 4xx/5xx/network errors",
    },
    {
        "url": "https://www.rfc-editor.org/rfc/rfc9457.html",
        "category": "standards",
        "title": "RFC 9457 — Problem Details for HTTP APIs",
        "slug": "rfc-9457-problem-details",
        "description": "Standardized machine-readable error format for HTTP APIs",
    },

    # ========================================================================
    # SECURITY STANDARDS
    # ========================================================================
    {
        "url": "https://owasp.org/API-Security/editions/2023/en/0xa1-broken-object-level-authorization/",
        "category": "security",
        "title": "OWASP API1:2023 — Broken Object Level Authorization",
        "slug": "owasp-api1-bola",
        "description": "BOLA (IDOR) — accessing other users' resources by manipulating object IDs",
    },
    {
        "url": "https://owasp.org/API-Security/editions/2023/en/0xa3-broken-object-property-level-authorization/",
        "category": "security",
        "title": "OWASP API3:2023 — Broken Object Property Level Authorization",
        "slug": "owasp-api3-bopla",
        "description": "Mass assignment and excessive data exposure — returning or writing fields the user shouldn't access",
    },
    {
        "url": "https://github.com/OWASP/API-Security/blob/master/editions/2023/en/0xa4-unrestricted-resource-consumption.md",
        "category": "security",
        "title": "OWASP API4:2023 — Unrestricted Resource Consumption",
        "slug": "owasp-api4-resource-consumption",
        "description": "Rate limiting, payload size limits, and denial-of-service via API abuse",
    },
    {
        "url": "https://cheatsheetseries.owasp.org/cheatsheets/SQL_Injection_Prevention_Cheat_Sheet.html",
        "category": "security",
        "title": "OWASP — SQL Injection Prevention",
        "slug": "owasp-sql-injection-prevention",
        "description": "Parameterized queries, stored procedures, and defensive SQL escaping",
    },
    {
        "url": "https://cheatsheetseries.owasp.org/cheatsheets/Authentication_Cheat_Sheet.html",
        "category": "security",
        "title": "OWASP — Authentication Cheat Sheet",
        "slug": "owasp-authentication",
        "description": "Session binding, MFA, reauthentication, and credential handling",
    },
    {
        "url": "https://cheatsheetseries.owasp.org/cheatsheets/Session_Management_Cheat_Sheet.html",
        "category": "security",
        "title": "OWASP — Session Management Cheat Sheet",
        "slug": "owasp-session-management",
        "description": "Session ID generation, cookie flags, timeouts, and token rotation",
    },
    {
        "url": "https://cheatsheetseries.owasp.org/cheatsheets/REST_Security_Cheat_Sheet.html",
        "category": "security",
        "title": "OWASP — REST Security Cheat Sheet",
        "slug": "owasp-rest-security",
        "description": "HTTPS enforcement, JWT denylisting, API key handling, CORS policies",
    },
    {
        "url": "https://cheatsheetseries.owasp.org/cheatsheets/Insecure_Direct_Object_Reference_Prevention_Cheat_Sheet.html",
        "category": "security",
        "title": "OWASP — IDOR Prevention Cheat Sheet",
        "slug": "owasp-idor-prevention",
        "description": "Object-level authorization patterns for backend queries — indirect references, access control checks",
    },
    {
        "url": "https://cheatsheetseries.owasp.org/cheatsheets/GraphQL_Cheat_Sheet.html",
        "category": "security",
        "title": "OWASP — GraphQL Security Cheat Sheet",
        "slug": "owasp-graphql-security",
        "description": "Query depth/cost limiting, batching attacks, access control, disabling introspection in prod",
    },
    {
        "url": "https://cheatsheetseries.owasp.org/cheatsheets/gRPC_Security_Cheat_Sheet.html",
        "category": "security",
        "title": "OWASP — gRPC Security Cheat Sheet",
        "slug": "owasp-grpc-security",
        "description": "TLS config, message size limits, streaming limits, rate limiting interceptors",
    },
    {
        "url": "https://cheatsheetseries.owasp.org/cheatsheets/JSON_Web_Token_Cheat_Sheet.html",
        "category": "security",
        "title": "OWASP — JSON Web Token Cheat Sheet",
        "slug": "owasp-jwt-validation",
        "description": "Securely verifying, signing, and invalidating stateless JWT tokens",
    },
    {
        "url": "https://datatracker.ietf.org/doc/html/rfc6749",
        "category": "security",
        "title": "RFC 6749 — OAuth 2.0 Authorization Framework",
        "slug": "rfc-6749-oauth2",
        "description": "The core OAuth 2.0 specification — grant types, access tokens, refresh tokens",
    },
]

def get_sources_by_category(category: str) -> list[dict]:
    """Return all sources for a given category."""
    return [s for s in SOURCES if s["category"] == category]

def get_all_categories() -> list[str]:
    """Return unique category names."""
    return sorted(set(s["category"] for s in SOURCES))