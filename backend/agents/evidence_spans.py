"""Recover source spans from formatting noise, never from semantic similarity."""
import re
import unicodedata


def recover_evidence_span(evidence: str, source: str) -> str | None:
    if not evidence or not evidence.strip():
        return None
    # Already exact evidence needs no formatting interpretation. Require a
    # unique occurrence, including overlapping occurrences.
    exact = list(re.finditer(f"(?=({re.escape(evidence)}))", source))
    if exact:
        return evidence if len(exact) == 1 else None

    tokens = re.findall(r"\w+|\s+|[^\w\s]", evidence.strip())
    if not any(re.fullmatch(r"\w+", token) for token in tokens):
        return None
    pattern = []
    for index, token in enumerate(tokens):
        if token.isspace():
            pattern.append(r"\s+")
        elif re.fullmatch(r"\w+", token):
            pattern.append(r"\b" + re.escape(token) + r"\b")
        else:
            # Never remove punctuation inside lexical units (can't, co-owner,
            # 1.5) or mathematical/currency symbols. Added sentence punctuation
            # and quote wrappers may disappear; source punctuation cannot.
            internal = (index > 0 and index + 1 < len(tokens)
                        and tokens[index - 1][-1].isalnum()
                        and tokens[index + 1][0].isalnum())
            optional = unicodedata.category(token).startswith("P") and not internal
            pattern.append(re.escape(token) + ("?" if optional else ""))
    matches = list(re.finditer("(?=(" + "".join(pattern) + "))", source, re.IGNORECASE))
    # Optional leading wrappers can create several matches ending at the same
    # location. Keep the maximal source span at that location.
    spans = {}
    for match in matches:
        start, end = match.span(1)
        spans[end] = min(start, spans.get(end, start))
    if len(spans) != 1:
        return None
    end, start = next(iter(spans.items()))
    return source[start:end]
