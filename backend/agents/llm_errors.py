"""Failures that must reach the durable interview boundary, never become facts."""


class LLMCallFailed(RuntimeError):
    def __init__(self, call_name, *, retryable=False, failure_kind=None, status_code=None):
        self.call_name = call_name
        self.retryable = retryable
        self.failure_kind = failure_kind
        self.status_code = status_code
        message = ("Unable to reach OpenAI after bounded retries" if retryable else
                   "OpenAI request failed; check API configuration, credentials, quota, or request compatibility")
        diagnostic = ""
        if failure_kind:
            diagnostic = f"; cause={failure_kind}"
            if status_code is not None:
                diagnostic += f"; status={status_code}"
        super().__init__(f"{message} (call: {call_name}{diagnostic})")


class ExtractionFailed(RuntimeError):
    """A failed interpretation is not a successful NO_FACTS_FOUND result."""


def raise_if_llm_failure(error):
    if isinstance(error, (LLMCallFailed, ExtractionFailed)):
        raise error
