"""Failures that must reach the durable interview boundary, never become facts."""


class LLMCallFailed(RuntimeError):
    def __init__(self, call_name, *, retryable=False):
        self.call_name = call_name
        self.retryable = retryable
        message = ("Unable to reach OpenAI after bounded retries" if retryable else
                   "OpenAI request failed; check API configuration, credentials, quota, or request compatibility")
        super().__init__(f"{message} (call: {call_name})")


class ExtractionFailed(RuntimeError):
    """A failed interpretation is not a successful NO_FACTS_FOUND result."""


def raise_if_llm_failure(error):
    if isinstance(error, (LLMCallFailed, ExtractionFailed)):
        raise error
