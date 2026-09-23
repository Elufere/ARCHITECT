The reported failure followed an answer describing buyer and seller responsibilities. The user confirmed the active gap remained responsibilities::customer. Earlier yes/no routing fixes did not explain this particular failure.

The saved live extraction produced two valid responsibility candidates owned by customer. The same turn also produced eleven permission candidates, most describing ordinary capabilities. Before this change, all candidates shared one grounding call and a protocol/parse failure could discard the entire batch. Grounding received canonical actor IDs but omitted the original confirmed declaration that customers act as buyer or seller in a transaction.

The change preserves confirmed actor declarations and their evidence as identity context in extraction and grounding. Direct answers to the current gap are grounded separately from incidental claims. Each batch still requires semantic support; rejected direct answers are not promoted automatically. Logs now expose the direct/incidental candidate counts and accepted fact count for the active gap.

Verification:

- The real extraction output is preserved in tests/fixtures/escrow_responsibilities_extraction.json.
- Real-graph regression tests replay the exact answer and reach permissions::customer, without guardrail retries, even when the incidental audit returns malformed output or rejects all incidental claims.
- Negative tests confirm that an unsupported direct answer remains rejected and that other-scope/inferred actor relationships are excluded from confirmed identity context.
- The live focused grounding check accepted both customer responsibility facts and the planner selected permissions::customer. Its result is saved in escrow_grounding_after.json.
- 238 targeted tests passed before the final diagnostic-only logging additions.

Limit: the original rejected verdict is unavailable. The saved extraction subset also passed a baseline mixed audit in the new replay, so the precise model decision that caused the user's historical run cannot be proven. This change fixes demonstrable context loss and failure coupling in the grounding stage, but it is not evidence that every future model response or full PRD run will succeed.
