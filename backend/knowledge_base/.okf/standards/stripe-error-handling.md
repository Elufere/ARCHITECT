---
source_url: "https://docs.stripe.com/error-low-level"
category: "standards"
title: "Stripe — Error Handling and Retries"
slug: "stripe-error-handling"
date_scraped: "2026-07-09 13:34:22 UTC"
description: "Error handling and retry semantics for 4xx/5xx/network errors"
---
# Advanced errorhandling

## Learn how to understand errors at the HTTP level.

This page covers two advanced error handling topics:

- HTTP responses that represent errors
- Idempotency and retries

This information might not apply to you. The official StripeSDKscan handle most details involving HTTP and retries. If you use a client library, start here:

- Error handling
- Error codes

## Errors in HTTP

Even when an API call fails, our client libraries make error information available byraising an exception or returning an error value. But if you don’t use a client library, or if an unusual situation arises, you might need low-level details about HTTP responses and when we emit them.

From an HTTP point of view, errors fall into three major categories:

- Content error: Invalid content in the API request.
- Network error: Intermittent communication problems between client and server.
- Server error: A problem on Stripe’s servers.

Each type of error requires a different approach and idempotency semantics.A full listing of response codesand their meaning is provided at the end of this page.

### Content errors

Content errors result from invalid content of an API request. They return an HTTP response with a4xxresponse code. For example, the API servers might return a401if you provided an invalid API key, or a400if a required parameter was missing. Integrations should correct the original request, and try again. Depending on the type of user error (for example, a card being declined), it might be possible to handle the problem programmatically. In these cases, include acodefield to help an integration react appropriately. Seeerror codesfor more details.

For aPOSToperation using an idempotency key, as long as an API method began execution, Stripe’s API servers will cache the results of the request regardless of what they were. A request that returns a400sends back the same400if followed by a new request with the same idempotency key. Generate a fresh idempotency key when modifying the original request to get a successful result. This operation does contain some caveats. For example, a request that’s rate limited with a429can produce a different result with the same idempotency key because rate limiters run before the API’s idempotency layer. The same goes for a401that omitted an API key, or most400s that sent invalid parameters. Even so, the safest strategy where4xxerrors are concerned is to always generate a new idempotency key.

### Network errors

Network errors are the result of connectivity problems between client and server. They return low-level errors, like socket or timeout exceptions. For example, a client might time out while trying to read from the Stripe servers, or an API response might never be received because a connection terminates prematurely. Although a network errorseemslikely to succeed after you fix the connectivity problems, sometimes another error might hide in the intermittent problem.

This class of errors is where the value of idempotency keys and request retries is most obvious. When intermittent problems occur, clients are usually left in a state where they don’t know whether or not the server received the request. To get a definitive answer, they should retry such requests with the same idempotency keys and the same parameters until they’re able to receive a result from the server. Sending the same idempotency with different parameters produces an error indicating that the new request didn’t match the original.

Most client libraries can generate idempotency keys and retry requests automatically, but need to be configured to do so. They perform their first retry quickly after the first failure, and subsequent retries on an exponential backoff schedule, the assumption being that a single failure is often a random occurrence, but a pattern of repeated failures likely represents a chronic problem.

Select a language
Ruby
Python
PHP
Node.js
Java
Go
.NET
No results

```
Stripe.max_network_retries = 2
```

### Server errors

Server errors result from a problem with Stripe’s servers. They return an HTTP response with a5xxerror code. These errors are the most difficult to handle and we work to make them as rare as possible, but a good integration handles them when they do arise.

As with user errors, the idempotency layer caches the result ofPOSTmutations that result in server errors (specifically500s, which are internal server errors), so retrying them with the same idempotency key usually produces the same result. The client can retry the request with a new idempotency key, but we advise against it because the original key may have produced side effects.

You should treat the result of a500request as indeterminate. The most likely time to observe one is during a production incident, and generally during such an incident’s remediation. Stripe engineers examine failed requests and try to appropriately reconcile the results of any mutations that result in500s. While the idempotency-cached response to those requests won’t change, we’ll try to firewebhooksfor any new objects created as part of Stripe’s reconciliation. The exact nature of any retroactive changes in the system depends heavily on the type of request. For example, if creating a charge returns a500error but we detect that the information has gone out to a payment network, we’ll try to roll it forward. If not, we’ll try to roll it back. If this doesn’t resolve the issue, you may still see requests with a500error that produce user-visible side effects.

#### Caution

Treat requests that return500errors as indeterminate. Although Stripe tries to reconcile their partial state in the most appropriate manner and also firewebhooksfor new objects that are created, ideal results aren’t guaranteed.

To let your integration handle the widest range of500s, configure webhook handlers to receive event objects that you never receive in normal API responses. One technique for cross-referencing these new objects with the data from an integration’s local state is to send in a local identifier with themetadatawhen creating new resources with the API. That identifier appears in the metadata field of an object going out through a webhook, even if the webhook is generated later as part of reconciliation.

## Idempotency

Idempotencyis a web API design principle defined as the ability to apply the same operation multiple times without changing the result beyond the first try. It makes it safe to retry API requests in some situations—in particular, when the first request gets no response because of a network error. Because a certain amount of intermittent failure is to be expected, clients need a way of reconciling failed requests with a server, and idempotency provides a mechanism for that.

Most client libraries can generate idempotency keys and retry requests automatically, but you need to configure it. For finer-grained control over retries, generateidempotency keysand write your own logic for retries.

### GET and DELETE requests

The Stripe API guarantees the idempotency ofGETandDELETErequests, so it’s always safe to retry them.

### POST requests

Including anidempotency keymakesPOSTrequests idempotent, which prompts the API to do the record keeping required to prevent duplicate operations. Clients can safely retry requests that include an idempotency key as long as the second request occurs within 24 hours from when you first receive the key (keys expire out of the system after 24 hours). For example, if a request to create an object doesn’t respond because of a network connection error, a client can retry the request with the same idempotency key to guarantee that no more than one object is created.

### Sending idempotency keys

Idempotency keys are sent in theIdempotency-Keyheader. Use them for allPOSTrequests to the Stripe API. Most official client libraries can send them automatically, as long as they’re configured to send retries.

If you decide tosend idempotency keys manually, make sure the tokens being used are sufficiently unique to unambiguously identify a single operation within your account over the last 24 hours, at a minimum. There are two common strategies for generating idempotency keys:

- Use an algorithm that generates a token with enough randomness, like UUID v4.
- Derive the key from a user-attached object, like the ID of a shopping cart. This provides a relatively straightforward way to protect against double submissions.

To identify a previously executed response that’s being replayed from the server, look for the headerIdempotent-Replayed: true.

### The Stripe-Should-Retry header

A client library can’t always determine with certainty if it should retry based solely on a status code or content in the response body. The API responds with theStripe-Should-Retryheader when it has additional information that the request is retryable.

- Stripe-Should-Retryset totrueindicates that a client should retry the request. Clients should still wait some amount of time (probably determined according to an exponential backoff schedule) before making the next request so as not to overload the API.
- Stripe-Should-Retryset tofalsemeans that a client shouldnotretry the request because it won’t have an additional effect.
- Stripe-Should-Retrynot set in the response indicates that the API can’t determine whether or not it can retry the request. Clients should fall back to other properties of the response (like the status code) to make a decision.

The retry mechanisms built into Stripe client libraries respectStripe-Should-Retryautomatically. If you’re using one of them, you don’t need to handle it manually.

## HTTP Status Code Reference

| 200 | OK | Everything worked as expected. |
| 400 | Bad Request | The request was unacceptable, often due to missing a required parameter. |
| 401 | Unauthorized | No valid API key provided. |
| 402 | Request Failed | The parameters were valid but the request failed. |
| 403 | Forbidden | The API key doesn’t have permissions to perform the request. |
| 409 | Conflict | The request conflicts with another request (perhaps due to using the sameidempotentkey). |
| 424 | External Dependency Failed | The request couldn’t be completed due to a failure in a dependency external to Stripe. |
| 429 | Too Many Requests | Too many requests hit the API too quickly.We recommend an exponential backoff of your requests. |
| 500, 502, 503, 504 | Server Errors | Something went wrong on Stripe’s end. |