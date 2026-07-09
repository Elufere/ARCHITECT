---
source_url: "https://docs.stripe.com/api/idempotent_requests"
category: "standards"
title: "Stripe — Idempotent Requests"
slug: "stripe-idempotent-requests"
date_scraped: "2026-07-09 13:34:19 UTC"
description: "Idempotency key specification for POST requests — error propagation and response caching"
---
# Idempotent requests

The API supportsidempotencyfor safely retrying requests without accidentally performing the same operation twice. When creating or updating an object, use an idempotency key. Then, if a connection error occurs, you can safely repeat the request without risk of creating a second object or performing the update twice.

To perform an idempotent request, provide an additionalIdempotencyKeyelement to the request options.

Stripe’s idempotency works by saving the resulting status code and body of the first request made for any given idempotency key, regardless of whether it succeeds or fails. Subsequent requests with the same key return the same result, including500errors.

A client generates an idempotency key, which is a unique key that the server uses to recognize subsequent retries of the same request. How you create unique keys is up to you, but we suggest using V4 UUIDs, or another random string with enough entropy to avoid collisions. Idempotency keys are up to 255 characters long. Avoid using sensitive data (for example, email addresses or personal identifiers) as idempotency keys.

You can remove keys from the system automatically after they’re at least 24 hours old. We generate a new request if a key is reused after the original is pruned. The idempotency layer compares incoming parameters to those of the original request and errors if they’re not the same to prevent accidental misuse.

We save results only after the execution of an endpoint begins. If incoming parameters fail validation, or the request conflicts with another request that’s executing concurrently, we don’t save the idempotent result because no API endpoint initiates the execution. You can retry these requests. Learn more about when you canretry idempotent requests.

AllPOSTrequests accept idempotency keys. Don’t send idempotency keys inGETandDELETErequests because it has no effect. These requests are idempotent by definition.

Was this section helpful?
Yes
No

```
curl https://api.stripe.com/v1/customers \  -u sk_test_BQokikJ...2HlWgH4olfQ2sk_test_BQokikJOvBiI2HlWgH4olfQ2: \  -H "Idempotency-Key: KG5LxwFBepaKHyUD" \  -d description="My First Test Customer (created for API docs at https://docs.stripe.com/api)"
```

# Include-dependent response values (API v2)

Some API v2 responses contain null values for certain properties by default, regardless of their actual values. That reduces the size of response payloads while maintaining the basic response structure. To retrieve the actual values for those properties, specify them in theincludearray request parameter.

To determine whether you need to use theincludeparameter in a given request, look at the request description. Theincludeparameter’s enum values represent the response properties that depend on theincludeparameter.

Note

Whether a response property defaults to null depends on the request endpoint, not the object that the endpoint references. If multiple endpoints return data from the same object, a particular property can depend onincludein one endpoint and return its actual value by default for a different endpoint.

A hash property can depend on a singleincludevalue, or on multipleincludevalues associated with its child properties. For example, when updating an Account, to return actual values for the entireidentityhash, specifyidentityin theincludeparameter. Otherwise, theidentityhash is null in the response. However, to return actual values for theconfigurationhash, you must specify individual configurations in the request. If you specify at least one configuration, but not all of them, specified configurations return actual values and unspecified configurations return null. If you don’t specify any configurations, theconfigurationhash is null in the response.

- Relatedguide:Include-dependent response values

```
curl -X POST https://api.stripe.com/v2/core/accounts \  -H "Authorization: Bearer sk_test_BQokikJ...2HlWgH4olfQ2sk_test_BQokikJOvBiI2HlWgH4olfQ2" \  -H "Stripe-Version: 2026-06-24.preview" \  --json '{    "include": [        "identity",        "configuration.customer"    ]  }'
```

Included response properties

The response includes actual values for the properties specified in theincludeparameter, and null for all other include-dependent properties.

Response

```
{  "id": "acct_123",  "object": "v2.core.account",  "applied_configurations": [    "customer",    "merchant"  ],  "configuration": {    "customer": {      "automatic_indirect_tax": {        ...      },      "billing": {        ...      },      "capabilities": {        ...      },      ...    },    "merchant": null,    "recipient": null  },  "contact_email": "furever@example.com",  "created": "2025-06-09T21:16:03.000Z",  "dashboard": "full",  "defaults": null,  "display_name": "Furever",  "identity": {    "business_details": {      "doing_business_as": "FurEver",      "id_numbers": [        {          "type": "us_ein"        }      ],      "product_description": "Saas pet grooming platform at furever.dev using Connect embedded components",      "structure": "sole_proprietorship",      "url": "http://accessible.stripe.com"    },    "country": "US"  },  "livemode": true,  "metadata": {},  "requirements": null}
```

# Metadata

Updateable Stripe objects—includingAccount,Charge,Customer,PaymentIntent,Refund,Subscription, andTransferhave ametadataparameter. You can use this parameter to attach key-value data to these Stripe objects.

You can specify up to 50 keys, with key names up to 40 characters long and values up to 500 characters long. Keys and values are stored as strings and can contain any characters with one exception: you can’t use square brackets ([ and ]) in keys.

You can use metadata to store additional, structured information on an object. For example, you could store your user’s full name and corresponding unique identifier from your system on a StripeCustomerobject. Stripe doesn’t use metadata—for example, we don’t use it to authorize or decline a charge and it won’t be seen by your users unless you choose to show it to them.

Some of the objects listed above also support adescriptionparameter. You can use thedescriptionparameter to annotate a charge-for example, a human-readable description such as2 shirts for test@example.com. Unlikemetadata,descriptionis a single string, which your users might see (for example, in email receipts Stripe sends on your behalf).

Don’t store any sensitive information (bank account numbers, card details, and so on) as metadata or in thedescriptionparameter.

- Relatedguide:Metadata

## Sample metadata use cases

- Link IDs: Attach your system’s unique IDs to a Stripe object to simplify lookups. For example, add your order number to a charge, your user ID to a customer or recipient, or a unique receipt number to a transfer.
- Refund papertrails: Store information about the reason for a refund and the individual responsible for its creation.
- Customer details: Annotate a customer by storing an internal ID for your future use.
Was this section helpful?
Yes
No

```
curl https://api.stripe.com/v1/customers \  -u "sk_test_BQokikJ...2HlWgH4olfQ2sk_test_BQokikJOvBiI2HlWgH4olfQ2:" \  -d "metadata[order_id]=6735"
```

```
{  "id": "cus_123456789",  "object": "customer",  "address": {    "city": "city",    "country": "US",    "line1": "line 1",    "line2": "line 2",    "postal_code": "90210",    "state": "CA"  },  "balance": 0,  "created": 1483565364,  "currency": null,  "default_source": null,  "delinquent": false,  "description": null,  "discount": null,  "email": null,  "invoice_prefix": "C11F7E1",  "invoice_settings": {    "custom_fields": null,    "default_payment_method": null,    "footer": null,    "rendering_options": null  },  "livemode": false,  "metadata": {    "order_id": "6735"  },  "name": null,  "next_invoice_sequence": 1,  "phone": null,  "preferred_locales": [],  "shipping": null,  "tax_exempt": "none"}
```