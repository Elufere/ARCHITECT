---
source_url: https://docs.stripe.com/pagination
category: standards
title: Stripe — Cursor-Based Pagination
slug: stripe-pagination
date_scraped: "2026-07-07 10:41:49 UTC"
description: "Cursor-based pagination with limit, has_more, and auto-pagination patterns"
---
# How paginationworks

## Learn how to paginate results for list and search endpoints.

The Stripe API has list and search endpoints that can return multiple objects, such as listing Customers or searching for PaymentIntents. To mitigate negative impacts to performance, these endpoints don’t return all results at once. Instead, Stripe returns one page of results per API call, with each page containing up to 10 results by default. Use thelimitparameter to change the number of results per page.

For example, this is an API request to list Customers, with alimitof 3:

Command Line
Select a language
cURL
Stripe CLI
Ruby
Python
PHP
Java
Node.js
Go
.NET
No results

```
curl -G https://api.stripe.com/v1/customers \
  -u "sk_test_BQokikJOvBiI2HlWgH4olfQ2:" \
  -d limit=3
```

The response from Stripe contains one page with 3 results:

Truncated API response

```
{
  "data": [
    {
      "id": "cus_005",
      "object": "customer",
      "name": "John Doe"
    },
    {
      "id": "cus_004",
      "object": "customer",
      "name": "Jane Doe"
    },
    {
      "id": "cus_003",
      "object": "customer",
      "name": "Jenny Rosen"
    }
  ],
  "has_more": true,
  /* ... */
}
```

Keep in mind the following details when using these endpoints:

- Objects are inside thedataproperty.
- Objects are in reverse chronological order, meaning the most recently created object is the first one.
- Thehas_moreproperty indicates if there are additional objects that weren’t returned in this request.

Instead of looping over thedataarray to go through objects, you should paginate results. This prevents you from missing some objects when thehas_moreparameter istrue.

## Auto-pagination

To retrieve all objects, use the auto-pagination feature. This automatically makes multiple API calls untilhas_morebecomesfalse.

Select a language
Ruby
Python
PHP
Java
Node.js
Go
.NET
No results

```
# Don't put any keys in code. See https://docs.stripe.com/keys-best-practices.
# Find your keys at https://dashboard.stripe.com/apikeys.
client = Stripe::StripeClient.new('sk_test_BQokikJOvBiI2HlWgH4olfQ2')
customers = client.v1.customers.list()

customers.auto_paging_each do |customer|
    # Do something with customer
end
```

#### Note

When using auto-pagination with a list endpoint and settingending_before, the results are in chronological order, meaning the most recently created customer is the last one.

## Manual pagination

Follow these steps to manually paginate results. This process is different when calling a list endpoint or a search endpoint.

1. Make an API call to list the objects that you want to find.
Command Line
Select a language
cURL
Stripe CLI
Ruby
Python
PHP
Java
Node.js
Go
.NET
No results

```
curl https://api.stripe.com/v1/customers \
  -u "sk_test_BQokikJOvBiI2HlWgH4olfQ2:"
```

1. In the response, check the value ofhas_more:
- If the value isfalse, you’ve retrieved all the objects.
- If the value istrue, get the ID of the last object returned, and make a new API call with thestarting_afterparameter set.

Repeat this step until you’ve retrieved all of the objects that you want to find.

Command Line
Select a language
cURL
Stripe CLI
Ruby
Python
PHP
Java
Node.js
Go
.NET
No results

```
curl -G https://api.stripe.com/v1/customers \
  -u "sk_test_BQokikJOvBiI2HlWgH4olfQ2:" \
  -d starting_after={{LAST_CUSTOMER_ID}}
```