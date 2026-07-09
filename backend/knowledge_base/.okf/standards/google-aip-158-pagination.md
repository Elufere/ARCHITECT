---
source_url: https://google.aip.dev/158
category: standards
title: Google AIP-158 — Pagination
slug: google-aip-158-pagination
date_scraped: "2026-07-07 10:41:39 UTC"
description: "Page tokens, opaque cursors, backward-compatibility rules for paginated APIs"
---
#### AIP-158

# Pagination

APIs often need to provide collections of data, most commonly in theListstandard method. However, collections can often be arbitrarily sized, and also
often grow over time, increasing lookup time as well as the size of the
responses being sent over the wire. Therefore, it is important that collections
be paginated.

## Guidance

RPCs returning collections of datamustprovide paginationat the outset,
as it is abackwards-incompatible changeto add
pagination to an existing method.

```
// The request structure for listing books.
message ListBooksRequest {
  // The parent, which owns this collection of books.
  // Format: publishers/{publisher}
  string parent = 1 [
    (google.api.field_behavior) = REQUIRED,
    (google.api.resource_reference) = {
      child_type: "library.googleapis.com/Book"
    }];

  // The maximum number of books to return. The service may return fewer than
  // this value.
  // If unspecified, at most 50 books will be returned.
  // The maximum value is 1000; values above 1000 will be coerced to 1000.
  int32 page_size = 2;

  // A page token, received from a previous `ListBooks` call.
  // Provide this to retrieve the subsequent page.
  //
  // When paginating, all other parameters provided to `ListBooks` must match
  // the call that provided the page token.
  string page_token = 3;
}

// The response structure from listing books.
message ListBooksResponse {
  // The books from the specified publisher.
  repeated Book books = 1;

  // A token that can be sent as `page_token` to retrieve the next page.
  // If this field is omitted, there are no subsequent pages.
  string next_page_token = 2;
}
```

- Request messages for collectionsshoulddefine anint32 page_sizefield, allowing users to specify the maximum number of results to return.Thepage_sizefieldmust notbe required.If the user does not specifypage_size(or specifies0), the API
  chooses an appropriate default, which the APIshoulddocument. The APImust notreturn an error.If the user specifiespage_sizegreater than the maximum permitted by the
  API, the APIshouldcoerce down to the maximum permitted page size.If the user specifies a negative value forpage_size, the APImustsend anINVALID_ARGUMENTerror.The APImayreturn fewer results than the number requested (including
  zero results), even if not at the end of the collection.
- Request messages for collectionsshoulddefine astring page_tokenfield, allowing users to advance to the next page in the collection.Thepage_tokenfieldmust notbe required.If the user changes thepage_sizein a request for subsequent pages, the
  servicemusthonor the new page size.The user is expected to keep all other arguments to the RPC the same; if
  any arguments are different, the APIshouldsend anINVALID_ARGUMENTerror.
- The responsemust notbe a streaming response.
- Response messages for collectionsshoulddefine astring next_page_tokenfield, providing the user with a page token that may
  be used to retrieve the next page.The field containing pagination resultsshouldbe the first field in
  the message and have a field number of1. Itshouldbe a repeated
  field containing a list of resources constituting a single page of results.If the end of the collection has been reached, thenext_page_tokenfieldmustbe empty. This is theonlyway to communicate
  "end-of-collection" to users.If the end of the collection has not been reached (or if the API can not
  determine in time), the APImustprovide anext_page_token.
- Response messages for collectionsmayprovide anint32 total_sizefield, providing the user with the total number of items in the list.This totalmaybe an estimate (but the APIshouldexplicitly
  document that).

### Skipping results

The request definition for a paginated operationmaydefine anint32 skipfield to allow the user to skip results.

Theskipvaluemustrefer to the number of individual resources to skip,
not the number of pages.

For example:

- A request with no page token and askipvalue of30returns a single page
  of results starting with the 31st result.
- A request with a page token corresponding to the 51st result (because the
  first 50 results were returned on the first page) and askipvalue of30returns a single page of results starting with the 81st result.

If askipvalue is provided that cannot be fulfilled e.g. due to latency of
querying a massive data set, the responsemustbe200 OKwith an empty
result set. If it isknownto put the cursor beyond the total size of the
collection, the responsemust notinclude anext_page_token.

### Opacity

Page tokens provided by APIsmustbe opaque (but URL-safe) strings, andmust notbe user-parseable. This is because if users are able to
deconstruct these,they will do so. This effectively makes the implementation
details of your API's pagination become part of the API surface, and it becomes
impossible to update those details without breaking users.

Warning:Base-64 encoding an otherwise-transparent page token isnota
sufficient obfuscation mechanism.

For page tokens which do not need to be stored in a database, and which do not
contain sensitive data, an APImayobfuscate the page token by defining an
internal protocol buffer message with any data needed, and send the serialized
proto, base-64 encoded.

Page tokensmustbe limited to providing an indication of where to continue
the pagination process only. Theymust notprovide any form of
authorization to the underlying resources, and authorizationmustbe
performed on the request as with any other regardless of the presence of a page
token.

### Expiring page tokens

Many APIs store page tokens in a database internally. In this situation, APIsmayexpire page tokens a reasonable time after they have been sent, in
order not to needlessly store large amounts of data that is unlikely to be
used. It is not necessary to document this behavior.

Note:While a reasonable time may vary between APIs, a good rule of thumb
is three days.

## Backwards compatibility

Adding pagination to an existing RPC is a backwards-incompatible change. This
may seem strange; adding fields to proto messages is generally backwards
compatible. However, this change isbehaviorallyincompatible.

Consider a user whose collection has 75 resources, and who has already written
and deployed code. If the API later adds pagination fields, and sets the
default to 50, then that user's code breaks; it was getting all resources, and
now is only getting the first 50 (and does not know to advance pagination).
Even if the API set a higher default limit, such as 100, the user's collection
could grow, andthenthe code would break.

Additionally,client libraries implement automatic
pagination, typically representing paginated
RPCs using different method signatures to unpaginated ones. This means that
adding pagination to a previously-unpaginated method causes a breaking change
in those libraries.

For this reason, it is important to always add pagination to RPCs returning
collectionsup front; they are consistently important, and they can not be
added later without causing problems for existing users.

Warning:This also entails that, in addition to presenting the pagination
fields, theymustbeactually implementedwith a non-infinite default
value. Implementing an in-memory version (which might fetch everything then
paginate) is reasonable for initially-small collections.

## Rationale

### Degradedskipresponse

Large collections, complex queries, and globally distributed data can all
contribute to a paginated method being unable to quickly or confidently fulfill
a givenskiprequest. Backend queries can timeout, data collation can take
time, and the end user experience need not suffer as a result. In such cases,
the pagination interface can be leveraged to keep the client engaged by
providing anext_page_token, while the service collects an appropriate result.
When the service has definitively determined that the requestedskipexceeds
the available results, the pagination interface is again applied andnext_page_tokenis omitted to signal the end of results.

## Changelog

- 2025-07-08: Clarify degraded skip response guidance
- 2020-05-24: Clarified that adding pagination breaks client
  libraries.
- 2020-05-13: Added guidance for skipping results.
- 2020-08-24: Clarified that responses are not streaming responses.
- 2020-06-24: Clarified that page size is always optional for users.
- 2019-02-12: Added guidance on the field being paginated over.
- 2019-08-01: Changed the examples from "shelves" to "publishers", to
  present a better example of resource ownership.
- 2019-07-19: Update the opacity requirement from "should" to "must".