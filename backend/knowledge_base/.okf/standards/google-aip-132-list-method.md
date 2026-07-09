---
source_url: https://google.aip.dev/132
category: standards
title: Google AIP-132 — Standard List Method
slug: google-aip-132-list-method
date_scraped: "2026-07-07 10:41:41 UTC"
description: "List method conventions\: page_size, page_token, filtering semantics"
---
#### AIP-132

# Standard methods: List

In many APIs, it is customary to make aGETrequest to a collection's URI
(for example,/v1/publishers/1/books) in order to retrieve a list of
resources, each of which lives within that collection.

Resource-oriented design (AIP-121) honors this pattern through theListmethod. These RPCs accept the parent collection (and potentially some other
parameters), and return a list of responses matching that input.

## Guidance

APIsmustprovide aListmethod for resources unless the resource is asingleton. The purpose of theListmethod is to return data from a finite
collection (generally singular unless the operation supportsreading across
collections).

List methods are specified using the following pattern:

```
rpc ListBooks(ListBooksRequest) returns (ListBooksResponse) {
  option (google.api.http) = {
    get: "/v1/{parent=publishers/*}/books"
  };
  option (google.api.method_signature) = "parent";
}
```

- The RPC's namemustbegin with the wordList. The remainder of the RPC
  nameshouldbe the plural form of the resource being listed.
- The request and response messagesmustmatch the RPC name, withRequestandResponsesuffixes.
- The HTTP verbmustbeGET.
- The collection whose resources are being listedshouldmap to the URI
  path.The collection's parent resourceshouldbe calledparent, andshouldbe the only variable in the URI path. All remaining parametersshouldmap to URI query parameters.The collection identifier (booksin the above example)mustbe a
  literal string.
- Thebodykey in thegoogle.api.httpannotationmustbe omitted.
- If the resource being listed is not a top-level resource, thereshouldbe exactly onegoogle.api.method_signatureannotation, with a value of"parent". If the resource being listed is a top-level resource, thereshouldbe either nogoogle.api.method_signatureannotation, or exactly
  onegoogle.api.method_signatureannotation, with a value of"".

### Request message

List methods implement a common request message pattern:

```
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
```

- Aparentfieldmustbe included unless the resource being listed is a
  top-level resource. Itshouldbe calledparent.The fieldshouldbeannotated as required.The fieldmustidentify theresource typeof the resource
  being listed.
- Thepage_sizeandpage_tokenfields, which support pagination,mustbe specified on all list request messages. For more information, seeAIP-158.The comment above thepage_sizefieldshoulddocument the maximum
  allowed value, as well as the default value if the field is omitted (or set
  to0). If preferred, the APImaystate that the server will use a
  sensible default. This defaultmaychange over time.If a user provides a value greater than the maximum allowed value, the APIshouldcoerce the value to the maximum allowed.If a user provides a negative or other invalid value, the APImustsend
  anINVALID_ARGUMENTerror.
- Thepage_tokenfieldmustbe included on all list request messages.
- The request messagemayinclude fields for common design patterns
  relevant to list methods, such asstring filterandstring order_by.
- The request messagemust notcontain any other required fields, andshould notcontain other optional fields except those described in this
  or another AIP.

Note:List methodsshouldreturn the same results for any user that has
permission to make a successful List request on the collection. Search methods
are more relaxed on this.

### Response message

List methods implement a common response message pattern:

```
message ListBooksResponse {
  // The books from the specified publisher.
  repeated Book books = 1;

  // A token, which can be sent as `page_token` to retrieve the next page.
  // If this field is omitted, there are no subsequent pages.
  string next_page_token = 2;
}
```

- The response messagemustinclude one repeated field corresponding to the
  resources being returned, andshould notinclude any other repeated
  fields unless described in another AIP (for example,AIP-217).The responseshouldusually include fully-populated resources unless
  there is a reason to return a partial response (seeAIP-157).
- Thenext_page_tokenfield, which supports pagination,mustbe included
  on all list response messages. Itmustbe set if there are subsequent
  pages, andmust notbe set if the response represents the final page. For
  more information, seeAIP-158.
- The messagemayinclude aint32 total_size(orint64 total_size)
  field with the number of items in the collection.The valuemaybe an estimate (the fieldshouldclearly document
  this if so).If filtering is used, thetotal_sizefieldshouldreflect the size of
  the collectionafterthe filter is applied.

### Ordering

Listmethodsmayallow clients to specify sorting order; if they do, the
request messageshouldcontain astring order_byfield.

- Valuesshouldbe a comma separated list of fields. For example:"foo,bar".
- The default sorting order is ascending. To specify descending order for a
  field, users append a" desc"suffix; for example:"foo desc, bar".
- Redundant space characters in the syntax are insignificant."foo, bar desc"," foo , bar desc ", and"foo,bar desc"are all
  equivalent.
- Subfields are specified with a.character, such asfoo.baroraddress.street.
- The resulting list ordershouldbe based on the field type's natural
  comparator e.g. numerics ordered numerically, strings ordered
  lexicographically, etc. However, APIsmaychoose to use a different
  ordering; if so, itmustbe documented in theorder_bydefinition.Furthermore,well-knowntypes, likeTimestampandDurationare
  compared as their representative type;Timestampis compared as time e.g.
  before or after,Durationis compared as a quantity e.g. more or less.
TODO(#220): Add a reference to [AIP-161](/161) once it is written.

Note:Only include ordering if there is an established need to do so. It is
always possible to add ordering later, but removing it is a breaking change.

### Filtering

List methodsmayallow clients to specify filters; if they do, the request
messageshouldcontain astring filterfield. Filtering is described in
more detail inAIP-160.

Note:Only include filtering if there is an established need to do so. It
is always possible to add filtering later, but removing it is a breaking
change.

### Soft-deleted resources

Some APIs need to "soft delete" resources, marking them as deleted or
pending deletion (and optionally purging them later).

APIs that do thisshould notinclude deleted resources by default in list
requests. APIs with soft deletion of a resourceshouldinclude abool show_deletedfield in the list request that, if set, will cause
soft-deleted resources to be included.

### Errors

Seeerrors, in particularwhen to use PERMISSION_DENIED and
NOT_FOUND errors.

## Further reading

- For details on pagination, seeAIP-158.
- For listing across multiple parent collections, seeAIP-159.

## Changelog

- 2025-02-25: Require documentation for ordering not matching field type
  with clarification on ordering of well-known types.
- 2023-03-22: Fix guidance wording to mentionAIP-159.
- 2023-03-17: Align withAIP-122and make Get a must.
- 2022-11-04: Aggregated error guidance toAIP-193.
- 2022-06-02: Changed suffix descriptions to eliminate superfluous "-".
- 2020-09-02: Add link to the filtering AIP.
- 2020-08-14: Added error guidance for permission denied cases.
- 2020-06-08: Added guidance on returning the full resource.
- 2020-05-19: Removed requirement to document ordering behavior.
- 2020-04-15: Added guidance on List permissions.
- 2019-10-18: Added guidance on annotations.
- 2019-08-01: Changed the examples from "shelves" to "publishers", to
  present a better example of resource ownership.
- 2019-07-30: Added guidance about documenting the ordering behavior.
- 2019-05-29: Added an explicit prohibition on arbitrary fields in standard
  methods.