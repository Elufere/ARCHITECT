---
source_url: https://google.aip.dev/121
category: standards
title: Google AIP-121 — Resource-Oriented Design
slug: google-aip-121-resource-design
date_scraped: "2026-07-07 10:41:43 UTC"
description: "Resource-oriented API design principles — nouns over verbs, standard methods"
---
#### AIP-121

# Resource-oriented design

Resource-oriented design is a pattern for specifyingRPCAPIs, based on
several high-level design principles (most of which are common to recent public
HTTP APIs):

- The fundamental building blocks of an API are individually-namedresources(nouns) and the relationships and hierarchy that exist between them.
- A small number of standardmethods(verbs) provide the semantics for most
  common operations. However, custom methods are available in situations where
  the standard methods do not fit.
- Stateless protocol: Each interaction between the client and the server is
  independent, and both the client and server have clear roles.

Readers might notice similarities between these principles and some principles
ofREST; resource-oriented design borrows many principles from REST, while
also defining its own patterns where appropriate.

## Guidance

When designing an API, consider the following (roughly in logical order):

- The resources (nouns) the API will provide
- The relationships and hierarchies between those resources
- The schema of each resource
- The methods (verbs) each resource provides, relying as much as possible on
  the standard verbs.

### Resources

A resource-oriented APIshouldgenerally be modeled as a resource
hierarchy, where each node is either a simple resource or a collection of
resources.

Acollectioncontains resources ofthe same type. For example, a publisher
has the collection of books that it publishes. A resource usually has fields,
and resources may have any number of sub-resources (usually collections).

Note:While there is some conceptual alignment between storage systems and
APIs, a service with a resource-oriented API is not necessarily a database, and
has enormous flexibility in how it interprets resources and methods. API
designersshould notexpect that their API will be reflective of their
database schema. In fact, having an API that is identical to the underlying
database schema is actually an anti-pattern, as it tightly couples the surface
to the underlying system.

### Methods

Resource-oriented APIs emphasize resources (data model) over the methods
performed on those resources (functionality). A typical resource-oriented API
exposes a large number of resources with a small number of methods on each
resource. The methods can be either the standard methods (Get,List,Create,Update,Delete), orcustom methods.

If the request to or the response from a standard method (or a custom method in
the sameservice)isthe resource orcontainsthe resource, the
resource schema for that resource across all methodsmustbe the same.

| Standard method | Request | Response |
| --- | --- | --- |
| Create | Contains the resource | Is the resource |
| Get | None | Is the resource |
| Update | Contains the resource | Is the resource |
| Delete | None | None |
| List | None | Contains the resources |

The table above describes each standard method's relationship to the resource,
where "None" indicates that the resource neitherisnoris containedin
the request or the response

A resourcemustsupport at minimumGet: clients must be
able to validate the state of resources after performing a mutation such
asCreate,Update, orDelete.

A resourcemustalso supportList, except forsingleton resourceswhere more than one resource is not possible.

Note:A custom method in resource-oriented design doesnotentail
defining a new or custom HTTP verb. Custom methods use traditional HTTP verbs
(usuallyPOST) and define the custom verb in the URI.

APIsshouldprefer standard methods over custom methods; the purpose of
custom methods is to define functionality that does not cleanly map to any of
the standard methods. Custom methods offer the same design freedom as
traditional RPC APIs, which can be used to implement common programming
patterns, such as database transactions, import and export, or data analysis.

### Strong Consistency

For methods that operate on themanagement plane, the completion of those
operations (either successful or with an error, long-running operation, or
synchronous)mustmean that the state of the resource's existence and all
user-settable values have reached a steady-state.

Output onlyvalues unrelated to the resourcestateshouldalso have
reached a steady-state for values that are related to the resourcestate.

Examples include:

- Following a successful create that is the latest mutation on a resource, a get
  request for a resourcemustreturn the resource.
- Following a successful update that is the latest mutation on a resource, a get
  request for a resourcemustreturn the final values from the update
  request.
- Following a successful delete that is the latest mutation on a resource, a get
  request for a resourcemustreturnNOT_FOUND(or the resource with theDELETEDstate value in the case ofsoft delete)

Clients of resource-oriented APIs often need to orchestrate multiple operations
in sequence (e.g. create resource A, create resource B which depends on A), and
ensuring that resources immediately reflect steady user state after an operation
is complete ensures clients can rely on method completion as a signal to begin
the next operation.

Output onlyfields ideally would follow the same guidelines, but as
these fields can often represent a resource's live state, it's sometimes
necessary for these values to change after a successful mutation operation to
reflect a state change.

### Stateless protocol

As with most public APIs available today, resource-oriented APIsmustoperate over astateless protocol: The fundamental behavior of any
individual request is independent of other requests made by the caller.
This is to say, each request happens in isolation of other requests made by that
client or another, and resources exposed by an API are directly addressable
without needing to apply a series of specific requests to "reach" the desired
resource.

In an API with a stateless protocol, the server has the responsibility for
persisting data, which may be shared between multiple clients, while clients
have sole responsibility and authority for maintaining the application state.

### Cyclic References

The relationship between resources, such as withresource references,mustbe representable via adirected acyclic graph. The parent-child
relationship alsomustbe acyclic, and as perAIP-124a given resource
instance will only have one canonical parent resource.

A cyclic relationship between resources increases the complexity of managing
resources. Consider resources A and B that refer to
each other. The process to create said resources are:

1. create resource A without a reference to B. Retrieve id for resource A.
2. create resource B with a reference to A. Retrieve id for resource B.
3. update resource A with the reference to B.

The delete operation may also become more complex, due to reasoning about which
resource must be dereferenced first for a successful deletion.

This requirement does not apply to relationships that are expressed viaoutput onlyfields, as they do not require the user to specify the values
and in turn do not increase resource management complexity.

## Changelog

- 2024-07-08: Clarify acyclic nature of parent-child relationship.
- 2023-08-24: Added guidance on consistency guarantees of methods.
- 2023-07-23: Clarify stateless protocol definition.
- 2023-01-21: Explicitly require matching schema across standard methods.
- 2022-12-19: Added a section requiring Get and List.
- 2022-11-02: Added a section restricting resource references.
- 2019-08-01: Changed the examples from "shelves" to "publishers", to
  present a better example of resource ownership.