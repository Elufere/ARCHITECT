# Is the API Vulnerable?

> Source: [owasp-api4-resource-consumption](https://github.com/OWASP/API-Security/blob/master/editions/2023/en/0xa4-unrestricted-resource-consumption.md)

## Is the API Vulnerable?

Satisfying API requests requires resources such as network bandwidth, CPU,
memory, and storage. Sometimes required resources are made available by service
providers via API integrations, and paid for per request, such as sending
emails/SMS/phone calls, biometrics validation, etc.

An API is vulnerable if at least one of the following limits is missing or set
inappropriately (e.g. too low/high):

* Execution timeouts
* Maximum allocable memory
* Maximum number of file descriptors
* Maximum number of processes
* Maximum upload file size
* Number of operations to perform in a single API client request (e.g. GraphQL
  batching)
* Number of records per page to return in a single request-response
* Third-party service providers' spending limit