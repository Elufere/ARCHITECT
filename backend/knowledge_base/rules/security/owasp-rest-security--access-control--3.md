# Access Control¶

> Source: [owasp-rest-security](https://cheatsheetseries.owasp.org/cheatsheets/REST_Security_Cheat_Sheet.html)

## Access Control¶

Non-public REST services must perform access control at each API endpoint. Web services in monolithic applications implement this by means of user authentication, authorization logic and session management. This has several drawbacks for modern architectures which compose multiple microservices following the RESTful style.

- in order to minimize latency and reduce coupling between services, the access control decision should be taken locally by REST endpoints
- user authentication should be centralised in a Identity Provider (IdP), which issues access tokens