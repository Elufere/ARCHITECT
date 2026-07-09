# Introduction¶

> Source: [owasp-authentication](https://cheatsheetseries.owasp.org/cheatsheets/Authentication_Cheat_Sheet.html)

## Introduction¶

Authentication(AuthN) is the process of verifying that an individual, entity, or website is who or what it claims to be by determining the validity of one or more authenticators (like passwords, fingerprints, or security tokens) that are used to back up this claim.

Digital Identityis the unique representation of a subject engaged in an online transaction. A digital identity is always unique in the context of a digital service but does not necessarily need to be traceable back to a specific real-life subject.

Identity Proofingestablishes that a subject is actually who they claim to be. This concept is related to KYC concepts and it aims to bind a digital identity with a real person.

Session Managementis a process by which a server maintains the state of an entity interacting with it. This is required for a server to remember how to react to subsequent requests throughout a transaction. Sessions are maintained on the server by a session identifier which can be passed back and forth between the client and server when transmitting and receiving requests. Sessions should be unique per user and computationally very difficult to predict. TheSession Management Cheat Sheetcontains further guidance on the best practices in this area.