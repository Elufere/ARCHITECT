# JWT¶

> Source: [owasp-rest-security](https://cheatsheetseries.owasp.org/cheatsheets/REST_Security_Cheat_Sheet.html)

## JWT¶

There seems to be a convergence towards usingJSON Web Tokens(JWT) as the format for security tokens. JWTs are JSON data structures containing a set of claims that can be used for access control decisions. A cryptographic signature or message authentication code (MAC) can be used to protect the integrity of the JWT.

- Ensure JWTs are integrity protected by either a signature or a MAC. Do not allow the unsecured JWTs:{"alg":"none"}.Seehere
- In general, signatures should be preferred over MACs for integrity protection of JWTs.

If MACs are used for integrity protection, every service that is able to validate JWTs can also create new JWTs using the same key. This means that all services using the same key have to mutually trust each other. Another consequence of this is that a compromise of any service also compromises all other services sharing the same key. Seeherefor additional information.

The relying party or token consumer validates a JWT by verifying its integrity and claims contained.

- A relying party must verify the integrity of the JWT based on its own configuration or hard-coded logic. It must not rely on the information of the JWT header to select the verification algorithm. Seehereandhere

Some claims have been standardized and should be present in JWT used for access controls. At least the following of the standard claims should be verified:

- issor issuer - is this a trusted issuer? Is it the expected owner of the signing key?
- audor audience - is the relying party in the target audience for this JWT?
- expor expiration time - is the current time before the end of the validity period of this token?
- nbfor not before time - is the current time after the start of the validity period of this token?

As JWTs contain details of the authenticated entity (user etc.) a disconnect can occur between the JWT and the current state of the users session, for example, if the session is terminated earlier than the expiration time due to an explicit logout or an idle timeout. When an explicit session termination event occurs, a unique, server-issued identifier (thejticlaim, optionally combined withaud) should be submitted to a denylist on the API which will invalidate that JWT for any requests until the expiration of the token. See theJSON_Web_Token_Cheat_Sheetfor further details.