# Management endpoints¶

> Source: [owasp-rest-security](https://cheatsheetseries.owasp.org/cheatsheets/REST_Security_Cheat_Sheet.html)

## Management endpoints¶

- Avoid exposing management endpoints via Internet.
- If management endpoints must be accessible via the Internet, make sure that users must use a strong authentication mechanism, e.g. multi-factor.
- Expose management endpoints via different HTTP ports or hosts preferably on a different NIC and restricted subnet.
- Restrict access to these endpoints by firewall rules  or use of access control lists.