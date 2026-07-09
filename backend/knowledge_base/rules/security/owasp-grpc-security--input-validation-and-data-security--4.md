# Input Validation and Data Security¶

> Source: [owasp-grpc-security](https://cheatsheetseries.owasp.org/cheatsheets/gRPC_Security_Cheat_Sheet.html)

## Input Validation and Data Security¶

### Validate All Protocol Buffer Messages¶

Protocol Buffers provide type safety but not business logic validation. Always perform thorough server-side validation.

```
// Use protoc-gen-validate for automatic validation
syntax = "proto3";
import "validate/validate.proto";

message CreateUserRequest {
  string email = 1 [(validate.rules).string.email = true];
  string name = 2 [(validate.rules).string = {min_len: 1, max_len: 100}];
  int32 age = 3 [(validate.rules).int32 = {gte: 0, lte: 150}];
}
```

Use allowlist validation for string inputs to prevent unexpected characters and injection attempts.

### Prevent Injection Attacks¶

Validate user input carefully when used in database queries or system operations.

```
// Go - Safe database query with parameterization
func getUserByEmail(email string) (*User, error) {
    if !isValidEmail(email) {
        return nil, errors.New("invalid email format")
    }

    query := "SELECT id, name, email FROM users WHERE email = ?"
    row := db.QueryRow(query, email)

    var user User
    err := row.Scan(&user.ID, &user.Name, &user.Email)
    return &user, err
}
```

Always use prepared statements for database operations to preventSQL injection.

### Implement Message Size Limits¶

gRPC's streaming capabilities allow clients to send arbitrarily large messages, potentially exhausting server memory and triggering denial-of-service conditions. Set clear limits on message sizes.

```
// Go - Set message size limits
s := grpc.NewServer(
    grpc.MaxRecvMsgSize(4*1024*1024), // 4MB max receive
    grpc.MaxSendMsgSize(4*1024*1024), // 4MB max send
```

Limit streaming sessions and message counts to prevent resource exhaustion. Monitor and enforce maximum messages per stream and maximum session duration.