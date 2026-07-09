# Authentication and Authorization¶

> Source: [owasp-grpc-security](https://cheatsheetseries.owasp.org/cheatsheets/gRPC_Security_Cheat_Sheet.html)

## Authentication and Authorization¶

### Implement Strong Authentication¶

Implement authentication checks for each protected service method.

#### Token-Based Authentication¶

```
// Go - JWT token validation interceptor
func authInterceptor(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
    md, ok := metadata.FromIncomingContext(ctx)
    if !ok {
        return nil, status.Errorf(codes.Unauthenticated, "missing metadata")
    }

    tokens := md["authorization"]
    if len(tokens) == 0 {
        return nil, status.Errorf(codes.Unauthenticated, "missing authorization token")
    }

    token := strings.TrimPrefix(tokens[0], "Bearer ")
    if !validateJWT(token) {
        return nil, status.Errorf(codes.Unauthenticated, "invalid token")
    }

    return handler(ctx, req)
}
```

#### API Key Authentication¶

```
// Go - API key validation
func validateAPIKey(ctx context.Context) error {
    md, ok := metadata.FromIncomingContext(ctx)
    if !ok {
        return status.Error(codes.Unauthenticated, "missing metadata")
    }

    keys := md["x-api-key"]
    if len(keys) == 0 || !isValidAPIKey(keys[0]) {
        return status.Error(codes.Unauthenticated, "invalid API key")
    }
    return nil
}
```

Implement token expiration and refresh mechanisms with short-lived tokens (15-60 minutes). Avoid embedding credentials in gRPC method parameters - use metadata headers.

### Enforce Granular Authorization¶

Implement method-level authorization checks based on the principle of least privilege.

```
// Go - Role-based authorization
func authorizeMethod(ctx context.Context, methodName string, userRoles []string) error {
    requiredRole, exists := methodPermissions[methodName]
    if !exists {
        return status.Errorf(codes.PermissionDenied, "method not found")
    }

    for _, role := range userRoles {
        if role == requiredRole {
            return nil
        }
    }

    return status.Errorf(codes.PermissionDenied, "insufficient permissions")
}
```

Log all authorization failures to detect potential attacks and compliance violations.