# Transport Security¶

> Source: [owasp-grpc-security](https://cheatsheetseries.owasp.org/cheatsheets/gRPC_Security_Cheat_Sheet.html)

## Transport Security¶

### Always Use TLS in Production¶

Production deployments need TLS encryption to protect against eavesdropping and man-in-the-middle attacks.

```
// Go - Secure server with TLS
creds, err := credentials.NewServerTLSFromFile(certFile, keyFile)
if err != nil {
    log.Fatalf("Failed to load TLS credentials: %v", err)
}
s := grpc.NewServer(grpc.Creds(creds))
```

Configure TLS 1.2 or higher with strong cipher suites, and disable weak protocols and ciphers.

### Implement Mutual TLS (mTLS) for Service-to-Service Communication¶

mTLS provides mutual authentication where both client and server verify each other's certificates, enabling zero-trust communication.

```
// Go - mTLS client configuration
cert, err := tls.LoadX509KeyPair(clientCertFile, clientKeyFile)
caCert, err := ioutil.ReadFile(caCertFile)
caCertPool := x509.NewCertPool()
caCertPool.AppendCertsFromPEM(caCert)

creds := credentials.NewTLS(&tls.Config{
    Certificates: []tls.Certificate{cert},
    RootCAs:      caCertPool,
})
conn, err := grpc.Dial(address, grpc.WithTransportCredentials(creds))
```

Use short-lived certificates (90 days or less) with automated rotation to limit the impact of compromised keys.