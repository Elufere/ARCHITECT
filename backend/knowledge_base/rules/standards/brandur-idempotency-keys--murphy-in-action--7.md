# Murphy in action

> Source: [brandur-idempotency-keys](https://brandur.org/idempotency-keys)

## Murphy in action

Now that we have all the pieces in place, let’s assume the
truth ofMurphy’s Lawand imagine some
scenarios that could go wrong while a client app is talking
to the newAtomic Rocket Ridesbackend:

- The client makes a request, but the connection breaks
before it reaches the backend:The client, having used
an idempotency key, knows that retries are safe and so
retries. The next attempt succeeds.
- Two requests try to create an idempotency key at the
same time:AUNIQUEconstraint in the database
guarantees that only one request can succeed. One goes
through, and the other gets a409 Conflict.
- An idempotency key is created, but the database goes
down and it fails soon after:The client continues to
retry against the API until it comes back online. Once it
does, the created key is recovered and the request is
continued.
- Stripe is down:The atomic phase containing the Stripe
request fails, and the API responds with an error that
tells the client to retry. They continue to do so until
Stripe comes back online and the charge succeeds.
- A server process dies while waiting for a response from
Stripe:Luckily, the call to Stripe was also made with
its own idempotency key. The client retries and a new
call to Stripe is invoked with the same key. Stripe’s own
idempotency guarantees ensure that we haven’t
double-charged our user.
- A bad deploy 500s all requests midway through:Developers scramble and deploy a fix for the bug. After
it’s out, clients retry and the original requests succeed
along the newly bug-free path. If the fix took so long to
get out that clients have long since gone away, then the
completer process pushes them through.

Our care around implementing a failsafe design has paid off
– the system is safe despite a wide variety of possible
failures.