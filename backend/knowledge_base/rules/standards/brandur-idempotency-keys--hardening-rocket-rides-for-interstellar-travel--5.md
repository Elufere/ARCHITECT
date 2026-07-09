# Hardening Rocket Rides for interstellar travel

> Source: [brandur-idempotency-keys](https://brandur.org/idempotency-keys)

## Hardening Rocket Rides for interstellar travel

Now that we’ve covered a few key concepts, we’re ready to
shore up Rocket Rides so that it’s resilient against any
kind of failure imaginable. Let’s put together the basic
schema, break the lifecycle up into atomic phases, and
assemble a simple implementation that will recover from
failures.

A working version (with testing) of all of this is
available in theAtomic Rocket Ridesrepository. It might be easier to download that code and
follow along.

```
git clone https://github.com/brandur/rocket-rides-atomic.git
```

### The idempotency key relation

Let’s design a Postgres schema for idempotency keys in our
app:

```
CREATE TABLE idempotency_keys (
    id              BIGSERIAL   PRIMARY KEY,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    idempotency_key TEXT        NOT NULL
        CHECK (char_length(idempotency_key) <= 100),
    last_run_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    locked_at       TIMESTAMPTZ DEFAULT now(),

    -- parameters of the incoming request
    request_method  TEXT        NOT NULL
        CHECK (char_length(request_method) <= 10),
    request_params  JSONB       NOT NULL,
    request_path    TEXT        NOT NULL
        CHECK (char_length(request_path) <= 100),

    -- for finished requests, stored status code and body
    response_code   INT         NULL,
    response_body   JSONB       NULL,

    recovery_point  TEXT        NOT NULL
        CHECK (char_length(recovery_point) <= 50),
    user_id         BIGINT      NOT NULL
);

CREATE UNIQUE INDEX idempotency_keys_user_id_idempotency_key
    ON idempotency_keys (user_id, idempotency_key);
```

There are a few notable fields here:

- idempotency_key: This is the user-specified idempotency
key. It’s good practice to send something with good
randomness like a UUID, but not necessarily required. We
constrain the field’s length so that nobody sends us
anything too exotic.We’ve madeidempotency_keyunique, but across(user_id, idempotency_key)so that it’s possible to
have the same idempotency key for different requests as
long as it’s across different user accounts.
- locked_at: A field that indicates whether this
idempotency key is actively being worked. The first API
request that creates the key will lock it automatically,
but subsequent retries will also set it to make sure that
they’re the only request doing the work.
- params: The input parameters of the request. This is
stored mostly so that we can error if the user sends two
requests with the same idempotency key but with different
parameters, but can also be used for our own backend to
push unfinished requests to completion (seethe
completionistbelow).
- recovery_point: A text label for the last phase
completed for the idempotent request (seerecovery
pointsabove). Gets an initial value
ofstartedand is set tofinishedwhen the request is
considered to be complete.

### Other schema

Recall our target API lifecycle for Rocket Rides from
above.

A typical API request to our embellished Rocket Rides backend.

Let’s bring up Postgres relations for everything else we’ll
need to build this app including audit records, rides, and
users. Given that we aim to maximize reliability, we’ll try
to follow database best practices and useNOT NULL,
unique, and foreign key constraints wherever we can.

```
--
-- A relation to hold records for every user of our app.
--
CREATE TABLE users (
    id                 BIGSERIAL       PRIMARY KEY,
    email              TEXT            NOT NULL UNIQUE
        CHECK (char_length(email) <= 255),

    -- Stripe customer record with an active credit card
    stripe_customer_id TEXT            NOT NULL UNIQUE
        CHECK (char_length(stripe_customer_id) <= 50)
);

--
-- Now that we have a users table, add a foreign key
-- constraint to idempotency_keys which we created above.
--
ALTER TABLE idempotency_keys
    ADD CONSTRAINT idempotency_keys_user_id_fkey
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE RESTRICT;

--
-- A relation that hold audit records that can help us piece
-- together exactly what happened in a request if necessary
-- after the fact. It can also, for example, be used to
-- drive internal security programs tasked with looking for
-- suspicious activity.
--
CREATE TABLE audit_records (
    id                 BIGSERIAL       PRIMARY KEY,

    -- action taken, for example "created"
    action             TEXT            NOT NULL
        CHECK (char_length(action) <= 50),

    created_at         TIMESTAMPTZ     NOT NULL DEFAULT now(),
    data               JSONB           NOT NULL,
    origin_ip          CIDR            NOT NULL,

    -- resource ID and type, for example "ride" ID 123
    resource_id        BIGINT          NOT NULL,
    resource_type      TEXT            NOT NULL
        CHECK (char_length(resource_type) <= 50),

    user_id            BIGINT          NOT NULL
        REFERENCES users ON DELETE RESTRICT
);

--
-- A relation representing a single ride by a user.
-- Notably, it holds the ID of a successful charge to
-- Stripe after we have one.
--
CREATE TABLE rides (
    id                 BIGSERIAL       PRIMARY KEY,
    created_at         TIMESTAMPTZ     NOT NULL DEFAULT now(),

    -- Store a reference to the idempotency key so that we can recover an
    -- already-created ride. Note that idempotency keys are not stored
    -- permanently, so make sure to SET NULL when a referenced key is being
    -- reaped.
    idempotency_key_id BIGINT
        REFERENCES idempotency_keys ON DELETE SET NULL,

    -- origin and destination latitudes and longitudes
    origin_lat         NUMERIC(13, 10) NOT NULL,
    origin_lon         NUMERIC(13, 10) NOT NULL,
    target_lat         NUMERIC(13, 10) NOT NULL,
    target_lon         NUMERIC(13, 10) NOT NULL,

    -- ID of Stripe charge like ch_123; NULL until we have one
    stripe_charge_id   TEXT            UNIQUE
        CHECK (char_length(stripe_charge_id) <= 50),

    user_id            BIGINT          NOT NULL
        REFERENCES users ON DELETE RESTRICT,
    CONSTRAINT rides_user_id_idempotency_key_unique UNIQUE (user_id, idempotency_key_id)
);

CREATE INDEX rides_idempotency_key_id
    ON rides (idempotency_key_id)
    WHERE idempotency_key_id IS NOT NULL;

--
-- A relation that holds our transactionally-staged jobs
-- (see "Background jobs and job staging" above).
--
CREATE TABLE staged_jobs (
    id                 BIGSERIAL       PRIMARY KEY,
    job_name           TEXT            NOT NULL,
    job_args           JSONB           NOT NULL
);
```

### Designing atomic phases

Now that we’ve got a feel for what our data should look
like, let’s break the API request into distinct atomic
phases. These are the basic rules for identifying them:

1. Upserting the idempotency key gets its own atomic phase.
2. Every foreign state mutation gets its own atomic phase.
3. After those phases have been identified,all other
operations betweenthem are grouped into atomic
phases. Even if there are 100 operations against an ACID
database between two foreign state mutations, they can
all safely belong to the same phase.

So in our example, we have an atomic phase for inserting
the idempotency key (tx1) and another for making our charge
call to Stripe (tx3) and storing the result. Every other
operation aroundtx1andtx3gets grouped together and
becomes part of two more phases,tx2andtx4.tx2throughtx4can each be reached by a recovery point
that’s set by the transaction that committed before it
(started,ride_created, andcharge_created).

API request to Rocket Rides broken into foreign state mutations and atomic phases.

### Atomic phase implementation

Our implementation for an atomic phase will wrap everything
in a transaction block (note we’re using Ruby, but this
same concept is possible in any language) and give each
phase three options for what it can return:

1. ARecoveryPointwhich sets a new recovery point. This
happens within the same transaction as the rest of the
phase so it’s all guaranteed to be atomic. Execution
continues normally into the next phase.
2. AResponsewhich sets the idempotent request’s
recovery point tofinishedand returns a response to
the user. This should be used as part of the normal
success condition, but can also be used to return early
with a non-recoverable error. Say for example that a
user’s credit card is not valid – no matter how many
times the request is retried, it will never go through.
3. ANoOpwhich indicates that program flow should
continue, but that neither a recovery point nor response
should be set.

Don’t worry about parsing the specific code too much, but
here’s what it might look like:

```
def atomic_phase(key, &block)
  error = false
  begin
    DB.transaction(isolation: :serializable) do
      ret = block.call

      if ret.is_a?(NoOp) || ret.is_a?(RecoveryPoint) || ret.is_a?(Response)
        ret.call(key)
      else
        raise "Blocks to #atomic_phase should return one of " \
          "NoOp, RecoveryPoint, or Response"
      end
    end
  rescue Sequel::SerializationFailure
    # you could possibly retry this error instead
    error = true
    halt 409, JSON.generate(wrap_error(Messages.error_retry))
  rescue
    error = true
    halt 500, JSON.generate(wrap_error(Messages.error_internal))
  ensure
    # If we're leaving under an error condition, try to unlock the idempotency
    # key right away so that another request can try again.
    if error && !key.nil?
      begin
        key.update(locked_at: nil)
      rescue StandardError
        # We're already inside an error condition, so swallow any additional
        # errors from here and just send them to logs.
        puts "Failed to unlock key #{key.id}."
      end
    end
  end
end

# Represents an action to perform a no-op. One possible option for a return
# from an #atomic_phase block.
class NoOp
  def call(_key)
    # no-op
  end
end

# Represents an action to set a new recovery point. One possible option for a
# return from an #atomic_phase block.
class RecoveryPoint
  attr_accessor :name

  def initialize(name)
    self.name = name
  end

  def call(key)
    raise ArgumentError, "key must be provided" if key.nil?
    key.update(recovery_point: name)
  end
end

# Represents an action to set a new API response (which will be stored onto an
# idempotency key). One  possible option for a return from an #atomic_phase
# block.
class Response
  attr_accessor :data
  attr_accessor :status

  def initialize(status, data)
    self.status = status
    self.data = data
  end

  def call(key)
    raise ArgumentError, "key must be provided" if key.nil?
    key.update(
      locked_at: nil,
      recovery_point: RECOVERY_POINT_FINISHED,
      response_code: status,
      response_body: data
    )
  end
end
```

In the case of a serialization error, we return a409
Conflictbecause that almost certainly means that a
concurrent request conflicted with what we were trying to
do. In a real app, you probably want to just retry the
operation right away because there’s a good chance it will
succeed this time.

For other errors we return a500 Internal Server Error.
For either type of error, we try to unlock the idempotency
key before finishing so that another request has a chance
to retry with it.

### Idempotency key upsert

When a new idempotency key value comes into the API, we’re
going to create or update a corresponding row that we’ll
use to track its progress.

The easiest case is if we’ve never seen the key before. If
so, just insert a new row with appropriate values.

If we have seen the key, lock it so that no other
requests that might be operating concurrently also try the
operation. If the key was already locked, return a409
Conflictto indicate that to the user.

A key that’s already set tofinishedis simply allowed to
fall through and have its response return on the standard
success path. We’ll see that in just a moment.

```
key = nil

atomic_phase(key) do
  key = IdempotencyKey.first(user_id: user.id, idempotency_key: key_val)

  if key
    # Programs sending multiple requests with different parameters but the
    # same idempotency key is a bug.
    if key.request_params != params
      halt 409, JSON.generate(wrap_error(Messages.error_params_mismatch))
    end

    # Only acquire a lock if the key is unlocked or its lock has expired
    # because the original request was long enough ago.
    if key.locked_at && key.locked_at > Time.now - IDEMPOTENCY_KEY_LOCK_TIMEOUT
      halt 409, JSON.generate(wrap_error(Messages.error_request_in_progress))
    end

    # Lock the key and update latest run unless the request is already
    # finished.
    if key.recovery_point != RECOVERY_POINT_FINISHED
      key.update(last_run_at: Time.now, locked_at: Time.now)
    end
  else
    key = IdempotencyKey.create(
      idempotency_key: key_val,
      locked_at:       Time.now,
      recovery_point:  RECOVERY_POINT_STARTED,
      request_method:  request.request_method,
      request_params:  Sequel.pg_jsonb(params),
      request_path:    request.path_info,
      user_id:         user.id,
    )
  end

  # no response and no need to set a recovery point
  NoOp.new
end
```

At first glance this code might not look like it’s safe
from having two concurrent requests come in in close
succession and try to the lock the same key, but it is
because the atomic phase is wrapped in aSERIALIZABLEtransaction. If two different transactions both try to lock
any one key, one of them will be aborted by Postgres.

### A directed and acyclic state machine

We’re going to implement the rest of the API request as a
simple state machines whose states are adirected acyclic
graph (DAG). Unlike a normal graph, a DAG moves only
in one direction and never cycles back on itself.

Each atomic phase will be activated from a recovery point,
which was either read from a recovered idempotency key, or
set by the previous atomic phase. We continue to move
through phases until reaching afinishedstate, upon
which the loop is broken and a response is sent back to the
user.

An idempotency key that was already finished will enter the
loop, break immediately, and send back whatever response
was stored onto it.

```
loop do
  case key.recovery_point
  when RECOVERY_POINT_STARTED
    atomic_phase(key) do
      ...
    end

  when RECOVERY_POINT_RIDE_CREATED
    atomic_phase(key) do
      ...
    end

  when RECOVERY_POINT_CHARGE_CREATED
    atomic_phase(key) do
      ....
    end

  when RECOVERY_POINT_FINISHED
    break

  else
    raise "Bug! Unhandled recovery point '#{key.recovery_point}'."
  end

  # If we got here, allow the loop to move us onto the next phase of the
  # request. Finished requests will break the loop.
end

[key.response_code, JSON.generate(key.response_body)]
```

### Initial bookkeeping

The second phase (tx2in the diagram above) is simple:
create a record for the ride in our local database, insert
an audit record, and set a new recovery point toride_created.

```
atomic_phase(key) do
  ride = Ride.create(
    idempotency_key_id: key.id,
    origin_lat:         params["origin_lat"],
    origin_lon:         params["origin_lon"],
    target_lat:         params["target_lat"],
    target_lon:         params["target_lon"],
    stripe_charge_id:   nil, # no charge created yet
    user_id:            user.id,
  )

  # in the same transaction insert an audit record for what happened
  AuditRecord.insert(
    action:        AUDIT_RIDE_CREATED,
    data:          Sequel.pg_jsonb(params),
    origin_ip:     request.ip,
    resource_id:   ride.id,
    resource_type: "ride",
    user_id:       user.id,
  )

  RecoveryPoint.new(RECOVERY_POINT_RIDE_CREATED)
end
```

### Calling Stripe

With basic records in place, it’s time to try our foreign
state mutation by trying to charge the customer via Stripe.
Here we initiate a charge for $20 using a Stripe customer
ID that was already stored on their user record. On
success, update the ride created in the last step with the
new Stripe charge ID and set recovery pointcharge_created.

```
atomic_phase(key) do
  # retrieve a ride record if necessary (i.e. we're recovering)
  ride = Ride.first(idempotency_key_id: key.id) if ride.nil?

  # if ride is still nil by this point, we have a bug
  raise "Bug! Should have ride for key at #{RECOVERY_POINT_RIDE_CREATED}." \
    if ride.nil?

  raise "Simulated fail with `raise_error` param." if raise_error

  # Rocket Rides is still a new service, so during our prototype phase
  # we're going to give $20 fixed-cost rides to everyone, regardless of
  # distance. We'll implement a better algorithm later to better
  # represent the cost in time and jetfuel on the part of our pilots.
  begin
    charge = Stripe::Charge.create({
      amount:      20_00,
      currency:    "usd",
      customer:    user.stripe_customer_id,
      description: "Charge for ride #{ride.id}",
    }, {
      # Pass through our own unique ID rather than the value
      # transmitted to us so that we can guarantee uniqueness to Stripe
      # across all Rocket Rides accounts.
      idempotency_key: "rocket-rides-atomic-#{key.id}"
    })
  rescue Stripe::CardError
    # Sets the response on the key and short circuits execution by
    # sending execution right to 'finished'.
    Response.new(402, wrap_error(Messages.error_payment(error: $!.message)))
  rescue Stripe::StripeError
    Response.new(503, wrap_error(Messages.error_payment_generic))
  else
    ride.update(stripe_charge_id: charge.id)
    RecoveryPoint.new(RECOVERY_POINT_CHARGE_CREATED)
  end
end
```

The call to Stripe produces a few possibilities for
unrecoverable errors (i.e. an error that no matter how many
times is retried will never see the call succeed). If we
run into one, set the request tofinishedand return an
appropriate response. This might occur if the credit card
was invalid or the transaction was otherwise declined by
the payment gateway.

### Send receipt and finish

Now that our charge has been persisted, the next step is to
send a receipt to the user. Making an external mail call
would normally require its own foreign state mutation, but
because we’re using a transactionally-staged job drain, we
get a guarantee that the operation commits along with the
rest of the transaction.

```
atomic_phase(key) do
  StagedJob.insert(
    job_name: "send_ride_receipt",
    job_args: Sequel.pg_jsonb({
      amount:   20_00,
      currency: "usd",
      user_id:  user.id
    })
  )
  Response.new(201, wrap_ok(Messages.ok))
end
```

The final step is to set a response telling the user that
everything worked as expected. We’re done!