# Don't use psql -W or --password

> Source: [postgres-dont-do-this](https://wiki.postgresql.org/wiki/Don't_Do_This)

## Don't use psql -W or --password

Don't usepsql -Worpsql --password.

### Why not?

Using the --password or -W flags will tellpsqlto prompt you for a password, before trying to connect to the server - so you'll be prompted for a password even if the server doesn't require one.

It's never required, as if the server does require a password psql will prompt you for one, and it can be very confusing when setting up permissions. If you're connecting with -W to a server configured to allow you access viapeerauthentication you may think that it's requiring a password when it really isn't. And if the user you're logging in as doesn't have a password set or you enter the wrong password at the prompt you'll still be logged in and think you have the right password - but you won't be able to log in from other clients (that connect via localhost) or when logged in as other users.

### When should you?

Never, pretty much. It will save a round trip to the server but that's about it.