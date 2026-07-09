# Don't use char(n) even for fixed-length identifiers

> Source: [postgres-dont-do-this](https://wiki.postgresql.org/wiki/Don't_Do_This)

## Don't use char(n) even for fixed-length identifiers

Sometimes people respond to "don't usechar(n)" with "but my values must always be exactly N characters long" (e.g. country codes, hashes, or identifiers from some other system).It is still a bad idea to usechar(n)even in these cases.

Usetext, or a domain over text, withCHECK(length(VALUE)=3)orCHECK(VALUE ~ '^[[:alpha:]]{3}$')or similar.

### Why not?

Becausechar(n)doesn't reject values that are too short, it just silently pads them with spaces. So there's no actual benefit over usingtextwith a constraint that checks for the exact length. As a bonus, such a check can also verify that the value is in the correct format.

Remember,there is no performance benefit whatsoever to usingchar(n)overvarchar(n).In fact the reverse is true. One particular problem that comes up is that if you try and compare achar(n)field against a parameter where the driver has explicitly specified a type oftextorvarchar, you may be unexpectedly unable to use an index for the comparison. This can be hard to debug since it doesn't show up on manual queries.

### When should you?

Never.