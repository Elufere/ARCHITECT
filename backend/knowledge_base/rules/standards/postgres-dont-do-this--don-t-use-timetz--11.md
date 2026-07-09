# Don't use timetz

> Source: [postgres-dont-do-this](https://wiki.postgresql.org/wiki/Don't_Do_This)

## Don't use timetz

Don't use thetimetztype. You probably wanttimestamptzinstead.

### Why not?

Even the manual tells you it's only implemented for SQL compliance.

> The type time with time zone is defined by the SQL standard, but the definition exhibits properties which lead to questionable usefulness. In most cases, a combination of date, time, timestamp without time zone, and timestamp with time zone should provide a complete range of date/time functionality required by any application.

### When should you?

Never.