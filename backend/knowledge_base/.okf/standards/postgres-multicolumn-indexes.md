---
source_url: "https://www.postgresql.org/docs/current/indexes-multicolumn.html"
category: "standards"
title: "PostgreSQL Multicolumn Indexes"
slug: "postgres-multicolumn-indexes"
date_scraped: "2026-07-09 13:33:58 UTC"
description: "Composite index behavior, leading-column rules, and column order selection"
---
## 11.3. Multicolumn Indexes#

An index can be defined on more than one column of a table. For example, if you have a table of this form:

```

CREATE TABLE test2 (
  major int,
  minor int,
  name varchar
);
```

(say, you keep your/devdirectory in a database...) and you frequently issue queries like:

```
constant
```

then it might be appropriate to define an index on the columnsmajorandminortogether, e.g.:

```

CREATE INDEX test2_mm_idx ON test2 (major, minor);
```

Currently, only the B-tree, GiST, GIN, and BRIN index types support multiple-key-column indexes. Whether there can be multiple key columns is independent of whetherINCLUDEcolumns can be added to the index. Indexes can have up to 32 columns, includingINCLUDEcolumns. (This limit can be altered when buildingPostgreSQL; see the filepg_config_manual.h.)

A multicolumn B-tree index can be used with query conditions that involve any subset of the index's columns, but the index is most efficient when there are constraints on the leading (leftmost) columns. The exact rule is that equality constraints on leading columns, plus any inequality constraints on the first column that does not have an equality constraint, will always be used to limit the portion of the index that is scanned. Constraints on columns to the right of these columns are checked in the index, so they'll always save visits to the table proper, but they do not necessarily reduce the portion of the index that has to be scanned. If a B-tree index scan can apply the skip scan optimization effectively, it will apply every column constraint when navigating through the index via repeated index searches. This can reduce the portion of the index that has to be read, even though one or more columns (prior to the least significant index column from the query predicate) lacks a conventional equality constraint. Skip scan works by generating a dynamic equality constraint internally, that matches every possible value in an index column (though only given a column that lacks an equality constraint that comes from the query predicate, and only when the generated constraint can be used in conjunction with a later column constraint from the query predicate).

For example, given an index on(x, y), and a query conditionWHERE y = 7700, a B-tree index scan might be able to apply the skip scan optimization. This generally happens when the query planner expects that repeatedWHERE x = N AND y = 7700searches for every possible value ofN(or for everyxvalue that is actually stored in the index) is the fastest possible approach, given the available indexes on the table. This approach is generally only taken when there are so few distinctxvalues that the planner expects the scan to skip over most of the index (because most of its leaf pages cannot possibly contain relevant tuples). If there are many distinctxvalues, then the entire index will have to be scanned, so in most cases the planner will prefer a sequential table scan over using the index.

The skip scan optimization can also be applied selectively, during B-tree scans that have at least some useful constraints from the query predicate. For example, given an index on(a, b, c)and a query conditionWHERE a = 5 AND b >= 42 AND c < 77, the index might have to be scanned from the first entry witha= 5 andb= 42 up through the last entry witha= 5. Index entries withc>= 77 will never need to be filtered at the table level, but it may or may not be profitable to skip over them within the index. When skipping takes place, the scan starts a new index search to reposition itself from the end of the currenta= 5 andb= N grouping (i.e. from the position in the index where the first tuplea = 5 AND b = N AND c >= 77appears), to the start of the next such grouping (i.e. the position in the index where the first tuplea = 5 AND b = N + 1appears).

A multicolumn GiST index can be used with query conditions that involve any subset of the index's columns. Conditions on additional columns restrict the entries returned by the index, but the condition on the first column is the most important one for determining how much of the index needs to be scanned. A GiST index will be relatively ineffective if its first column has only a few distinct values, even if there are many distinct values in additional columns.

A multicolumn GIN index can be used with query conditions that involve any subset of the index's columns. Unlike B-tree or GiST, index search effectiveness is the same regardless of which index column(s) the query conditions use.

A multicolumn BRIN index can be used with query conditions that involve any subset of the index's columns. Like GIN and unlike B-tree or GiST, index search effectiveness is the same regardless of which index column(s) the query conditions use. The only reason to have multiple BRIN indexes instead of one multicolumn BRIN index on a single table is to have a differentpages_per_rangestorage parameter.

Of course, each column must be used with operators appropriate to the index type; clauses that involve other operators will not be considered.

Multicolumn indexes should be used sparingly. In most situations, an index on a single column is sufficient and saves space and time. Indexes with more than three columns are unlikely to be helpful unless the usage of the table is extremely stylized. See alsoSection 11.5andSection 11.9for some discussion of the merits of different index configurations.

---

| 11.2. Index Types | Home | 11.4. Indexes andORDER BY |