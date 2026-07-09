---
source_url: "https://use-the-index-luke.com/"
category: "standards"
title: "Use The Index, Luke — Indexing Deep Dive"
slug: "use-the-index-luke"
date_scraped: "2026-07-09 13:34:07 UTC"
description: "Markus Winand's vendor-agnostic guide to B-tree internals, index scans, and query optimization"
---
# SQL Indexing and Tuning e-Book

---

A site explaining SQL indexing to developers—no crap about administration.

SQL indexing is the most effective tuning method—yet it is often neglected during development.Use The Index, Lukeexplains SQL indexing from grounds up and doesn’t stop at ORM tools like Hibernate.

Use The Index, Lukeis the free web-edition ofmy bookSQL Performance Explained(from Є9.95). Be sure tosubscribe myfree newsletter.

## SQL Indexing in MySQL, Oracle, SQL Server, etc.

Use The Index, Lukepresents indexing in a vendor agnostic fashion. Product specific notes are provided like here:

Db2 (LUW)

Use The Index, Lukecovers SQL indexing for IBM Db2. Tests were conducted with Db2 for Linux, UNIX and Windows, (LUW) V10.5 through 12.1.

MySQL

Use The Index, Lukecovers SQL indexing for MySQL. Tests were conducted with MySQL 5.5 through 9.7.0.

Oracle

Use The Index, Lukecovers SQL indexing for the Oracle database. Tests were conducted with Oracle 11g through 26ai (23.26.2).

PostgreSQL

Use The Index, Lukecovers SQL indexing for PostgreSQL. Tests were conducted with PostgreSQL 9.0 through 17.

SQL Server

Use The Index, Lukecovers SQL indexing for Microsoft SQL Server. Tests were conducted with SQL Server 2008R2 through 2025.

Have more questions about SQL indexing or tuning? No problem—have a look at my training and tuning services atwinand.at.

## Table of Contents

1. Preface— Why is indexing a development task?
2. Anatomy of an Index— What does an index look like?The Leaf Nodes— A doubly linked listThe B-Tree— It’s a balanced treeSlow Indexes, Part I— Two ingredients make the index slow
3. The Where Clause— Indexing to improve search performanceThe Equals Operator— Exact key lookupPrimary Keys— Verifying index usageConcatenated Keys— Multi-column indexesSlow Indexes, Part II— The first ingredient, revisitedFunctions— Using functions in thewhereclauseCase-Insensitive Search—UPPERandLOWERUser-Defined Functions— Limitations of function-based indexesOver-Indexing— Avoid redundancyBind Variables— For security and performanceSearching for Ranges— Beyond equalityGreater, Less andBETWEEN— The column order revisitedIndexing SQLLIKEFilters—LIKEis not for full-text searchIndex Combine— Why not using one index for every column?Partial Indexes— Indexing selected rowsNULLin the Oracle Database— An important curiosityNULLin Indexes— Every index is a partial indexNOT NULLConstraints— affect index usageEmulating Partial Indexes— using function-based indexingObfuscated Conditions— Common anti-patternsDates— Pay special attention toDATEtypesNumeric Strings— Don’t mix typesCombining Columns— use redundantwhereclausesSmart Logic— The smartest way to make SQL slowMath— Databases don’t solve equations
4. Testing and Scalability— About hardwareData Volume— Sloppy indexing bites backSystem Load— Production load affects response timeResponse Time and Throughput— Horizontal scalability
5. The Join Operation— Not slow, if done rightNested Loops— About the N+1 selects problem in ORMHash Join— Requires an entirely different indexing approachSort-Merge Join‌— Like a zipper on two sorted sets
6. Clustering Data— To reduce IOIndex Filter Predicates Intentionally Used— to tuneLIKEIndex-Only Scan— Avoiding table accessIndex-Organized Table— Clustered indexes without tables
7. Sorting and Grouping— Pipelinedorder by: the third powerIndexed Order By—whereclause interactionsASC/DESCandNULL FIRST/LAST— changing index orderIndexed Group By— Pipelininggroup by
8. Partial Results— Paging efficientlySelecting Top-N Rows— if you need the first few rows onlyFetching The Next Page— The offset and seek methods comparedWindow-Functions— Pagination using analytic queries
9. Insert, Delete and Update— Indexing impacts on DML statementsInsert— cannot take direct benefit from indexesDelete— uses indexes for thewhereclauseUpdate— does not affect all indexes of the table

You can’t learn everything in one day. Subscribe the newsletter viaE-Mail,BlueskyorRSSto gradually catch up. Have a look atmodern-⁠sql.comas well.

## About the Author

Markus Winand provides insights into SQL and shows how different systems support it atmodern-sql.com. Previously he madeuse-the-index-luke.com, which is still actively maintained. Markus can be hired as trainer, speaker and consultant viawinand.at.

## Buy the Book

The essence of SQL tuning in 200 pages

Buy now!(paperback and/or PDF)

Paperback also available atAmazon.com.

## Hire Markus

Markus offers SQL training and consulting for developers working at companies of all sizes.Learn more »