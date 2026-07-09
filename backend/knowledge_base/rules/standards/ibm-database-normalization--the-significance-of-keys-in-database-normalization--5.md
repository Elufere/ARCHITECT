# The significance of keys in database normalization

> Source: [ibm-database-normalization](https://www.ibm.com/think/topics/database-normalization)

## The significance of keys in database normalization

Inrelational databases, a key is a column or an ordered collection of columns used to identify rows of data in a table. Keys in relational models also establish associations between related tables. These capabilities support successful, efficientSQL databasequeries. Keys that figure prominently in database normalization rules include:

- Primary keys
- Composite keys
- Candidate keys
- Foreign keys
- Super keys

### Primary keys

Aprimary keyis a column or columns in adatabasetable with values that serve as unique identifiers for each row or record. For example, a student ID column could be a primary key in a table of student information. Defining characteristics of primary keys are that they exclude null values, have no duplicate values and may consist of either single columns or multiple columns.

### Composite keys

Keys that consist of two or more columns are called composite keys. When primary keys are composite keys, they may be called composite primary keys.

### Candidate keys

A candidate key is a column or group of columns that has the characteristics of a primary key but has not been assigned primary key status.

### Foreign keys

A foreign key in one table refers to a specific primary key in another table in order to define a relationship between the tables. When larger tables are divided into smaller tables during normalization, foreign keys and primary keys establish an association between the new tables.

### Super keys

Super keys, while similar to composite primary keys, also consist of more columns than are necessary to uniquely identify records.