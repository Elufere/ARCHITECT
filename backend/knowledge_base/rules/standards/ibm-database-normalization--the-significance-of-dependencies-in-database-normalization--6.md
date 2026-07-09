# The significance of dependencies in database normalization

> Source: [ibm-database-normalization](https://www.ibm.com/think/topics/database-normalization)

## The significance of dependencies in database normalization

Several database normalization constraints are based on the relationships (also known as dependencies) between primary keys and columns that are neither primary nor candidate keys. The latter are known as non-key attributes or non-prime attributes.

Relationships between attributes in databases where one attribute (the determinant) determines the value of another attribute are known as functional dependencies. Types of functional dependencies between attributes include partial dependency, transitive dependency, multi-valued dependency and join dependency. These relationships are best understood when discussed in the context of relevant sets of normalization rules, or normal forms.