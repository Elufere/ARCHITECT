---
source_url: "https://www.ibm.com/think/topics/database-normalization"
category: "standards"
title: "IBM Database Normalization — 1NF through 4NF"
slug: "ibm-database-normalization"
date_scraped: "2026-07-09 13:34:12 UTC"
description: "Formal normalization forms with functional dependency definitions and worked examples"
---
# What is database normalization?

Authors

[Alice Gomstyn](https://www.ibm.com/think/author/alice-gomstyn.html)

Staff Writer

IBM Think

[Alexandra Jonker](https://www.ibm.com/think/author/alexandra-jonkeribm-com)

Staff Editor

IBM Think

## What is database normalization?

#### Database normalization is adatabasedesign process that organizes data into specific table structures. It helps to improvedata integrity, prevent data anomalies, minimizedata redundancyand bolster query performance.

Normalization optimizes tables in database management systems (DBMS) to meet what are known as normal forms: sets of rules governing how attributes are organized within a table. These rules are based largely on relationships between attributes (columns), including keys used for uniquely identifying rows.

## Why is database normalization important?

At its core, database normalization—sometimes called data normalization—helps businesses and institutions more effectively organize, query and maintain large volumes of complex, interrelated and dynamicdata. Though enterprises now generate and store data at an unprecedented scale, the need for database normalization isn’t new. It predatescloud storageand even the invention ofdata warehouses.

Since the 1960s, corporations have struggled to manage large datasets. In the 1970s,Edgar F. Codd, the IBM mathematician known for his landmark paperintroducing relational databases, proposed that database normalization couldavoid “undesirable” dependenciesbetween attributes (columns) and the problems they can create.

In other words, when data records are related to each other in adatabasestructure, changes to single values or rows in a large, complicated table might yield unintended consequences—such as data inconsistency and data loss. Database normalization is designed to minimize such risks.

## What are the benefits of database normalization?

The benefits of database normalization include:

Prevention of data anomalies

When larger, more complicated tables are decomposed (or divided) into smaller, simpler tables, altering a database becomes an easier, less error-prone process, and limits changes to the now-smaller tables of related data.

Reduction of unintentional data redundancy

While intentionaldata redundancycan help improvedata quality,securityand availability, uninentional data redundancy is the effect of systems inadvertently creating duplicate data, which results in inefficiencies.

Data storage cost savings

Reducing duplicate data through database normalization can lowerdata storagecosts. This is especially important forcloudenvironments where pricing is often based on the volume of data storage used.

Faster data retrieval

Less data redundancy due to normalization can also lead to faster data queries as lower redundancy often requires lessdata processingduring searches.

## What data anomalies does database normalization address?

The normalization of data structures can prevent three key types of anomalies:

Insertion anomalies:An insertion anomaly occurs when a data record cannot be inserted into a table because it is missing values required by one or more columns in the table.

Deletion anomalies:A deletion anomaly occurs when the deletion of a record results in the unintentional deletion of important data included in that record.

Update anomalies:An update anomaly occurs when an instance of data is updated in one location in a database but not in other locations where that data value is also stored, resulting in a lack of data consistency.

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

## The significance of dependencies in database normalization

Several database normalization constraints are based on the relationships (also known as dependencies) between primary keys and columns that are neither primary nor candidate keys. The latter are known as non-key attributes or non-prime attributes.

Relationships between attributes in databases where one attribute (the determinant) determines the value of another attribute are known as functional dependencies. Types of functional dependencies between attributes include partial dependency, transitive dependency, multi-valued dependency and join dependency. These relationships are best understood when discussed in the context of relevant sets of normalization rules, or normal forms.

## What are the normal forms of database normalization?

Executing normalization in data models involves designing tables to conform with one or more levels of normalization, also known as normal forms. Common forms include:

- First normal form
- Second normal form
- Third normal form and Boyce-Codd Normal Form
- Fourth normal form
- Fifth normal form

### First normal form

First normal form, the most basic database normalization criteria, requires that a database table schema includes a primary key while excluding repetition among columns. To be more specific, a table in first normal form should not have fields with arrays of values—for instance, a single cell with three different names in it—nor should it include repeating groups, which are different columns that store the same type of data.

To better understand first normal form, let’s use the following set of columns as an example:1

| rec_num | lname | fname | bdate | anniv | email | child1 | child2 | child3 |

The columns comprise a table of a group of parents, including their names, birthdays, wedding anniversaries, emails and children’s names.

This table violates first normal form because it contains three separate columns storing the same type of information: children’s names. In this case in particular, the table structure could open the door to insertion errors. For example, in the real world, many parents have fewer than three children.

In our example table, it’s not possible to add such parents’ records to the table. In addition, querying this table for a child’s name would be inefficient, requiring searching data in three different columns in every row.

Achieving first normal form for the data in the table requires separating the original table into two. One table would include most of the attributes of the original table, while the other would focus on children.

TABLE 1

| rec_num | lname | fname | bdate | anniv | email |

TABLE 2

rec_num         child_name

In this example, the new tables remain linked through the “rec_num” column, which is the primary key in Table 1 and is referenced by Table 2’s “rec_num” column, which serves as a foreign key.

While satisfying first normal form might not reduce redundant data (“rec_num” values will appear in multiple rows of Table 2 when parents have more than one child) the elimination of repeating groups can make queries simpler.

### Second normal form

In second normal form, no non-key attribute has a partial dependency on the primary key in the table. In other words, if a primary key is a composite key, the non-key attribute should depend on every column in that composite key.

For example, consider an inventory table that has records of quantities of specific parts that are stored at particular warehouses. The following figure shows the attributes of the inventory entity.2

| part | warehouse | quantity | warehouse_address |

In this example, the “part” and “warehouse” columns form a composite primary key. However, the attribute “warehouse_address” depends only on the value of “warehouse,” so the table violates second normal form.

This table is also prone to data redundancy, with the value for warehouse_address listed each time a record for a part from the same warehouse appears in the table. This raises the risk of update errors should the address be updated in one row and not in others. A deletion error may also occur if any one warehouse stops storing parts—should records of those parts be deleted, the warehouse address would be deleted as well.

To satisfy second normal form and reduce the likelihood of errors, the data can be distributed between two new tables:

TABLE 1

| part | warehouse | quantity |

TABLE 2

warehouse                                warehouse_address

### Third normal form and Boyce-Codd Normal Form

A table in third normal form satisfies both first and second normal forms while also avoiding situations where non-key attributes depend on other non-key attributes instead of primary keys. When non-key attributes do depend on other non-key attributes, this is known as a transitive dependency—a violation of third normal form.

Consider the following table of employee information:3

| emp_num | emp_fname | emp_lname | dept_num | dept_name |
| 0200 | David | Brown | D11 | Manufacturing System |
| 0320 | Ramlal | Mehta | E21 | Software Support |
| 0220 | Jennifer | Lutz | D11 | Manufacturing System |

In this table, the primary key is the “emp_num” column. However, the “dept_name” column depends on the “dept_num” column, a non-key attribute. Therefore, the table does not meet third normal form and raises the risk of errors such as update anomalies—if a department name, such as “manufacturing system,” changed, it would have to be updated in more than one row under the current table schema.

Organizing the data into third normal form in a normalized database could prevent such errors. In this case, this process would entail structuring the data into three separate tables:EMPLOYEE, DEPARTMENT, and EMPLOYEE_DEPARTMENT4

EMPLOYEE Table

| emp_num | emp_fname | emp_lname |
| 0200 | David | Brown |
| 0320 | Ramlal | Mehta |
| 0220 | Jennifer | Lutz |

DEPARTMENT Table

| dept_num | dept_name |
| D11 | Manufacturing System |
| E21 | Software Support |

EMPLOYEE_DEPARTMENT Table

| dept_num | emp_num |
| D11 | 0200 |
| D11 | 0220 |
| E21 | 0320 |

Boyce-Codd Normal Form, or BCNF, is a normal form that is considered a stricter or stronger version of third normal form. BCNF requires the use of super keys.

### Fourth normal form

A table is in fourth normal form if it does not have multi-valued dependencies. Multi-valued dependencies occur when the values of two or more columns are independent of each other and only dependent on the primary key.

A commonly cited example in tutorials centers on employee tables listing both skills and languages. An employee can have several skills and speak multiple languages. Two relationships exist: one between employees and skills and one between employees and languages.

A table is not in fourth normal form if it represents both relationships. Converting the data into fourth normal form would require structuring it into two tables—one for employee skills and one for languages.

### Fifth normal form

Commonly considered the highest level of normalization, fifth normal form is a criterion centered on join dependency. In join dependency, after a table is divided into smaller tables, it is possible to reconstitute the original table by bringing the new tables back together again—all without losing any data or accidentally creating new rows of data. It is comparable to a completed jigsaw puzzle that, when broken apart, can be put back together into its original form.

In fifth normal form, a table should be divided into smaller tables only when join dependency is achievable. If, however, attempts to reconstitute the original table from smaller tables unintentionally leads to the creation of a slightly different table, then decomposition of the original table should not take place. Returning to our jigsaw puzzle analogy, it would be like putting a puzzle back together again, only to find a piece is missing or that an extra piece has materialized.

## What are the challenges of database normalization?

For all its benefits, database normalization comes with trade-offs. For instance, prior to normalization, a user seeking specific data might only have to query one table. However, if a database has more tables following a normalization, the user may find themselves having to query multiple tables—which can be a slower and more expensive process.

Additionally, even as normalization makes individual tables simpler, it can increase the complexity of the database overall, requiring significant expertise on the part of database designers and administrators to ensure proper implementation.

[Read the Data Leader's guide to learn how you can make your organization's data AI-ready.](https://www.ibm.com/forms/mkt-53325)

## Resources

[AI Agents run on data - is yours ready?Your data is your competitive edge. Learn how to unlock it securely and drive measurable ROI from AI in this short webinar.](https://ibm.webcasts.com/starthere.jsp?ei=1740011&tp_key=ada6dc37d2&sti=inbound)
[Data management explainedTechsplainers by IBM breaks down the essentials of data for AI, from key concepts to real‑world use cases. Clear, quick episodes help you learn the fundamentals fast.](https://www.ibm.com/think/podcasts/techsplainers#tabs-fw-44e285b2cc-item-16e8334e37-tab)
[Unify and access your data to help scale your AILearn why the path to AI-ready data often starts with effective access to both structured and unstructured data and the challenges that can impede data leaders.](https://www.ibm.com/forms/mkt-54006)
[Legal overhead turned into strategic insightLearn how an AI-powered legal agent helps accelerate decision-making, reduce manual work and improve compliance.](https://www.ibm.com/case-studies/dynamiq)
[AI Academy: Building a data strategy for enterprise AIIn this episode, Cathy Reese explains how organizations today need a data strategy that’s ready for advanced AI, which will require them to harness their highest quality data assets.](https://www.ibm.com/think/videos/ai-academy/building-data-strategy-enterprise-ai)
[The hybrid, open data lakehouse for AISimplify data access and automate data governance. Discover the power of integrating a data lakehouse strategy into your data architecture, including cost-optimizing your workloads and scaling AI and analytics, with all your data, anywhere.](https://www.ibm.com/forms/mkt-52131)
[Cost of a Data Breach Report 2025Data breach costs have hit a new high. Get up-to-date insights into cybersecurity threats and their financial impacts on organizations.](https://www.ibm.com/forms/mkt-53830)
[The data leader’s guide to AI-ready dataUnderstand the actionable steps data leaders can take to overcome data challenges, establish the groundwork for a trusted data foundation and help get your organization’s data ready for AI.](https://www.ibm.com/forms/mkt-53325)
[How the C-suite is turning information into impactExplore insights from 1,700 CDOs in this cross-industry report for data leaders.](https://www.ibm.com/account/reg/signup?formid=urx-54212)
Related solutions
IBM® watsonx.data™

Watsonx.data enables you to scale analytics and AI with all your data, wherever it resides, through an open, hybrid and governed data store.

[Discover watsonx.data](https://www.ibm.com/products/watsonx-data)
Data management software and solutions

Design a data strategy that eliminates data silos, reduces complexity and improves data quality for exceptional customer and employee experiences.

[Discover data management solutions](https://www.ibm.com/solutions/data-management)
Data and AI consulting services

Successfully scale AI with the right strategy, data, security and governance in place.

[Explore data and AI consulting services](https://www.ibm.com/consulting/data-ai)
Author markup
Take the next step

Unify all your data for AI and analytics with IBM® watsonx.data™. Put your data to work, wherever it resides, with the hybrid, open data lakehouse for AI and analytics.

1. Discover watsonx.data
2. Explore data management solutions

##### Footnotes

1“First normal form.” IBM Documentation, Informix Servers. 19 November 2024.

2, 3, 4“Normalization in database design.” IBM Documentation, Db2 for z/OS. 22 January 2025.

Added for Adobe analytics implementation ADCMS-5834
<script type="text/javascript">

    adobeDataLayer.push({
        "event": "linkClick",
        "web": {
            "webPageDetails": {
                "URL": document.URL,
                "name": "home"
            },
            "webInteraction": {
                "linkClick":"event",
                "value":"1",
                "type": "other",
                "URL": document.URL,
                "name": "linkClick: " +  document.URL
            }
        },
    })
</script>  !
Added for Adobe analytics implementation ADCMS-5834 & ADCMS-6152 !