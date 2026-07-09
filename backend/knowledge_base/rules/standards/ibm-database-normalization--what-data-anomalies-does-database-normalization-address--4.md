# What data anomalies does database normalization address?

> Source: [ibm-database-normalization](https://www.ibm.com/think/topics/database-normalization)

## What data anomalies does database normalization address?

The normalization of data structures can prevent three key types of anomalies:

Insertion anomalies:An insertion anomaly occurs when a data record cannot be inserted into a table because it is missing values required by one or more columns in the table.

Deletion anomalies:A deletion anomaly occurs when the deletion of a record results in the unintentional deletion of important data included in that record.

Update anomalies:An update anomaly occurs when an instance of data is updated in one location in a database but not in other locations where that data value is also stored, resulting in a lack of data consistency.