# Cloud Data and ML Architecture Summary

A study guide for a medium-difficulty technical interview, covering key services from AWS, GCP, and Azure.

## AWS (Amazon Web Services)

### Amazon S3 (Simple Storage Service)
- **What it is:** A highly scalable and durable object storage service. It is the foundational data store for many AWS services.
- **Use Case:** The primary location for a data lake. Stores raw data, processed datasets, model artifacts, application assets, and logs.
- **Interview Focus:** Durability/availability guarantees (99.999999999%), storage classes (Standard, Intelligent-Tiering, Glacier) for cost optimization, and its event-driven capabilities (e.g., S3 event triggers Lambda).

### AWS Lambda
- **What it is:** A serverless, event-driven compute service that lets you run code without provisioning or managing servers.
- **Use Case:** "Glue" code for data pipelines. Used for data validation, transformation, and enrichment tasks triggered by events (e.g., a new file in S3, a message in SQS). Also used for creating serverless API backends.
- **Interview Focus:** Cold starts, concurrency limits, execution time limits (max 15 mins), and its role in event-driven architectures.

### AWS Batch
- **What it is:** A service for running large-scale batch computing jobs. It dynamically provisions the optimal quantity and type of compute resources (e.g., EC2 instances) based on job requirements.
- **Use Case:** Large, long-running, non-interactive workloads like batch data processing, ETL jobs, or large-scale model training.
- **Interview Focus:** Differentiate it from Lambda (Batch is for long-running jobs, Lambda for short-lived functions). Understand the concept of Job Definitions, Job Queues, and Compute Environments.

### Amazon API Gateway
- **What it is:** A fully managed service for creating, publishing, maintaining, monitoring, and securing APIs at any scale.
- **Use Case:** Acts as the "front door" for applications to access data or business logic from your back-end services (like Lambda, ECS, or EC2).
- **Interview Focus:** RESTful vs. WebSocket APIs, integration with Lambda, handling authentication and authorization (IAM, Cognito), and traffic management (throttling, caching).

### Amazon SQS (Simple Queue Service)
- **What it is:** A fully managed message queuing service that enables you to decouple and scale microservices, distributed systems, and serverless applications.
- **Use Case:** Decoupling components in a data pipeline. A producer service sends messages (e.g., a file path to be processed) to a queue, and a consumer service (e.g., a Lambda function or ECS task) processes them at its own pace. This improves fault tolerance.
- **Interview Focus:** Standard vs. FIFO queues, visibility timeout, and the concept of a dead-letter queue (DLQ) for handling message failures.

### AWS IoT Core
- **What it is:** A managed cloud service that lets connected devices easily and securely interact with cloud applications and other devices.
- **Use Case:** Ingesting streaming data from a large fleet of IoT devices (sensors, cameras, etc.). It handles device authentication, authorization, and communication.
- **Interview Focus:** The roles of the Message Broker (using MQTT protocol), the Rules Engine (for routing data to other AWS services like S3 or SQS), and the Device Shadow.

### Amazon Athena
- **What it is:** An interactive, serverless query service that makes it easy to analyze data directly in Amazon S3 using standard SQL.
- **Use Case:** Ad-hoc analysis of data in your S3-based data lake without needing to load it into a database. Great for data exploration and business intelligence.
- **Interview Focus:** It's serverless (pay-per-query), schema-on-read, and works directly on S3 data. Mention its integration with AWS Glue Data Catalog for schema management.

### AWS Glue
- **What it is:** A fully managed extract, transform, and load (ETL) service.
- **Use Case:** Discovering, preparing, and combining data for analytics, machine learning, and application development. It can crawl data sources (like S3 or RDS), identify data formats, and suggest schemas (stored in the Glue Data Catalog) to create a central metadata repository. It also generates and runs ETL jobs (in a managed Spark environment).
- **Interview Focus:** The three main components: the Glue Data Catalog (central metadata repository), the crawler (populates the catalog), and the ETL job system.

### Amazon DynamoDB
- **What it is:** A fast, flexible, and highly scalable NoSQL key-value and document database.
- **Use Case:** Low-latency data storage and retrieval for applications that need consistent, single-digit millisecond performance at any scale. Often used for user profiles, session state, or lookup tables.
- **Interview Focus:** Single-table vs. multi-table design, provisioned throughput vs. on-demand capacity, and understanding of primary keys (partition key and sort key). Not suitable for ad-hoc analytical queries.

### Amazon Redshift
- **What it is:** A fully managed, petabyte-scale data warehouse service.
- **Use Case:** Running complex analytical queries against large datasets (business intelligence, reporting). It uses columnar storage and parallel processing to achieve high performance.
- **Interview Focus:** Columnar storage vs. traditional row-based storage (OLAP vs. OLTP). Understand its role as the central repository for structured, cleaned data ready for analytics.

### Amazon ECS (Elastic Container Service)
- **What it is:** A highly scalable, high-performance container orchestration service that supports Docker containers.
- **Use Case:** Running containerized applications or services, including data processing workers or ML model inference endpoints.
- **Interview Focus:** Differentiate it from EKS (ECS is AWS-proprietary, EKS is managed Kubernetes). Understand the concepts of Task Definitions, Tasks, Services, and Clusters.

### Amazon RDS (Relational Database Service)
- **What it is:** A managed service that makes it easy to set up, operate, and scale a relational database in the cloud.
- **Use Case:** Storing structured data for applications that require a traditional relational database (e.g., PostgreSQL, MySQL, Oracle). Manages patching, backups, and scaling.
- **Interview Focus:** It's a managed service, not a database itself. You choose the underlying engine. Understand the benefits (automated management) and trade-offs (less control than running a DB on EC2).

### Amazon EC2 (Elastic Compute Cloud)
- **What it is:** The most fundamental compute service, providing secure, resizable compute capacity (virtual servers) in the cloud.
- **Use Case:** The underlying compute for many other services. Can be used directly to host applications, run custom data processing jobs, or train ML models when you need maximum control over the environment.
- **Interview Focus:** The foundation of AWS. Understand instance types, pricing models (On-Demand, Spot, Reserved), and the trade-off between control (EC2) and managed services (Lambda, Batch).

### AWS Secrets Manager
- **What it is:** A secret management service that helps you protect access to your applications, services, and IT resources.
- **Use Case:** Storing and retrieving secrets like database credentials, API keys, and other tokens. Avoids hardcoding secrets in application code.
- **Interview Focus:** Integration with IAM for fine-grained access control and the ability to automatically rotate secrets.

### Amazon CloudWatch
- **What it is:** A monitoring and observability service.
- **Use Case:** Collecting and tracking metrics, collecting and monitoring log files, setting alarms, and automatically reacting to changes in your AWS resources. Essential for debugging and maintaining data/ML pipelines.
- **Interview Focus:** Key components: Metrics, Logs, and Alarms. Understand how it's used to monitor the health and performance of a pipeline.

## GCP (Google Cloud Platform)

### Google BigQuery
- **What it is:** A fully managed, serverless, and highly scalable enterprise data warehouse.
- **Use Case:** The GCP equivalent of Redshift/Athena combined. Used for large-scale data analytics and interactive SQL queries on massive datasets.
- **Interview Focus:** Serverless architecture (no infrastructure to manage), separation of compute and storage, and its speed on large analytical queries.

### Google Cloud Functions
- **What it is:** A scalable pay-as-you-go serverless compute platform that runs your code in response to events.
- **Use Case:** The GCP equivalent of AWS Lambda. Used for event-driven data processing, creating lightweight APIs, and gluing services together.
- **Interview Focus:** Similar concepts to Lambda: event-driven, stateless, and short-lived.

### Google Cloud DataProc
- **What it is:** A fully managed and highly scalable service for running Apache Spark and Apache Hadoop clusters.
- **Use Case:** The GCP equivalent of AWS EMR. Used for large-scale batch data processing, ETL, and machine learning on big data frameworks.
- **Interview Focus:** It's a managed service for the open-source big data ecosystem (Spark, Hadoop).

### Google Cloud Scheduler
- **What it is:** A fully managed enterprise-grade cron job scheduler.
- **Use Case:** Triggering batch data processing jobs, or any other application, on a recurring schedule. For example, running a DataProc job every night to process the previous day's data.
- **Interview Focus:** It's a reliable, serverless cron tool for scheduling automated tasks.

## Azure

### Azure Blob Storage
- **What it is:** Microsoft's object storage solution for the cloud.
- **Use Case:** The Azure equivalent of Amazon S3. Used for storing massive amounts of unstructured data, creating data lakes, and storing model artifacts.
- **Interview Focus:** Similar concepts to S3: high scalability, tiered storage for cost management (Hot, Cool, Archive), and event-driven capabilities.

### Azure Database for PostgreSQL
- **What it is:** A fully managed relational database service based on the open-source PostgreSQL database engine.
- **Use Case:** The Azure equivalent of Amazon RDS for PostgreSQL. Provides a managed, scalable, and highly available PostgreSQL database for applications requiring a relational data store.
- **Interview Focus:** It's a managed PaaS (Platform as a Service) offering, abstracting away the underlying infrastructure management.
