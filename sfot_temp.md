# **GOTO: FILE \-\> MAKE A COPY to save for your own use.**

# 

# **Section 1 \- Project Description** {#section-1---project-description}

## **1.1 Project** {#1.1-project}

The project name

## **1.2 Description** {#1.2-description}

Brief overall description of the project

## **1.3 Revision History** {#1.3-revision-history}

| Date | Comment | Author |
| :---- | :---- | :---- |
|  |  |  |
|  |  |  |
|  |  |  |
|  |  |  |
|  |  |  |

**Contents**  
[Section 1 \- Project Description](#section-1---project-description)  
[1.1 Project](#1.1-project)  
[1.2 Description](#1.2-description)  
[1.3 Revision History](#1.3-revision-history)  
[Section 2 \- Overview](#section-2---overview)  
[2.1 Purpose](#2.1-purpose)  
[2.2 Scope](#2.2-scope)  
[2.3 Requirements](#2.3-requirements)  
[2.3.1 Estimates](#2.3.1-estimates)  
[2.3.2 Traceability Matrix](#2.3.2-traceability-matrix)  
[Section 3 \- System Architecture](#section-3---system-architecture)  
[Section 4 \- Data Dictionary](#section-4---data-dictionary)  
[Section 5 \- Software Domain Design](#section-5---software-domain-design)  
[5.1 Software Application Domain Chart](#5.1-software-application-domain-chart)  
[5.2 Software Application Domain](#5.2-software-application-domain)  
[5.2.1 Domain X](#5.2.1-domain-x)  
[5.2.1.1 Component Y of Domain X](#5.2.1.1-component-y-of-domain-x)  
[5.2.1.1.1 Task Z of Component Y1 of Domain X](#5.2.1.1.1-task-z-of-component-y1-of-domain-x)  
[Section 6 – Data Design](#section-6-–-data-design)  
[6.1 Persistent/Static Data](#6.1-persistent/static-data)  
[6.1.1 Dataset](#6.1.1-dataset)  
[6.1.2 Static Data](#6.1.2-static-data)  
[6.1.3 Persisted data](#6.1.3-persisted-data)  
[6.2 Transient/Dynamic Data](#6.2-transient/dynamic-data)  
[6.3 External Interface Data](#6.3-external-interface-data)  
[6.4 Transformation of Data](#6.4-transformation-of-data)  
[Section 7 \- User Interface Design](#section-7---user-interface-design)  
[7.1 User Interface Design Overview](#7.1-user-interface-design-overview)  
[7.2 User Interface Navigation Flow](#7.2-user-interface-navigation-flow)  
[7.3 Use Cases / User Function Description](#7.3-use-cases-/-user-function-description)  
[Section 8 \- Other Interfaces](#section-8---other-interfaces)  
[8.1 Interface X](#8.1-interface-x)  
[Section 9 \- Extra Design Features / Outstanding Issues](#section-9---extra-design-features-/-outstanding-issues)  
[Section 10 – References](#section-10-–-references)  
[Section 11 – Glossary](#section-11-–-glossary)

# **Section 2 \- Overview** {#section-2---overview}

## **2.1 Purpose** {#2.1-purpose}

Brief description of the focus of this module of the overall project and its intended audience.

## **2.2 Scope** {#2.2-scope}

Describe the scope of the module to be produced

## **2.3 Requirements** {#2.3-requirements}

Your mileage may vary \-- we typically break down the requirements to provide a ballpark estimate.

### **2.3.1 Estimates** {#2.3.1-estimates}

| \# | Description | Hrs. Est. |
| ----- | ----- | ----- |
| 1 | Brief description of task / module with link | \# est |
|  | **TOTAL**: | \# est tot |

### **2.3.2 Traceability Matrix** {#2.3.2-traceability-matrix}

Cross reference this document with your requirements document and link where you satisfy each requirement

| SRS Requirement | SDD Module |
| :---- | :---- |
| Req 1 | 5.1.1 (link to module), 5.1.2 (link) |
|  |  |
|  |  |

# **Section 3 \- System Architecture** {#section-3---system-architecture}

Describe/include a figure of the overall system architecture (and where this module fits in)

# **Section 4 \- Data Dictionary** {#section-4---data-dictionary}

Brief description of each element in this module or a link to an actual data dictionary

(template of a database table description)

| Table |
| :---: |

| Field | Notes | Type |
| ----- | ----- | :---: |
| ID | Unique Identifier from TABLE\_SEQ | DECIMAL |
| NAME | The Name in Object.Name() | VARCHAR |
| VALUE | The Value output from somewhere | VARCHAR |

# **Section 5 \- Software Domain Design** {#section-5---software-domain-design}

## **5.1 Software Application Domain Chart** {#5.1-software-application-domain-chart}

Describe / chart each major software application domain and the relationships between objects (UML, etc)

## **5.2 Software Application Domain** {#5.2-software-application-domain}

A Comprehensive high level description of each domain (package/object wherever it is better to start) within the scope of this module (or within the greater scope of the project if applicable)

### **5.2.1 Domain X** {#5.2.1-domain-x}

A high level description of the family of components within this domain and their relationship. Include database domain, stored procedures, triggers, packages, objects, functions, etc.

#### ***5.2.1.1 Component Y of Domain X*** {#5.2.1.1-component-y-of-domain-x}

Define Component Y, describe data flow/control at component level

##### 5.2.1.1.1 Task Z of Component Y1 of Domain X {#5.2.1.1.1-task-z-of-component-y1-of-domain-x}

	Define Task Z, describe data flow/control at task level

# **Section 6 – Data Design** {#section-6-–-data-design}

Describe the data contained in databases and other shared structures between domains or within the scope of the overall project architecture

## **6.1 Persistent/Static Data** {#6.1-persistent/static-data}

Describe/illustrate the logical data model or entity relationship diagrams for the persistent data (or static data if static)

### **6.1.1 Dataset** {#6.1.1-dataset}

Describe persisted object/dataset and its relationships to other entities/datasets

### **6.1.2 Static Data** {#6.1.2-static-data}

Describe static data

### **6.1.3 Persisted data** {#6.1.3-persisted-data}

Describe persisted data

## **6.2 Transient/Dynamic Data** {#6.2-transient/dynamic-data}

Describe any transient data, include any necessary subsections

## **6.3 External Interface Data** {#6.3-external-interface-data}

Any external interfaces’ data goes here (this is for the data, section 8 is for the interface itself)

## **6.4 Transformation of Data** {#6.4-transformation-of-data}

Describe any data transformation that goes on between design elements

# **Section 7 \- User Interface Design** {#section-7---user-interface-design}

## **7.1 User Interface Design Overview** {#7.1-user-interface-design-overview}

Pictures, high level requirements, mockups, etc.

## **7.2 User Interface Navigation Flow** {#7.2-user-interface-navigation-flow}

Diagram the flow from one screen to the next

## **7.3 Use Cases / User Function Description** {#7.3-use-cases-/-user-function-description}

Describe screen usage / function using use cases, or on a per function basis

# **Section 8 \- Other Interfaces** {#section-8---other-interfaces}

Identify any external interfaces used in the execution of this module, include technology and other pertinent data

## **8.1 Interface X** {#8.1-interface-x}

Describe interactions, protocols, message formats, failure conditions, handshaking, etc

# **Section 9 \- Extra Design Features / Outstanding Issues** {#section-9---extra-design-features-/-outstanding-issues}

Does not fit anywhere else above, but should be mentioned \-- goes here

# **Section 10 – References**  {#section-10-–-references}

Any documents which would be useful to understand this design document or which were used in drawing up this design.

# **Section 11 – Glossary** {#section-11-–-glossary}

Glossary of terms / acronyms