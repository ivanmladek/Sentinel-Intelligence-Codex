Docs
Pricing
Resources
Researchers
Get Access
Accelerated batch inference for
high-leverage teams
Securely transform, structure, and generate datasets with LLMs in minutes instead of days. Up to 20x faster, 10x cheaper, and zero infrastructure setup.
Apply for Access
Get $50 in free credits when you get started

From Idea to Millions of Requests, Simplified
Sutro takes the pain away from testing and scaling LLM batch jobs to unblock your most ambitious AI projects.
import sutro as so
from pydantic import BaseModel
class ReviewClassifier(BaseModel):
    sentiment: str
user_reviews = './User_reviews-2.csv'
User_reviews.csv
User_reviews-1.csv
User_reviews-2.csv
User_reviews-3.csv
system_prompt = 'Classify the review as positive, neutral, or negative.'
results = so.infer(user_reviews, system_prompt, output_schema=ReviewClassifier)
Progress: 100% | 514,879/514,879 | Input tokens processed: 48.98m, Tokens generated: 5.12m
██████████████████████████████████████████
Processed in 47 minutes.
✓ Job results received. You can re-obtain the results with 'so.get_job_results('job-14adcc44-acde-45de-afb3-6775c60ff344')'
Rapidly Prototype
Shorten development cycles by getting feedback from large batch jobs in as little as minutes before scaling up.
Reduce Costs
Get results faster and reduce costs by 10x or more by parallelizing your LLM calls through Sutro.
Scale Effortlessly
Confidently handle millions of requests, and billions of tokens at a time without the pain of managing infrastructure.
A Simple Workflow For Batch Jobs

Prototype
Test prompts and models on a small sample. Get feedback in minutes.
Scale
Scale your LLM workflows so your team can do more in less time. Process billions of tokens in hours, not days, with no infrastructure headaches or exploding costs.
Progress: 100% | 2.5M/2.5M Rows
██████████████████████████████████████████
Data Orchestrators






Object Storage and Open Data Formats



Notebooks and Pythonic Coding Tools



Integrate
Seamlessly connect Sutro to your existing LLM workflows. Sutro's Python SDK is compatible with popular data orchestration tools, like Airflow and Dagster.
Purpose-Built Tools for Scalable LLM Workflows
Ship faster results without complex infrastructure to scale up any LLM workflow.
Synthesize
Generate high-quality, diverse, and representative synthetic data to improve model or RAG retrieval performance, without the complexity.
Classify
Automatically organize your data into meaningful categories without involving your ML engineer.
Evaluate
Benchmark your LLM outputs to continuously improve workflows, agents and assistants, or easily evaluate custom models against a new use-case.
Extract
Transform unstructured data into structured insights that drive business decisions.
Embed
Easily convert large corpuses of free-form text into vector representations for semantic search and recommendations.
Label
Enrich your data with meaningful labels to improve model training and data preparation.
Common Use Cases
View All Use Cases →
Unlock Product Insights
Easily sift through thousands of product reviews and unlock valuable product insights while brewing your morning coffee.
Unstructured ETL
Convert your massive amounts of free-form text into analytics-ready datasets without the pains of managing your own infrastructure.
Personalize Content
Tailor your marketing and advertising efforts to thousands, or millions of individuals, personas, and demographics to dramatically increase response rates and ad conversions.
Enrich Data
Improve your messy product catalog data, enrich your CRM entries, or gather insights from your historical meeting notes without involving your machine learning engineer.
Structure Web Pages
Crawl millions of web pages, and extract analytics-ready datasets for your company or your customers. Run standalone or successive batch jobs to explore complex link tree structures.
Improve Model Performance
Improve your LLM or RAG retrieval performance with synthetic data. Generate diverse and representative responses to fill statistical gaps.
Synthetic Data Generation
Create high-quality instruction-tuning datasets at scale.
Scale RL Rollouts
Run high-speed, large-scale model rollouts to continuously improve task-specific model performance.
 Large-Scale Model Evals
Rigorously test model performance across millions of data points.
Agentic Simulations
Simulate thousands of interacting agents to test emergent behaviors.
Population and Market Modeling
Run social simulations against massive populations of synthetic respondents and economic agents.
Scientific Modeling
Run large-scale simulations for genomics, climate science, and more.
Unlock Product Insights
Easily sift through thousands of product reviews and unlock valuable product insights while brewing your morning coffee.
Unstructured ETL
Convert your massive amounts of free-form text into analytics-ready datasets without the pains of managing your own infrastructure.
Personalize Content
Tailor your marketing and advertising efforts to thousands, or millions of individuals, personas, and demographics to dramatically increase response rates and ad conversions.
Enrich Data
Improve your messy product catalog data, enrich your CRM entries, or gather insights from your historical meeting notes without involving your machine learning engineer.
Structure Web Pages
Crawl millions of web pages, and extract analytics-ready datasets for your company or your customers. Run standalone or successive batch jobs to explore complex link tree structures.
Improve Model Performance
Improve your LLM or RAG retrieval performance with synthetic data. Generate diverse and representative responses to fill statistical gaps.
Synthetic Data Generation
Create high-quality instruction-tuning datasets at scale.
Scale RL Rollouts
Run high-speed, large-scale model rollouts to continuously improve task-specific model performance.
 Large-Scale Model Evals
Rigorously test model performance across millions of data points.
Agentic Simulations
Simulate thousands of interacting agents to test emergent behaviors.
Population and Market Modeling
Run social simulations against massive populations of synthetic respondents and economic agents.
Scientific Modeling
Run large-scale simulations for genomics, climate science, and more.
Unlock Product Insights
Easily sift through thousands of product reviews and unlock valuable product insights while brewing your morning coffee.
Unstructured ETL
Convert your massive amounts of free-form text into analytics-ready datasets without the pains of managing your own infrastructure.
Personalize Content
Tailor your marketing and advertising efforts to thousands, or millions of individuals, personas, and demographics to dramatically increase response rates and ad conversions.
Enrich Data
Improve your messy product catalog data, enrich your CRM entries, or gather insights from your historical meeting notes without involving your machine learning engineer.
Structure Web Pages
Crawl millions of web pages, and extract analytics-ready datasets for your company or your customers. Run standalone or successive batch jobs to explore complex link tree structures.
Improve Model Performance
Improve your LLM or RAG retrieval performance with synthetic data. Generate diverse and representative responses to fill statistical gaps.
Synthetic Data Generation
Create high-quality instruction-tuning datasets at scale.
Scale RL Rollouts
Run high-speed, large-scale model rollouts to continuously improve task-specific model performance.
 Large-Scale Model Evals
Rigorously test model performance across millions of data points.
Agentic Simulations
Simulate thousands of interacting agents to test emergent behaviors.
Population and Market Modeling
Run social simulations against massive populations of synthetic respondents and economic agents.
Scientific Modeling
Run large-scale simulations for genomics, climate science, and more.
FAQ
What Will You Scale with Sutro?
Get Access
team@sutro.sh
Classification
Data Scraping
Synthetic Data Generation
Bulk Content Generation
View All
Blog
Privacy Policy
Documentation
Pricing
Terms of Service
Jobs


Member of Technical Staff - Infrastructure & LLMs
Role
We are seeking an infrastructure and LLM-focused engineer to join a small, mighty team working at the forefront of large-scale inference technologies. You’ll be directly responsible for designing, owning, and improving systems designed to expand the scope of what is possible with data.
Examples of projects you might work on:
Scaling a secure and fault-tolerant inference system to handle many billions, or trillions of tokens in a single workload.
Building a high-performance data caching and lookup system.
Researching novel distillation techniques around a heterogenous set of use-cases.
Writing performance profilers, job schedulers, cost attribution models, and other scientifically-oriented code.
You’ll be a good fit if:
You’re very technically curious, interested or experienced in distributed systems and open-source LLM technologies and have demonstrated experience in such areas (professional, academic, or personal).
You learn fast and take ownership. 
You think from first principles, and question default “expert” opinions. 
You want to ship important, well-crafted products, and have a bias towards building.
You are a kind, honest, and positive-sum person ready to have fun and learn a ton along the way.
About 
Skysight is an AI-infrastructure company based in San Francisco, CA. We’re quietly working on tooling for large-scale inference use-cases - a technology we think is profoundly important. Our mission is to build tools that increase human leverage. 
We have a great set of investors, angels, and advisors - as well as significant early demand and traction.
Values and Culture
It’s a strange time in history when it comes to building startups. Between the confusing mega-rounds in venture capital, AGI threatening the future of work, and the mercenary effect created by post-pandemic work culture, it’s tough for many great candidates to want to stomach working at an early-stage startup. 
As much as we’re interested in building important, groundbreaking new technology - we’re just as interested in creating an incredible environment to work in: one that we can see ourselves in 2, 5, and 10 years from now. We take our values seriously; you can expect part of the interview process to center around our existing values and how you’ll contribute to upholding and growing them.
Other
Compensation: $170,000-$220,000 base salary, high target equity, other benefits to be discussed.
Location: SF Bay Area or willing to move. Exceptions may be made in rare cases.
We will not be able to accommodate visa sponsorship at this time.
Applying
If you’re interested in applying, send us an email at jobs@skysight.inc. Please include a resume or LinkedIn profile, as well as a brief description of your interest and any other questions you may have.


Sutro
Member of Technical Sta 
LLMs
Full-time Hybrid San Francisco, CA $170k - $220k
- Infrastructure &
About this role
We’re looking for a M
ember of Technical Staff - Infrastructure &
LL
Ms with deep
curiosity and strong technical instincts to join us at the earliest stages of building an AI-
native inference platform. We’re rethinking how modern analytical workloads—such as
structured extraction, classification, sentiment analysis, and multimodal question
answering—run at scale, designing systems that spin up 100s of GPUs to process data
efficiently, securely, and cost-effectively. This role is ideal for someone who thrives on
hard technical problems and wants to work at the intersection of infrastructure and
large language models. You’ll join a two-person full-time team with a few fractional
contributors, working closely with the founder to build and own foundational systems
from the ground up.
You’ll have real ownership over performance-critical code and systems, from
distributed job schedulers to secure multi-tenant inference infrastructure. Example
projects might include scaling fault-tolerant inference workloads to trillions of tokens,
building ultra-fast data caching and lookup pipelines, writing performance profilers and
cost attribution models, or experimenting with novel
M
LL
distillation techniques. We're
not looking for a particular pedigree—we care most about your ability to learn fast,
build well, and go deep. This is a rare opportunity to shape a foundational AI platform
early, contribute meaningfully to the technical roadmap, and grow into a key leader or
co-founder. The team is based in San Francisco and prefers in-person collaboration 2–4
days per week.
2+ years of experience
as a software engineer across the full stack, with strong experience and
knowledge in infrastructure
Salary
$170k - $220k
Equity
High target equity 1% - 3%
Visa sponsorship not available
Visas will not be offered.
Hybrid work policy
San Francisco, CA
Full-time position
Location
San Francisco, CA
Report to
https:/ /www.linkedin.com/in/seth-kimmel-5574a5115/
T ech stack
Python – Core programming language for the product (SDK and backend),
Distributed Systems – Custom infra for batch inference across 100+ GPUs, CUDA
/ GPU orchestration – Running large-scale
LL
M inference workloads efficiently,
Docker / Containerization – Used for infra deployment and scaling, Kubernetes or
equivalent – Relevant for managing distributed workloads at scale
About Sutro
We build large-scale AI inference infrastructure to increase human leverage, grow
productivity, and enable discovery.
T eam size
2 people
Website
sutro.sh
Company locations
San Francisco, CA
Founded
2023
LinkedIn
Visit
About the team
Based in SF Bay Area; strong preference for local hires
Typically 2–4 days in person; ~8–12 hours/day of work with some nights/weekends,
but no face time culture
“Use all your gas, but don’t burn out”—values creative, thoughtful work over brute
force hours
Wants self-managing team members—Seth doesn’t want to micromanage
Open to generous equity and founder-level responsibility over time
Tech stack
Python – Core programming language for the product (SDK and backend), Distributed
Systems – Custom infra for batch inference across 100+ GPUs, CUDA / GPU
orchestration – Running large-scale
LL
M inference workloads efficiently, Docker /
Containerization – Used for infra deployment and scaling, Kubernetes or equivalent –
Relevant for managing distributed workloads at scale
Interview process
1 Initial Screen
2 Take-home Project
3 Review session
4 Cultural/values alignment
Paraform
Terms
Privacy