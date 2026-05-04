# RecruitMind AI

An AI-powered recruitment intelligence chat application that enables HR teams to search candidates, analyze talent pool data, compare candidates, and generate interview questions using natural language.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://recruitmind-ai.streamlit.app/)

## Live Demo

[https://recruitmind-ai.streamlit.app/](https://recruitmind-ai.streamlit.app/)

## Overview

RecruitMind AI is built on a LangGraph conditional routing architecture with 9 execution nodes and a router that automatically classifies 8 user intents, routing each query to the appropriate node without explicit user instruction. The system combines semantic vector search, natural language to SQL analytics, LLM-based candidate evaluation, and interview question generation in a single chat interface.

## Features

- **Semantic Candidate Search** -- Retrieves candidates based on competency relevance using vector similarity, not keyword matching. Supports both short queries and full Job Descriptions as input.
- **Talent Analytics** -- Answers natural language questions about talent pool composition, distributions, and statistics by generating and executing SQL queries against a structured database.
- **Candidate Comparison** -- Compares 2 to 3 candidates side by side with a structured table and explicit recommendation reasoning.
- **Interview Question Generator** -- Generates 7 candidate-specific interview questions based on individual profile data. No generic questions.
- **Automatic Intent Routing** -- Classifies 8 intents automatically: RAG, SQL, EVALUATOR, GENERATOR, CLARIFY, CHITCHAT, CONVERSATION, GENERAL.
- **Conversational AI** -- Answers follow-up questions from conversation history and general recruitment and HR knowledge questions.
- **Cross-Session Memory** -- Persists up to 5 sessions with LLM-generated titles and full conversation history.
- **Guardrails** -- Blocks prompt injection, social engineering, out-of-scope requests, and sensitive data access attempts.

## Tech Stack

| Component | Technology |
|---|---|
| Agent Framework | LangGraph 1.1.10 |
| LLM | OpenAI gpt-4o-mini |
| Embeddings | OpenAI text-embedding-3-small (1536 dimensions) |
| Vector Database | Qdrant Cloud |
| Relational Database | SQLite |
| Reranker | FlashRank (rank-T5-flan) |
| UI | Streamlit 1.57.0 |
| Monitoring | LangFuse |
| Deployment | Streamlit Community Cloud |

## Architecture

The system uses LangGraph Conditional Routing with a shared state graph. All nodes read from and write to a single AgentState. Routing is fully deterministic after the Router Node classifies intent.

**9 Nodes:** input, router, rag, sql, evaluator, generator, clarification, task_dispatcher, response

**8 Intents classified by router:** RAG, SQL, EVALUATOR, GENERATOR, CLARIFY, CHITCHAT, CONVERSATION, GENERAL

Multiple intents can route to the same node. CHITCHAT, CONVERSATION, GENERAL, and CLARIFY all route to the clarification node. RAG and MULTI_STEP both start at the rag node.

```
Input -> Router -> rag / sql / evaluator / generator / clarification
                -> task_dispatcher (multi-step queue)
                -> response -> END
```

Multi-step queries are handled via a task_queue in state, allowing sequential execution of multiple capabilities in a single user message.

## Dataset

- 2,483 resumes across 24 job categories
- Processed through a 4-stage ETL pipeline: preprocessing, LLM extraction, SQL ingestion, vector ingestion
- Resume summaries generated via gpt-4o-mini structured extraction
- Embedded using text-embedding-3-small and stored in Qdrant Cloud

## Project Structure

```
recruitmind-ai/
├── agent/
│   ├── nodes/          # LangGraph execution nodes
│   ├── prompts/        # System prompts per node
│   ├── graph.py        # Graph definition and compilation
│   └── state.py        # AgentState TypedDict
├── app/
│   ├── components/     # Streamlit UI components
│   ├── logger.py       # Centralized logging setup
│   └── main.py         # Streamlit entry point
├── pipeline/
│   ├── 01_preprocess.py    # HTML stripping and normalization
│   ├── 02_extract.py       # LLM structured extraction
│   ├── 03_ingest_sql.py    # SQLite ingestion
│   └── 04_ingest_vectordb.py  # Qdrant ingestion
├── services/
│   ├── embedding_service.py    # OpenAI embedding wrapper
│   ├── qdrant_service.py       # Qdrant search wrapper
│   ├── reranker_service.py     # FlashRank reranker wrapper
│   └── sqlite_service.py       # SQLite query wrapper
├── data/
│   └── processed/      # CSV files committed to repo
├── config.py           # System-wide constants
└── requirements.txt    # Pinned dependencies
```

## Setup

### Prerequisites

- Python 3.12
- OpenAI API key
- Qdrant Cloud account
- LangFuse account (optional, for monitoring)

### Installation

1. Clone the repository

```bash
git clone https://github.com/daffaalhanif/recruitmind-ai.git
cd recruitmind-ai
```

2. Create and activate virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Configure secrets

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Edit `.streamlit/secrets.toml` with your actual credentials:

```toml
OPENAI_API_KEY = "sk-..."
QDRANT_URL = "https://your-cluster-url.qdrant.io"
QDRANT_API_KEY = "your-qdrant-api-key"
LANGFUSE_PUBLIC_KEY = "pk-lf-..."
LANGFUSE_SECRET_KEY = "sk-lf-..."
LANGFUSE_HOST = "https://cloud.langfuse.com"
```

5. Run the ETL pipeline (first time only)

```bash
python pipeline/01_preprocess.py
python pipeline/02_extract.py
python pipeline/03_ingest_sql.py
python pipeline/04_ingest_vectordb.py
```

6. Run the application

```bash
streamlit run app/main.py
```

## Deployment

The application is deployed on Streamlit Community Cloud. SQLite is ephemeral on Streamlit Community Cloud and will be regenerated automatically from the committed CSV files on each redeploy. Qdrant Cloud data persists across redeployments.

## Known Limitations

- **Dataset is static** -- New resumes cannot be added at runtime. Adding new data requires re-running the ETL pipeline.
- **Skills not normalized** -- Skill name variations such as "Project Management" and "project management" are counted as separate entries, affecting analytics accuracy.
- **No candidate contact information** -- The dataset uses anonymized IDs. The system can find and evaluate candidates but cannot provide contact details.
- **No location data** -- All location fields in the dataset have been anonymized. Location-based filtering is not supported.
- **Active candidates overwrite** -- Each new search replaces the current active candidates list. Cross-search candidate comparison is not supported.
- **Relevance labels are relative** -- Labels (Highly Relevant, Relevant, etc.) are relative among the candidates returned, not absolute measures of fit.
- **No real-time data access** -- The system has no internet access and cannot provide real-time data such as current market salaries or industry trends.
- **Chat history lost on redeploy** -- Session history stored in SQLite is lost when the app is redeployed on Streamlit Community Cloud.

## Author

Muhammad Daffa Al Hanif