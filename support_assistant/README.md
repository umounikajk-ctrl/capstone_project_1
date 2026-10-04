# Module 3 — Support Assistant

A small GenAI-powered customer support assistant for Zepto. The application uses document embeddings, ChromaDB retrieval, LangGraph orchestration, structured outputs, and FastAPI to answer customer support questions using a predefined policy document corpus.

The system also supports a deterministic **MOCK_LLM mode** so that the complete pipeline can be tested offline without requiring an external LLM API.

---

## Project Overview

The Support Assistant follows a Retrieval-Augmented Generation (RAG) architecture.

### Main capabilities

* Load and process Zepto support documents
* Generate embeddings using `all-MiniLM-L6-v2`
* Store and retrieve document chunks using ChromaDB
* Classify user questions into:

  * `policy_question`
  * `general_question`
* Orchestrate the workflow using LangGraph
* Retrieve relevant policy information
* Generate structured responses
* Return answer, sources, and confidence
* Expose the assistant through a FastAPI API
* Support deterministic offline testing using `MOCK_LLM=1`
* Support optional real LLM usage using `MOCK_LLM=0`
* Containerize the application using Docker
* Run the application locally in Docker
* Deploy the containerized Streamlit application to a Docker-based hosting environment such as Hugging Face Spaces

## Architecture

The application has two interfaces:

Streamlit UI — user-facing chat application

FastAPI API — programmatic/API access to the support assistant

The Streamlit application directly uses the RAG/LangGraph pipeline.

                         User
                           |
                           v
                 +-------------------+
                 |   Streamlit UI    |
                 |      app.py       |
                 +---------+---------+
                           |
                           v
                  LangGraph Workflow
                           |
                 +---------+---------+
                 |                   |
                 v                   v
          Classify Intent      General Question
                 |
                 v
          Policy Question
                 |
                 v
         Retrieve Documents
                 |
                 v
              ChromaDB
                 |
                 v
         Relevant Context
                 |
                 v
           Generate Answer
                 |
                 v
        Structured Response
                 |
                 v
              Streamlit

The same RAG pipeline can also be accessed through FastAPI:

Client
  |
  v
FastAPI /ask
  |
  v
Query Validation
  |
  v
LangGraph Workflow
  |
  v
RAG Pipeline
  |
  v
JSON Response

## Project Structure

```text
support_assistant/
├── docs/                       # Zepto support policy documents
│   ├── doc_01.txt
│   ├── doc_02.txt
│   ├── doc_03.txt
│   ├── doc_04.txt
│   ├── doc_05.txt
│   ├── doc_06.txt
│   ├── doc_07.txt
│   └── doc_08.txt
│
├── data/                       # Persistent ChromaDB storage
│   └── chroma/
│
├── ingest.py                   # Loads documents and creates embeddings
├── graph.py                    # LangGraph workflow and RAG logic
├── prompts.py                  # Prompt templates and instructions
├── main.py                     # FastAPI application
├── app.py                      # Streamlit user interface
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker container configuration
├── .env                        # Environment variables
└── README.md                   # Project documentation
```

# Support Documents

The application uses eight support documents.

| Document     | Topic                     |
| ------------ | ------------------------- |
| `doc_01.txt` | Delivery Policy           |
| `doc_02.txt` | Returns & Refunds         |
| `doc_03.txt` | Membership Tiers          |
| `doc_04.txt` | Order Tracking            |
| `doc_05.txt` | Order Cancellation Policy |
| `doc_06.txt` | Damaged or Missing Items  |
| `doc_07.txt` | Gift Cards                |
| `doc_08.txt` | Customer Support Hours    |

These documents provide the knowledge base used by the RAG pipeline.

# Technologies Used/Requirements

| Technology            | Purpose                         |
| --------------------- | ------------------------------- |
| Python                | Application development         |
| FastAPI               | REST API                        |
| LangGraph             | Workflow orchestration          |
| ChromaDB              | Vector database                 |
| Sentence Transformers | Text embeddings                 |
| Pydantic              | Request/response validation     |
| OpenAI API            | Optional real LLM               |
| python-dotenv         | Environment variable management |
| Docker                | Containerization                |

# Ingestion

1. `ingest.py` reads all 8 policy documents from `docs/doc_01.txt` through `docs/doc_08.txt`.
2. Each file is treated as a single chunk (the documents are short enough that per-document chunking is sufficient per the module spec).
3. Each chunk's `id` is the filename stem (e.g. `doc_01`), and its `metadata` records
the source filename.

# Embedding Model

1. Still in `ingest.py`, `embed_text()` uses the `sentence-transformers/all-MiniLM-L6-v2` model (loaded locally, no API key) to
encode each of the 8 document chunks into a vector. 
2. These vectors, along with their ids, raw text, and metadata, are upserted into a persistent ChromaDB
collection named `zepto_policies` (stored at `data/chroma`), configured to use
cosine similarity (`hnsw.space = "cosine"`).

# ChromaDB

ChromaDB is used as the vector database.
The persistent database is stored at: `data/chroma`
The collection used by the application is: `zepto_policies`
The collection uses cosine similarity for vector search.

# Prompt template

See `prompts.py` for the full text. It follows the role–context–task–format–length
skeleton, includes an explicit negative constraint ("Do not use information that
is not present in the retrieved context... Do not use outside knowledge"), and
two few-shot examples. It is only used by the optional `MOCK_LLM=0` path.

# LangGraph Workflow

The application uses LangGraph to organize the RAG workflow.

A simplified workflow is:

START
  |
  v
classify_intent
  |
  +--------------------+
  |                    |
  v                    v
policy_question   general_question
  |                    |
  v                    v
retrieve            generate
  |                    |
  v                    |
generate <-------------+
  |
  v
END

The workflow maintains state using `TypedDict`.

Example state:

```python
class SimpleState(TypedDict):
    query: str
    intent: str
    answer: str
    sources: list
    confidence: float
```

# Intent Classification

The application distinguishes between two types of questions.

### Policy Question

Questions related to Zepto policies, such as:
eg : What is the refund policy?

These questions are routed through document retrieval.

### General Question

Questions that are not related to the policy documents are handled separately.

For general questions:

```text
sources = []
```
# RAG Pipeline

The Retrieval-Augmented Generation pipeline works as follows:

```text
User Query
    |
    v
Intent Classification
    |
    v
Policy Question?
    |
   Yes
    |
    v
Create Query Embedding
    |
    v
Search ChromaDB
    |
    v
Retrieve Relevant Documents
    |
    v
Build Context
    |
    v
Prompt
    |
    v
LLM / MOCK_LLM
    |
    v
Structured Response
```

The retrieved document content provides grounding for policy-related answers.

# Structured Output

The API returns a structured response containing:

```json
{
    "answer": "Your order can be cancelled according to the cancellation policy.",
    "sources": ["doc_05.txt"],
    "confidence": 0.9
}
```

The response contains three fields:

| Field        | Description                           |
| ------------ | ------------------------------------- |
| `answer`     | Assistant's response                  |
| `sources`    | Documents used to answer the question |
| `confidence` | Confidence value between 0 and 1      |

The confidence value is validated using Pydantic.

# Retrieval

1. `graph.py`'s `retrieve_and_answer` node embeds the incoming query with the same `embed_text()` function, then queries the `zepto_policies` Chroma collection for the top-3 most similar chunks by cosine similarity.
2.  This step always runs for real, in both mock and real-LLM modes, since embedding and ChromaDB require no API key or network call.

# Generation

1. Also inside `retrieve_and_answer` (for policy questions) and `direct_answer` (for general questions). 2. This is the only stage that branches on the `MOCK_LLM` environment variable.

## `MOCK_LLM` unset or `1` (default — graded baseline):** No LLM is called.

  - `retrieve_and_answer` returns a canned string, `f"Based on the retrieved context:
     {top_chunk_snippet}"`, where `top_chunk_snippet` is the first ~200 characters of the top retrieved chunk.
  - `direct_answer` returns a fixed string:
     `"I can only answer questions about Zepto policies right now."`
  - In both cases the `SupportResponse` Pydantic schema (`answer`, `sources`,
     `confidence`) is populated deterministically in code — `sources` are the
     retrieved chunk ids for policy questions, empty for general questions;
    `confidence` is a fixed `1.0` / `0.0`.

Examples:

    Query : "what is the delivery time?"
    Mock_LLM response :
             {'query': 'what is the delivery time?',
              'intent': 'policy_question', 
              'answer': 'Based on the retrieved context: "Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer\'s delivery zone and current order volume. Standard de', 
              'sources': ['doc_01', 'doc_02', 'doc_04'], 
              'confidence': 1.0}
  
    Query : "what is the capital of India?"
    LLM Response :
            {'query': 'what is the capital of India', 
            'intent': 'general_question', 
            'answer': 'I can only answer questions about Zepto policies right now.', 
            'sources': [], 
            'confidence': 1.0}

## `MOCK_LLM=0` (Real LLM (`MOCK_LLM=0` via Groq free tier)): 

  - The structured prompt template in `prompts.py` (role–context–task–format–length skeleton, with a
     negative constraint and two few-shot examples) is filled with the retrieved context and sent to a real LLM (Groq's free tier, `openai/gpt-oss-120b`) via `call_llm()`.
  -  The LLM's raw output is parsed as JSON and validated against
     `SupportResponse`; on parse or validation failure, the request is retried up
     to 2 additional times with a corrective instruction naming the exact error
     before falling back to a clearly marked error response.

Examples:

    Query : "what is the delivery time?"
    LLM response :
                {'query': 'what is the delivery time?', 
                'intent': 'policy_question', 
                'answer': "Delivery takes between 10 and 30 minutes after order confirmation, depending on the customer's delivery zone and current order volume.", 
                'sources':['doc_01', 'doc_02', 'doc_04'], 
                'confidence': 0.99}

    Query : "what is the capital of India?"
    LLM Response : 
            {'query': 'what is the capital of India', 
            'intent': 'general_question', 
            'answer': 'The capital of India is New Delhi. It serves as the seat of the government and is located in the northern part of the country.', 
            'sources': [], 
            'confidence': 1.0}

# Streamlit Application Entry Point

The Streamlit application is started using:
SS
streamlit run app.py

For Docker deployment, the application listens on:

0.0.0.0:7860

The Docker container therefore exposes the Streamlit interface through port 7860.


# Docker

### Build the image

From the `support_assistant` directory:

```powershell
docker build -t zepto-support-assistant .
```
### Run the container

```powershell
docker run -p 7860:7860 zepto-support-assistant
```
The container installs dependencies from `requirements.txt`, copies in
`main.py`, `graph.py`, `ingest.py`, `prompts.py`, and the `docs/` corpus,
and serves the FastAPI app via Uvicorn on `0.0.0.0:7860` inside the
container, mapped to `localhost:7860` on the host.

## Tech stack

- Embeddings: `sentence-transformers` (`all-MiniLM-L6-v2`), local, no API key
- Vector store: ChromaDB (persistent, cosine similarity)
- Orchestration: LangGraph (`StateGraph`, `TypedDict` state)
- Serving: FastAPI + Uvicorn
- Schema validation: Pydantic (`SupportResponse` / `ChatResponse`)
- Optional LLM backend: Groq free tier (`openai/gpt-oss-120b`)

# 26. End-to-End Flow

The complete application can be summarized as:

                         ┌──────────────────┐
                         │       User       │
                         └────────┬─────────┘
                                  |
                                  v
                         ┌──────────────────┐
                         │  Streamlit UI    │
                         │     app.py       │
                         └────────┬─────────┘
                                  |
                                  v
                         ┌──────────────────┐
                         │ LangGraph Router │
                         └────────┬─────────┘
                                  |
                    ┌─────────────┴─────────────┐
                    |                           |
                    v                           v
             Policy Question             General Question
                    |                           |
                    v                           v
             Query Embedding              Direct Response
                    |
                    v
             ┌──────────────┐
             │   ChromaDB   │
             │  Retrieval   │
             └──────┬───────┘
                    |
                    v
             Retrieved Context
                    |
                    v
             ┌──────────────┐
             │ MOCK / LLM   │
             └──────┬───────┘
                    |
                    v
           Structured Response
                    |
                    v
             Streamlit UI

The same LangGraph/RAG pipeline can also be accessed through:

Client
  |
  v
FastAPI /ask
  |
  v
LangGraph
  |
  v
Structured JSON

# 29. Conclusion

The Zepto Support Assistant demonstrates a complete small-scale GenAI application:
The application is designed to provide grounded responses to policy-related questions while supporting deterministic offline execution through mock mode.
