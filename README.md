# Agentic Research RAG

A modular Agentic RAG backend implementing:

Planner -> Discovery -> Ingestion -> Hybrid Retrieval -> Analysis -> Synthesis -> Claim Extraction -> NLI/Grounding -> Confidence -> Final / Retry / Human Review.

## Stack

- FastAPI
- LangGraph
- Groq `openai/gpt-oss-20b`
- Sentence Transformers `all-MiniLM-L6-v2`
- Cross-encoder reranker
- NLI cross-encoder
- PostgreSQL + pgvector
- Redis
- RabbitMQ

## 1. Configure

Copy `.env.example` to `.env` and add your Groq API key.

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

## 2. Start infrastructure

```bash
docker compose up -d
```

Services:

- PostgreSQL: `localhost:5432`
- Redis: `localhost:6379`
- RabbitMQ: `localhost:5672`
- RabbitMQ UI: `http://localhost:15672` (`guest` / `guest`)

## 3. Python environment

```bash
python -m venv venv
```

Windows:

```powershell
venv\Scripts\activate
```

Install:

```bash
pip install -r requirements.txt
```

## 4. Run API

```bash
uvicorn app.main:app --reload
```

Swagger:

`http://127.0.0.1:8000/docs`

## 5. Optional workers

```bash
python -m app.workers.ingestion_worker
python -m app.workers.retrieval_worker
python -m app.workers.analysis_worker
python -m app.workers.report_worker
```

## 6. Ingest internal knowledge

POST `/documents/text`

```json
{
  "tenant_id": "demo",
  "title": "Revenue Report",
  "text": "The company generated 80 million dollars in revenue in 2024 and 100 million dollars in 2025."
}
```

## 7. Run research

POST `/research`

```json
{
  "tenant_id": "demo",
  "query": "How much did company revenue grow from 2024 to 2025?"
}
```

## Runtime pipeline

```text
User Query
   |
   v
Planner (GPT-OSS 20B)
   |
   v
Discovery (Groq Browser Search)
   |
   v
Ingestion + Hybrid Chunking
   |
   v
SentenceTransformer Embeddings
   |
   v
PostgreSQL + pgvector
   |
   v
Dense Vector Search + PostgreSQL FTS
   |
   v
Reciprocal Rank Fusion
   |
   v
CrossEncoder Reranker
   |
   v
Analysis Agent + Calculator
   |
   v
Synthesis Agent
   |
   v
Claim Extraction
   |
   v
NLI Gate
   |
   v
Grounding / Faithfulness / Hallucination / Citation Scores
   |
   v
Confidence Gate
   |---------------------------|
   |             |             |
 Final         Retry       Human Review
```

## Notes

This is a strong reference implementation, not a drop-in production security baseline. Before production deployment, add authentication, tenant authorization, request quotas, stricter URL allow/deny rules, SSRF protection, secrets management, observability, migrations, connection health checks, and worker retry/dead-letter policies.
