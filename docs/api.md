# API

## Endpoints

- `POST /ask/llm-only`
- `POST /ask/basic-rag`
- `POST /ask/graphrag`
- `POST /ask/all`
- `POST /benchmark`
- `GET /metrics`
- `GET /health`

Each answer response includes:

- `answer`
- `tokens`
- `latency`
- `estimated_cost`
- `sources`
- `retrieval_info`
- `timing_breakdown`
