# Docker

The compose file starts:

- `backend`
- `frontend`
- optional `qdrant`

TigerGraph GraphRAG is intentionally kept as a separate official deployment under `graphrag/`. This avoids forking or reimplementing the official stack and matches the hackathon requirement to build on top of the TigerGraph repo.

If port `8008` is already used by a local `uvicorn` process, either stop that process or run:

```bash
BACKEND_HOST_PORT=8009 NEXT_PUBLIC_API_BASE=http://localhost:8009 docker compose -f docker/docker-compose.yml up --build
```

The backend container reaches host Ollama through `host.docker.internal:11434`, so keep `ollama serve` running on the host.

Recommended dev workflow:

```bash
make docker-build
make docker-up
```

After the first build, do not pass `--build` unless `pyproject.toml`, `frontend/package-lock.json`, or a Dockerfile changed. Backend and frontend source directories are bind-mounted for development.
