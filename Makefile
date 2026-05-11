COMPOSE=docker compose -f docker/docker-compose.yml
GRAPHRAG_COMPOSE=docker compose -f graphrag/tigergraph-graphrag/docker-compose.paperweave.yml

.PHONY: docker-build docker-up docker-down docker-logs docker-rebuild docker-prune graphrag-build graphrag-up graphrag-down graphrag-logs

docker-build:
	$(COMPOSE) build

docker-up:
	$(COMPOSE) up

docker-down:
	$(COMPOSE) down

docker-logs:
	$(COMPOSE) logs -f

docker-rebuild:
	$(COMPOSE) build --no-cache

docker-prune:
	docker builder prune

graphrag-build:
	cd graphrag/tigergraph-graphrag && docker compose -f docker-compose.paperweave.yml build

graphrag-up:
	cd graphrag/tigergraph-graphrag && docker compose -f docker-compose.paperweave.yml up -d

graphrag-down:
	cd graphrag/tigergraph-graphrag && docker compose -f docker-compose.paperweave.yml down

graphrag-logs:
	cd graphrag/tigergraph-graphrag && docker compose -f docker-compose.paperweave.yml logs -f
