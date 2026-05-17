"""
Cloud-based GraphRAG service using TigerGraph Cloud/Savanna + Gemini API.

Pipeline:
  User query
  → Retrieve chunks from Chroma (passed in by caller)
  → Extract entities + relationships via Gemini
  → Upsert/query TigerGraph Cloud (if credentials set) OR NetworkX fallback
  → Build enriched context from graph traversal
  → Generate final answer + reasoning via Gemini

Environment variables (all optional — NetworkX fallback is always available):
  TIGERGRAPH_URL      TigerGraph Cloud workspace URL (e.g. https://xxx.i.tgcloud.io)
  TIGERGRAPH_API_KEY  TigerGraph Cloud API key / Bearer token
  GEMINI_API_KEY      Google Gemini API key (preferred)
  GOOGLE_API_KEY      Fallback if GEMINI_API_KEY is not set
"""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, Any

import httpx
import networkx as nx

logger = logging.getLogger(__name__)

_GEMINI_BASE = "https://generativelanguage.googleapis.com"

if TYPE_CHECKING:
    from app.core.config import Settings


class CloudGraphRAGService:
    """GraphRAG using Gemini for LLM + TigerGraph Cloud or NetworkX for graph storage."""

    def __init__(
        self,
        gemini_api_key: str,
        tigergraph_url: str | None = None,
        tigergraph_api_key: str | None = None,
        graph_name: str = "PaperWeave",
        model: str = "gemini-2.0-flash",
    ):
        self.gemini_api_key = gemini_api_key
        self.tigergraph_url = tigergraph_url.rstrip("/") if tigergraph_url else None
        self.tigergraph_api_key = tigergraph_api_key
        self.graph_name = graph_name
        self.model = model
        # Persistent session-level graph for accumulated context across queries
        self._session_graph: nx.Graph = nx.Graph()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def answer(
        self,
        question: str,
        chunks: list[dict[str, Any]],
        top_k: int = 5,
    ) -> dict[str, Any]:
        """
        Full GraphRAG pipeline returning answer, reasoning, and graph metadata.

        Args:
            question: User query string
            chunks:   Retrieved document chunks (dicts with 'text' or 'snippet' key)
            top_k:    Maximum number of chunks to use

        Returns:
            Dict with keys: answer, reasoning, entities, relationships,
                            graph_nodes, graph_edges, graph_context, base_context
        """
        chunk_texts = [
            c.get("text", c.get("snippet", ""))
            for c in chunks[:top_k]
            if c.get("text") or c.get("snippet")
        ]
        base_context = "\n\n".join(
            f"[{i + 1}] {text[:800]}" for i, text in enumerate(chunk_texts)
        )

        entities: list[dict] = []
        relationships: list[dict] = []
        graph_context = ""
        graph_nodes: list[dict] = []
        graph_edges: list[dict] = []

        if self.gemini_api_key:
            # Step 1: Entity + relationship extraction
            entities, relationships = await self._extract_entities(question, base_context)

            # Step 2: Graph query (TigerGraph Cloud → NetworkX)
            graph_context, graph_nodes, graph_edges = await self._query_graph(
                question, entities, relationships, chunk_texts
            )
        else:
            logger.warning("No Gemini API key — skipping entity extraction and graph enrichment")

        enriched_context = base_context
        if graph_context:
            enriched_context += f"\n\nGraph-enriched relationships:\n{graph_context}"

        # Step 3: Generate answer + reasoning
        if self.gemini_api_key:
            answer_text, reasoning = await self._gemini_answer(
                question, enriched_context, entities
            )
        else:
            answer_text = (
                "GraphRAG is not configured: set GEMINI_API_KEY in your environment. "
                "The basic-rag pipeline is available as a fallback."
            )
            reasoning = ""

        return {
            "answer": answer_text,
            "reasoning": reasoning,
            "entities": entities,
            "relationships": relationships,
            "graph_nodes": graph_nodes,
            "graph_edges": graph_edges,
            "graph_context": graph_context,
            "base_context": base_context,
        }

    # ------------------------------------------------------------------
    # Entity / relationship extraction
    # ------------------------------------------------------------------

    async def _extract_entities(
        self, question: str, context: str
    ) -> tuple[list[dict], list[dict]]:
        prompt = (
            "You are an expert at extracting knowledge graphs from AI/ML research papers.\n\n"
            f"Question: {question}\n\nContext:\n{context[:2500]}\n\n"
            "Extract key entities and relationships. Return ONLY valid JSON:\n"
            "{\n"
            '  "entities": [\n'
            '    {"name": "...", "type": "concept|method|model|dataset|author", "description": "..."}\n'
            "  ],\n"
            '  "relationships": [\n'
            '    {"source": "...", "target": "...", "type": "...", "description": "..."}\n'
            "  ]\n"
            "}\n"
            "Limit: 10 entities, 15 relationships. Focus on what is most relevant to the question."
        )
        try:
            raw = await self._gemini_generate(prompt, temperature=0.0, max_tokens=1024)
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(raw[start:end])
                return data.get("entities", []), data.get("relationships", [])
        except Exception as exc:
            logger.warning("Entity extraction failed: %s", exc)
        return [], []

    # ------------------------------------------------------------------
    # Graph operations (TigerGraph Cloud → NetworkX)
    # ------------------------------------------------------------------

    async def _query_graph(
        self,
        question: str,
        entities: list[dict],
        relationships: list[dict],
        chunk_texts: list[str],
    ) -> tuple[str, list[dict], list[dict]]:
        if self.tigergraph_url and self.tigergraph_api_key:
            try:
                ctx, nodes, edges = await self._tigergraph_query(
                    question, entities, relationships
                )
                if ctx or nodes:
                    return ctx, nodes, edges
            except Exception as exc:
                logger.warning("TigerGraph Cloud failed, falling back to NetworkX: %s", exc)

        return self._networkx_query(question, entities, relationships, chunk_texts)

    async def _tigergraph_query(
        self,
        question: str,
        entities: list[dict],
        relationships: list[dict],
    ) -> tuple[str, list[dict], list[dict]]:
        """
        Upsert entities/relationships to TigerGraph Cloud via RESTPP and query neighbors.

        Requires a TigerGraph graph with:
          - Vertex type:  Entity  (attributes: entity_type STRING, description STRING)
          - Edge type:    RelatedTo (Entity → Entity, attributes: relation_type STRING)
        """
        headers = {
            "Authorization": f"Bearer {self.tigergraph_api_key}",
            "Content-Type": "application/json",
        }
        base = self.tigergraph_url

        async with httpx.AsyncClient(timeout=20.0) as client:
            # --- Upsert vertices ---
            vertex_payload: dict[str, Any] = {}
            for entity in entities:
                name = entity.get("name", "").strip()
                if not name:
                    continue
                vertex_payload[name] = {
                    "entity_type": entity.get("type", "concept"),
                    "description": entity.get("description", ""),
                }

            if vertex_payload:
                upsert_body: dict[str, Any] = {
                    "vertices": {"Entity": vertex_payload},
                    "edges": {},
                }
                # Add edges if both endpoints exist
                for rel in relationships:
                    src = rel.get("source", "").strip()
                    tgt = rel.get("target", "").strip()
                    if src and tgt and src in vertex_payload and tgt in vertex_payload:
                        upsert_body["edges"].setdefault("RelatedTo", {})
                        upsert_body["edges"]["RelatedTo"].setdefault(src, {})[tgt] = {
                            "relation_type": rel.get("type", "related_to"),
                        }

                upsert_resp = await client.post(
                    f"{base}/restpp/upsert/{self.graph_name}",
                    headers=headers,
                    json=upsert_body,
                )
                if upsert_resp.status_code >= 400:
                    raise RuntimeError(
                        f"TigerGraph upsert failed HTTP {upsert_resp.status_code}: "
                        f"{upsert_resp.text[:300]}"
                    )

            # --- Query vertices for context ---
            context_parts: list[str] = []
            graph_nodes: list[dict] = []
            graph_edges: list[dict] = []

            for entity in entities[:6]:
                name = entity.get("name", "").strip()
                if not name:
                    continue
                resp = await client.get(
                    f"{base}/restpp/graph/{self.graph_name}/vertices/Entity/{name}",
                    headers=headers,
                )
                if resp.status_code == 200:
                    for result in resp.json().get("results", []):
                        attrs = result.get("attributes", {})
                        desc = attrs.get("description", "")
                        graph_nodes.append({"id": name, "type": attrs.get("entity_type", "concept"), "description": desc})
                        if desc:
                            context_parts.append(f"{name}: {desc}")

                # Query 1-hop neighbors
                edge_resp = await client.get(
                    f"{base}/restpp/graph/{self.graph_name}/edges/Entity/{name}",
                    headers=headers,
                )
                if edge_resp.status_code == 200:
                    for edge in edge_resp.json().get("results", []):
                        tgt_id = edge.get("to_id", "")
                        rel_type = edge.get("attributes", {}).get("relation_type", "related_to")
                        if tgt_id:
                            graph_edges.append({"source": name, "target": tgt_id, "type": rel_type})
                            context_parts.append(f"'{name}' {rel_type} '{tgt_id}'")

        return "\n".join(context_parts[:15]), graph_nodes, graph_edges

    def _networkx_query(
        self,
        question: str,
        entities: list[dict],
        relationships: list[dict],
        chunk_texts: list[str],
    ) -> tuple[str, list[dict], list[dict]]:
        """Build a per-request NetworkX graph and traverse it for enriched context."""
        local_graph: nx.Graph = nx.Graph()

        for entity in entities:
            name = entity.get("name", "").strip()
            if name:
                local_graph.add_node(
                    name,
                    entity_type=entity.get("type", "concept"),
                    description=entity.get("description", ""),
                )
                # Accumulate into session graph for cross-query context
                if not self._session_graph.has_node(name):
                    self._session_graph.add_node(name, entity_type=entity.get("type", "concept"), description=entity.get("description", ""))

        for rel in relationships:
            src = rel.get("source", "").strip()
            tgt = rel.get("target", "").strip()
            if src and tgt and local_graph.has_node(src) and local_graph.has_node(tgt):
                attrs = {"relation_type": rel.get("type", "related_to"), "description": rel.get("description", "")}
                local_graph.add_edge(src, tgt, **attrs)
                if not self._session_graph.has_node(tgt):
                    self._session_graph.add_node(tgt, entity_type="concept", description="")
                if not self._session_graph.has_edge(src, tgt):
                    self._session_graph.add_edge(src, tgt, **attrs)

        # Find entities relevant to the question
        q_terms = {t.lower() for t in question.split() if len(t) > 3}
        context_parts: list[str] = []

        for node in local_graph.nodes():
            node_terms = {t.lower() for t in node.split()}
            is_relevant = bool(node_terms & q_terms) or any(
                t in node.lower() for t in q_terms if len(t) > 4
            )
            if not is_relevant:
                continue

            node_data = local_graph.nodes[node]
            desc = node_data.get("description", "")
            if desc:
                context_parts.append(f"Entity '{node}': {desc}")

            for neighbor in list(local_graph.neighbors(node))[:4]:
                edge = local_graph.edges[node, neighbor]
                rel_type = edge.get("relation_type", "related_to")
                rel_desc = edge.get("description", "")
                neighbor_desc = local_graph.nodes[neighbor].get("description", "")
                context_parts.append(
                    f"'{node}' {rel_type} '{neighbor}': {rel_desc or neighbor_desc}"
                )

        # Also pull in session-level context for persistent cross-query enrichment
        if self._session_graph.number_of_nodes() > local_graph.number_of_nodes():
            for node in list(self._session_graph.nodes())[:20]:
                if node not in local_graph:
                    node_terms = {t.lower() for t in node.split()}
                    if node_terms & q_terms:
                        desc = self._session_graph.nodes[node].get("description", "")
                        if desc:
                            context_parts.append(f"[Prior context] '{node}': {desc}")

        graph_nodes = [
            {
                "id": n,
                "type": local_graph.nodes[n].get("entity_type", "concept"),
                "description": local_graph.nodes[n].get("description", ""),
            }
            for n in local_graph.nodes()
        ]
        graph_edges = [
            {
                "source": u,
                "target": v,
                "type": local_graph.edges[u, v].get("relation_type", "related_to"),
            }
            for u, v in local_graph.edges()
        ]

        return "\n".join(context_parts[:20]), graph_nodes, graph_edges

    # ------------------------------------------------------------------
    # Answer generation
    # ------------------------------------------------------------------

    async def _gemini_answer(
        self,
        question: str,
        context: str,
        entities: list[dict],
    ) -> tuple[str, str]:
        entity_list = ", ".join(
            e.get("name", "") for e in entities[:8] if e.get("name")
        )
        prompt = (
            "You are PaperWeave's GraphRAG system — an expert at answering questions about AI/ML research papers "
            "using both vector-retrieved context AND knowledge-graph enrichment.\n\n"
            f"Key entities identified: {entity_list or 'N/A'}\n\n"
            f"Question: {question}\n\n"
            f"Retrieved + graph-enriched context:\n{context[:3500]}\n\n"
            "Provide a comprehensive, grounded answer. Format exactly as:\n\n"
            "ANSWER:\n"
            "[Your answer here, citing sources as [1], [2], etc. where applicable]\n\n"
            "GRAPH REASONING:\n"
            "[One short paragraph explaining which graph relationships enriched your answer]"
        )
        try:
            raw = await self._gemini_generate(prompt, temperature=0.1, max_tokens=1024)
            if "GRAPH REASONING:" in raw:
                parts = raw.split("GRAPH REASONING:", 1)
                answer_part = parts[0].replace("ANSWER:", "").strip()
                reasoning_part = parts[1].strip()
            else:
                answer_part = raw.replace("ANSWER:", "").strip()
                reasoning_part = "Graph traversal provided entity relationship context."
            return answer_part, reasoning_part
        except Exception as exc:
            logger.error("Gemini answer generation failed: %s", exc)
            return f"GraphRAG generation failed: {exc}", ""

    # ------------------------------------------------------------------
    # Gemini API helper
    # ------------------------------------------------------------------

    async def _gemini_generate(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> str:
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{_GEMINI_BASE}/v1beta/models/{self.model}:generateContent"
                f"?key={self.gemini_api_key}",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]


def build_cloud_graphrag_service(settings: "Settings", graph_name: str = "PaperWeave") -> CloudGraphRAGService:
    """Factory — prefers app settings for Gemini and env vars for TigerGraph cloud connectivity."""
    llm_config = settings.providers.llm
    llm_api_env = llm_config.api_key_env or "GEMINI_API_KEY"
    gemini_key = os.getenv(llm_api_env, "") or os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
    tg_url = os.getenv("TIGERGRAPH_URL", "").strip() or None
    tg_api_key = os.getenv("TIGERGRAPH_API_KEY", "").strip() or None
    model = os.getenv("GRAPHRAG_GEMINI_MODEL", llm_config.model)
    return CloudGraphRAGService(
        gemini_api_key=gemini_key,
        tigergraph_url=tg_url,
        tigergraph_api_key=tg_api_key,
        graph_name=graph_name,
        model=model,
    )
