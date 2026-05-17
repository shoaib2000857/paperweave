"""Generate PaperWeave architecture diagram as PNG."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

fig, ax = plt.subplots(figsize=(20, 14))
ax.set_xlim(0, 20)
ax.set_ylim(0, 14)
ax.axis("off")
fig.patch.set_facecolor("#0f1117")

# ── colour palette ──────────────────────────────────────────────────────
C_BG       = "#0f1117"
C_PANEL    = "#1a1d2e"
C_BORDER   = "#2d3154"
C_BLUE     = "#4f79e8"
C_TEAL     = "#2ec4b6"
C_PURPLE   = "#9b59f5"
C_ORANGE   = "#f5a623"
C_GREEN    = "#27ae60"
C_RED      = "#e74c3c"
C_TEXT     = "#e8eaf6"
C_SUBTEXT  = "#9fa8da"
C_ARROW    = "#5c6bc0"

def box(ax, x, y, w, h, color, alpha=1.0, radius=0.3, zorder=2):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0,rounding_size={radius}",
                       facecolor=color, edgecolor="none",
                       alpha=alpha, zorder=zorder)
    ax.add_patch(p)
    return p

def outline_box(ax, x, y, w, h, fc, ec, lw=1.5, radius=0.3, zorder=2, alpha=1.0):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0,rounding_size={radius}",
                       facecolor=fc, edgecolor=ec, linewidth=lw,
                       alpha=alpha, zorder=zorder)
    ax.add_patch(p)
    return p

def label(ax, x, y, text, size=10, color=C_TEXT, weight="normal", ha="center", va="center", zorder=5):
    ax.text(x, y, text, fontsize=size, color=color, fontweight=weight,
            ha=ha, va=va, zorder=zorder, fontfamily="monospace")

def arrow(ax, x1, y1, x2, y2, color=C_ARROW, lw=1.8, style="->"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, connectionstyle="arc3,rad=0.0"),
                zorder=4)

def dashed_arrow(ax, x1, y1, x2, y2, color=C_ARROW, lw=1.5):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                                linestyle="dashed",
                                connectionstyle="arc3,rad=0.0"),
                zorder=4)

# ════════════════════════════════════════════════════════════════════════
# TITLE
# ════════════════════════════════════════════════════════════════════════
ax.text(10, 13.4, "PaperWeave — System Architecture",
        fontsize=18, color=C_TEXT, fontweight="bold",
        ha="center", va="center", fontfamily="monospace")
ax.text(10, 13.0, "Multi-Pipeline RAG QA Comparison  |  Live Evaluation  |  Hackathon Leaderboard",
        fontsize=9.5, color=C_SUBTEXT, ha="center", va="center", fontfamily="monospace")

# ════════════════════════════════════════════════════════════════════════
# LAYER 1 — USER / FRONTEND  (top strip)
# ════════════════════════════════════════════════════════════════════════
outline_box(ax, 0.4, 11.2, 19.2, 1.5, fc="#141729", ec=C_BLUE, lw=1.5, zorder=1)
label(ax, 1.6, 12.55, "FRONTEND", 8, C_BLUE, "bold", ha="center")
label(ax, 1.6, 12.2, "Next.js 14", 7.5, C_SUBTEXT)

components = [
    ("query-form.tsx\n(Submit question)", 4.0),
    ("result-card.tsx\n(Pipeline answer)", 7.2),
    ("evaluation-dashboard.tsx\n(Live metrics / leaderboard)", 11.2),
    ("comparison-chart.tsx\n(Score radar chart)", 15.6),
]
for name, cx in components:
    outline_box(ax, cx - 1.6, 11.45, 3.2, 1.0, fc="#1e2235", ec=C_BLUE, lw=1.0, radius=0.2, zorder=3)
    label(ax, cx, 11.95, name, 7.5, C_TEXT, ha="center")

# ════════════════════════════════════════════════════════════════════════
# LAYER 2 — FASTAPI BACKEND
# ════════════════════════════════════════════════════════════════════════
outline_box(ax, 0.4, 8.5, 19.2, 2.5, fc="#141729", ec=C_TEAL, lw=1.5, zorder=1)
label(ax, 1.6, 10.7, "BACKEND", 8, C_TEAL, "bold")
label(ax, 1.6, 10.35, "FastAPI", 7.5, C_SUBTEXT)

# Routes
routes = [
    ("POST /ask/all\norchestrator", 4.0, C_TEAL),
    ("GET /evaluation/results\nlive + offline merge", 7.5, C_TEAL),
    ("GET /benchmark\nstored results", 11.0, C_TEAL),
    ("GET /metrics\ntoken / latency", 14.5, C_TEAL),
]
for name, cx, col in routes:
    outline_box(ax, cx - 1.55, 9.55, 3.1, 1.2, fc="#192030", ec=col, lw=1.0, radius=0.2, zorder=3)
    label(ax, cx, 10.15, name, 7.5, C_TEXT, ha="center")

# ════════════════════════════════════════════════════════════════════════
# LAYER 3 — THREE PIPELINES
# ════════════════════════════════════════════════════════════════════════
outline_box(ax, 0.4, 5.5, 19.2, 2.7, fc="#141729", ec=C_PURPLE, lw=1.5, zorder=1)
label(ax, 1.7, 7.9, "PIPELINES", 8, C_PURPLE, "bold")
label(ax, 1.7, 7.55, "asyncio.gather", 7.5, C_SUBTEXT)

# LLM-Only
outline_box(ax, 2.0, 5.75, 4.2, 1.95, fc="#221a33", ec=C_PURPLE, lw=1.2, radius=0.25, zorder=3)
label(ax, 4.1, 7.45, "LLM-Only Pipeline", 8.5, C_PURPLE, "bold")
label(ax, 4.1, 7.08, "llm_only.py", 7.5, C_SUBTEXT)
label(ax, 4.1, 6.75, "Direct LLM call", 7.5, C_TEXT)
label(ax, 4.1, 6.45, "no retrieval", 7.5, C_SUBTEXT)
label(ax, 4.1, 6.1, "Claude / GPT-4", 7.5, C_ORANGE)

# Basic RAG
outline_box(ax, 7.4, 5.75, 4.8, 1.95, fc="#1a2233", ec=C_BLUE, lw=1.2, radius=0.25, zorder=3)
label(ax, 9.8, 7.45, "Basic RAG Pipeline", 8.5, C_BLUE, "bold")
label(ax, 9.8, 7.08, "basic_rag.py", 7.5, C_SUBTEXT)
label(ax, 9.8, 6.75, "Embed → Chroma", 7.5, C_TEXT)
label(ax, 9.8, 6.45, "top-k chunks → LLM", 7.5, C_TEXT)
label(ax, 9.8, 6.1, "ChromaDB", 7.5, C_BLUE)

# GraphRAG
outline_box(ax, 13.5, 5.75, 5.0, 1.95, fc="#1a2a1a", ec=C_GREEN, lw=1.2, radius=0.25, zorder=3)
label(ax, 16.0, 7.45, "TigerGraph GraphRAG", 8.5, C_GREEN, "bold")
label(ax, 16.0, 7.08, "graphrag.py", 7.5, C_SUBTEXT)
label(ax, 16.0, 6.75, "Graph traversal", 7.5, C_TEXT)
label(ax, 16.0, 6.45, "entity relations → LLM", 7.5, C_TEXT)
label(ax, 16.0, 6.1, "TigerGraph DB", 7.5, C_GREEN)

# ════════════════════════════════════════════════════════════════════════
# LAYER 4 — LIVE EVALUATION
# ════════════════════════════════════════════════════════════════════════
outline_box(ax, 0.4, 2.8, 19.2, 2.4, fc="#141729", ec=C_ORANGE, lw=1.5, zorder=1)
label(ax, 1.8, 4.9, "LIVE EVAL", 8, C_ORANGE, "bold")
label(ax, 1.8, 4.55, "live_evaluation.py", 7.5, C_SUBTEXT)

metrics = [
    ("BERTScore\nSemantic similarity", 4.2, C_ORANGE),
    ("LLM-as-Judge\nAnswer accuracy", 7.8, C_RED),
    ("Token Reduction\n30% weight", 11.3, C_TEAL),
    ("Hallucination\nDetection", 14.6, C_PURPLE),
    ("Retrieval\nQuality", 17.8, C_BLUE),
]
for name, cx, col in metrics:
    outline_box(ax, cx - 1.45, 3.05, 2.9, 1.5, fc="#1d1d2a", ec=col, lw=1.0, radius=0.2, zorder=3)
    label(ax, cx, 3.80, name, 7.5, C_TEXT, ha="center")

# Score formula
box(ax, 3.5, 2.82, 13.0, 0.38, "#25200a", zorder=3)
label(ax, 10.0, 3.01,
      "Weighted Score = Token Reduction×30% + Answer Accuracy×30% + Latency×20% + Storytelling×20%",
      7.5, C_ORANGE, ha="center")

# ════════════════════════════════════════════════════════════════════════
# LAYER 5 — DATA STORES + EVALUATION MODULE
# ════════════════════════════════════════════════════════════════════════
outline_box(ax, 0.4, 0.3, 19.2, 2.2, fc="#141729", ec=C_SUBTEXT, lw=1.0, zorder=1)
label(ax, 1.6, 2.2, "STORAGE &", 8, C_SUBTEXT, "bold")
label(ax, 1.6, 1.85, "EVAL MODULE", 8, C_SUBTEXT, "bold")

stores = [
    ("ChromaDB\n(Vector Store)", 3.8, C_BLUE),
    ("TigerGraph\n(Graph DB)", 7.0, C_GREEN),
    ("arxiv PDFs\n/data/papers/", 10.2, C_SUBTEXT),
    ("Benchmarks\nJSON store", 13.3, C_ORANGE),
    ("evaluation/\nmetrics · judge\nbertscore · dataset", 17.0, C_RED),
]
for name, cx, col in stores:
    outline_box(ax, cx - 1.5, 0.55, 3.0, 1.55, fc="#1a1a2e", ec=col, lw=1.0, radius=0.2, zorder=3)
    label(ax, cx, 1.33, name, 7.5, C_TEXT, ha="center")

# ════════════════════════════════════════════════════════════════════════
# ARROWS — inter-layer connections
# ════════════════════════════════════════════════════════════════════════
# Frontend → Backend
for fx in [4.0, 7.2, 11.2]:
    arrow(ax, fx, 11.45, fx, 10.75, C_BLUE)

# Backend /ask/all → Pipelines (fan-out)
ax.annotate("", xy=(4.1, 7.70), xytext=(4.0, 9.55),
            arrowprops=dict(arrowstyle="->", color=C_PURPLE, lw=1.8,
                            connectionstyle="arc3,rad=0.0"), zorder=4)
ax.annotate("", xy=(9.8, 7.70), xytext=(4.0, 9.55),
            arrowprops=dict(arrowstyle="->", color=C_PURPLE, lw=1.8,
                            connectionstyle="arc3,rad=0.2"), zorder=4)
ax.annotate("", xy=(16.0, 7.70), xytext=(4.0, 9.55),
            arrowprops=dict(arrowstyle="->", color=C_PURPLE, lw=1.8,
                            connectionstyle="arc3,rad=0.3"), zorder=4)

# Pipelines → Live Eval (fan-in)
for px in [4.1, 9.8, 16.0]:
    arrow(ax, px, 5.75, px, 4.55, C_ORANGE)

# Live Eval → Evaluation module store
arrow(ax, 10.0, 2.8, 17.0, 2.1, C_RED)

# RAG pipeline ↔ ChromaDB
dashed_arrow(ax, 8.5, 5.75, 3.8, 2.1)
# GraphRAG ↔ TigerGraph
dashed_arrow(ax, 14.5, 5.75, 7.0, 2.1)

# Benchmark route → JSON store
dashed_arrow(ax, 11.0, 9.55, 13.3, 2.1, C_ORANGE)

# ════════════════════════════════════════════════════════════════════════
# LEGEND
# ════════════════════════════════════════════════════════════════════════
legend_items = [
    (C_BLUE,   "Frontend / API"),
    (C_PURPLE, "Pipelines"),
    (C_ORANGE, "Live Evaluation"),
    (C_GREEN,  "Graph Store"),
    (C_RED,    "Eval Module"),
]
lx, ly = 0.45, 0.18
for i, (col, text) in enumerate(legend_items):
    cx = lx + i * 3.8
    ax.add_patch(plt.Rectangle((cx, ly - 0.06), 0.22, 0.12,
                                color=col, zorder=6))
    ax.text(cx + 0.3, ly, text, fontsize=7, color=C_SUBTEXT,
            va="center", fontfamily="monospace", zorder=6)

plt.tight_layout(pad=0)
plt.savefig("architecture.png", dpi=180, bbox_inches="tight",
            facecolor=C_BG, edgecolor="none")
print("Saved architecture.png")
