# UdaPlay — AI Game Research Agent

An AI-powered research assistant for the video game industry.  
Given a natural-language question it retrieves relevant games from a local
semantic vault, evaluates the quality of the results, and falls back to live
web search when the vault falls short — returning a structured, cited report.

---

## Project Structure

```
udaplay-ai-agent/
├── Udaplay_01_solution_project.py   # Part 1 — build the game knowledge vault
├── Udaplay_02_solution_project.py   # Part 2 — run the research agent
├── games/                           # 15 game JSON files (the catalogue)
├── lib/                             # shared AI utilities
├── requirements.txt
├── config.env.example               # copy to config.env and add your keys
└── README.md
```

---

## Setup

### 1. Clone and enter the repo
```bash
git clone <repo-url>
cd udaplay-ai-agent
```

### 2. Create a Python 3.11 virtual environment
```bash
python3.11 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure credentials
```bash
cp config.env.example config.env
# Edit config.env and fill in your keys
```

`config.env` must contain:
```
OPENAI_API_KEY=...
OPENAI_BASE_URL=https://openai.vocareum.com/v1
TAVILY_API_KEY=...
```

---

## Running

**Part 1 — Build the knowledge vault (run once)**
```bash
python Udaplay_01_solution_project.py
```
Loads all 15 games from `games/` into a persistent ChromaDB collection and
runs semantic search smoke tests to confirm everything is indexed correctly.

**Part 2 — Run the research agent**
```bash
python Udaplay_02_solution_project.py
```
Executes three example queries demonstrating:
- Internal vault retrieval (no web search needed)
- Web search fallback (game not in the local catalogue)
- Multi-game query with long-term session memory

---

## Workflow (Part 2)

```
memory_probe → vault_search → quality_check
                                   ↓
                          passed?  ├── Yes → synthesise
                                   └── No  → web_lookup → synthesise
                                                               ↓
                                                           archive
```

Each step is a node in a `StateMachine` — the LLM cannot skip or reorder steps.

---

## Output Schema

Every query returns a `ResearchFindings` report:

| Field | Description |
|---|---|
| `summary` | Concise answer |
| `data_points` | Key extracted facts |
| `sources` | Cited game titles and/or URLs |
| `reliability` | `high` / `medium` / `low` |
| `web_assisted` | Whether Tavily was invoked |
| `quality_note` | Vault quality check summary |

---

## Author
Amala Deepan Antony Samy
