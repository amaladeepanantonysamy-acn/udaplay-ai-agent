# UdaPlay — AI Game Research Agent

An AI-powered research assistant for the video game industry.  
Given a natural-language question it retrieves relevant games from a local
semantic vault, evaluates the quality of the results, and falls back to live
web search when the vault falls short — returning a structured, cited answer.

---

## Project Structure

```
udaplay-ai-agent/
├── Udaplay_01_solution_project.ipynb   # Part 1 — build the game knowledge vault
├── Udaplay_02_solution_project.ipynb   # Part 2 — run the research agent
├── games/                              # 15 game JSON files (the catalogue)
├── lib/                                # shared AI utilities
│   ├── agents.py
│   ├── llm.py
│   ├── messages.py
│   ├── parsers.py
│   ├── state_machine.py
│   ├── tooling.py
│   └── memory.py
├── requirements.txt
├── config.env.example                  # copy to config.env and add your keys
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

Open `Udaplay_01_solution_project.ipynb` in Jupyter and run all cells.  
Loads all 15 games from `games/` into a persistent ChromaDB collection and
runs semantic search smoke tests to confirm everything is indexed correctly.

**Part 2 — Run the research agent**

Open `Udaplay_02_solution_project.ipynb` in Jupyter and run all cells.  
Executes three example queries demonstrating:
- Internal vault retrieval (game is in the local catalogue)
- Retrieval evaluation to assess result quality
- Web search fallback via Tavily (game not in the local catalogue)

---

## Workflow (Part 2)

```
vault_search → quality_check
                    ↓
           passed?  ├── Yes → synthesise → answer with Source (internal DB)
                    └── No  → web_lookup → synthesise → answer with Source (URL)
```

Each step is a node in a `StateMachine` — the LLM cannot skip or reorder steps.

---

## Agent Tools

| Tool | Description |
|---|---|
| `retrieve_game` | Semantic search against the local ChromaDB vault |
| `evaluate_retrieval` | LLM judge — decides if retrieved results are sufficient |
| `game_web_search` | Tavily web search fallback when the vault falls short |

---

## Output Format

Every answer ends with a `Source:` citation line:

- **Internal DB path** — `Source: UdaPlay internal database.`
- **Web search path** — `Source: <URL(s) from Tavily search results>`

---

## Author
Amaladeepan Antonysamy
