"""
UdaPlay - Part 02: Game Research Assistant
===========================================
An AI agent that answers video-game questions using a two-stage strategy:
  1. Retrieve candidates from the local semantic vault (Part 01)
  2. Grade the retrieval quality with an LLM judge
  3. Fall back to live Tavily web search when the local vault falls short
  4. Synthesise a structured ResearchFindings report with citations

Workflow (StateMachine — deterministic, not LLM-routed):
  memory_probe → vault_search → quality_check
      → [web_lookup if needed] → synthesise → archive

Author : Amala Deepan Antony Samy
"""

import importlib.util, sys

if importlib.util.find_spec("pysqlite3"):
    import pysqlite3
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import os
from typing import Optional, List
from typing_extensions import TypedDict
from datetime import datetime
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from tavily import TavilyClient

from lib.llm      import LLM
from lib.messages import UserMessage, SystemMessage
from lib.parsers  import JsonOutputParser
from lib.state_machine import StateMachine, Step, EntryPoint, Termination, Run
from lib.memory   import ShortTermMemory


# ── Credentials ───────────────────────────────────────────────────────────────

load_dotenv("config.env") if os.path.exists("config.env") else load_dotenv()

assert os.getenv("OPENAI_API_KEY"), "Set OPENAI_API_KEY in config.env"
assert os.getenv("TAVILY_API_KEY"), "Set TAVILY_API_KEY in config.env"

OPENAI_KEY  = os.environ["OPENAI_API_KEY"]
TAVILY_KEY  = os.environ["TAVILY_API_KEY"]
OPENAI_BASE = os.getenv("OPENAI_BASE_URL", "https://openai.vocareum.com/v1")

print(f"[config] base_url = {OPENAI_BASE}")


# ── Vault connection (built by Part 01) ───────────────────────────────────────

_embed_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_KEY, api_base=OPENAI_BASE, model_name="text-embedding-ada-002"
)
_db = chromadb.PersistentClient(path="chromadb")
game_vault = _db.get_or_create_collection("udaplay", embedding_function=_embed_fn)
print(f"[vault]  Connected — {game_vault.count()} games indexed")


# ── Pydantic output schemas ───────────────────────────────────────────────────

class QualityCheck(BaseModel):
    """LLM judge output: did the vault results actually answer the question?"""
    passed:    bool  = Field(description="True if the retrieved games answer the question")
    score:     int   = Field(description="Relevance score 0-10")
    reasoning: str   = Field(description="One-sentence rationale for the verdict")


class ResearchFindings(BaseModel):
    """Final structured report returned for every query."""
    summary:      str        = Field(description="Concise answer to the question")
    data_points:  List[str]  = Field(description="Key facts extracted (name, year, platform, etc.)")
    sources:      List[str]  = Field(description="Cited sources: game titles and/or URLs")
    reliability:  str        = Field(description="high | medium | low")
    web_assisted: bool       = Field(description="True if Tavily web search was used")
    quality_note: str        = Field(description="Summary of the vault quality check")


# ── Session state schema ──────────────────────────────────────────────────────

class ResearchSession(TypedDict):
    question:      str
    session_id:    str
    prior_context: Optional[str]    # from long-term memory
    vault_hits:    Optional[list]   # ChromaDB results
    quality:       Optional[dict]   # QualityCheck result
    web_data:      Optional[dict]   # Tavily results
    findings:      Optional[dict]   # ResearchFindings result


# ── Persistent session memory (bonus) ─────────────────────────────────────────

class SessionMemoryStore:
    """
    Persists Q&A pairs across runs using a dedicated ChromaDB collection.
    Surfaced at the start of each query so prior answers enrich new ones.
    """
    COLL = "udaplay_session_log"
    PATH = "chromadb_memory"

    def __init__(self):
        client = chromadb.PersistentClient(path=self.PATH)
        self._coll = client.get_or_create_collection(
            self.COLL, embedding_function=_embed_fn
        )

    def save(self, question: str, answer: str) -> None:
        uid = f"log_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        self._coll.add(
            ids=[uid],
            documents=[f"Q: {question}\nA: {answer[:400]}"],
            metadatas=[{"ts": datetime.now().isoformat()}],
        )

    def recall(self, question: str, k: int = 2) -> str:
        if self._coll.count() == 0:
            return ""
        res = self._coll.query(
            query_texts=[question],
            n_results=min(k, self._coll.count()),
            include=["documents"],
        )
        return "\n\n".join(res["documents"][0])


# ── Main agent class ──────────────────────────────────────────────────────────

class GameResearchAssistant:
    """
    Stateful agent that answers video-game questions with a guaranteed
    retrieve-then-evaluate-then-search workflow enforced at the graph level.
    """

    PERSONA = (
        "You are an expert video-game analyst with encyclopaedic knowledge of "
        "gaming history across all platforms and generations. You answer questions "
        "precisely, cite your sources, and flag when information may be incomplete."
    )

    def __init__(self):
        self._llm        = LLM(model="gpt-4o-mini", temperature=0.0)
        self._tavily     = TavilyClient(api_key=TAVILY_KEY)
        self._memory     = SessionMemoryStore()
        self._session    = ShortTermMemory()
        self._workflow   = self._build_graph()

    # ── Step 1: probe long-term memory ───────────────────────────────────────

    def _memory_probe(self, state: ResearchSession) -> dict:
        recalled = self._memory.recall(state["question"])
        tag = f"({len(recalled.splitlines())} prior lines)" if recalled else "(none)"
        print(f"  [1/memory_probe]  recalled context {tag}")
        return {"prior_context": recalled}

    # ── Step 2: search the local vault ────────────────────────────────────────

    def _vault_search(self, state: ResearchSession) -> dict:
        raw = game_vault.query(
            query_texts=[state["question"]],
            n_results=3,
            include=["documents", "metadatas", "distances"],
        )
        hits = [
            {
                "title":       m.get("Name"),
                "platform":    m.get("Platform"),
                "year":        m.get("YearOfRelease"),
                "genre":       m.get("Genre"),
                "publisher":   m.get("Publisher"),
                "description": m.get("Description"),
                "distance":    round(d, 4),
            }
            for m, d in zip(raw["metadatas"][0], raw["distances"][0])
        ]
        print(f"  [2/vault_search]  top hit → "
              f"{hits[0]['title']} (dist {hits[0]['distance']})" if hits else "  no hits")
        return {"vault_hits": hits}

    # ── Step 3: LLM quality check ─────────────────────────────────────────────

    def _quality_check(self, state: ResearchSession) -> dict:
        hits_text = "\n".join(
            f"• {h['title']} ({h['platform']}, {h['year']}) — {h['description']}"
            for h in (state["vault_hits"] or [])
        )
        msgs = [
            SystemMessage(content=(
                "You are a retrieval quality judge. Decide whether the documents "
                "below are sufficient to answer the question accurately and completely."
            )),
            UserMessage(content=(
                f"Question: {state['question']}\n\n"
                f"Retrieved:\n{hits_text}"
            )),
        ]
        ai_msg = self._llm.invoke(msgs, response_format=QualityCheck)
        result = JsonOutputParser().parse(ai_msg)
        verdict = "✓ PASS" if result.get("passed") else "✗ FAIL"
        print(f"  [3/quality_check] {verdict}  score={result.get('score')}/10  "
              f"→ {result.get('reasoning','')[:70]}")
        return {"quality": result}

    # ── Routing: skip web if vault passed ────────────────────────────────────

    def _route(self, state: ResearchSession):
        if state.get("quality", {}).get("passed"):
            print("  [route] vault sufficient → skip web search")
            return self._synth_step
        print("  [route] vault insufficient → invoking web search")
        return self._web_step

    # ── Step 4 (conditional): Tavily web search ───────────────────────────────

    def _web_lookup(self, state: ResearchSession) -> dict:
        print(f"  [4/web_lookup]    querying Tavily …")
        res = self._tavily.search(
            query=state["question"],
            search_depth="advanced",
            include_answer=True,
            include_raw_content=False,
        )
        web = {
            "answer":   res.get("answer", ""),
            "urls":     [r["url"]                 for r in res.get("results", [])[:4]],
            "snippets": [r.get("content","")[:200] for r in res.get("results", [])[:4]],
        }
        print(f"  [4/web_lookup]    {len(web['urls'])} sources returned")
        return {"web_data": web}

    # ── Step 5: synthesise the final report ───────────────────────────────────

    def _synthesise(self, state: ResearchSession) -> dict:
        vault_block = "\n".join(
            f"• {h['title']} | {h['platform']} | {h['year']} | {h['publisher']}\n"
            f"  {h['description']}"
            for h in (state["vault_hits"] or [])
        )
        web_block = ""
        if state.get("web_data"):
            wd = state["web_data"]
            web_block = "\nWeb Results:\n" + wd.get("answer", "") + "\n" + "\n".join(
                f"  [{i+1}] {u}" for i, u in enumerate(wd["urls"])
            )
        memory_block = (
            f"\nPrior session context:\n{state['prior_context']}"
            if state.get("prior_context") else ""
        )
        quality_note = state.get("quality", {}).get("reasoning", "")

        msgs = [
            SystemMessage(content=self.PERSONA),
            UserMessage(content=(
                f"Answer the question below. Cite every fact.\n\n"
                f"Question: {state['question']}\n\n"
                f"Vault data:\n{vault_block}"
                f"{web_block}{memory_block}\n\n"
                f"Vault quality assessment: {quality_note}"
            )),
        ]
        ai_msg  = self._llm.invoke(msgs, response_format=ResearchFindings)
        report  = JsonOutputParser().parse(ai_msg)
        print(f"  [5/synthesise]    reliability={report.get('reliability')}  "
              f"web_assisted={report.get('web_assisted')}")
        return {"findings": report}

    # ── Step 6: archive to long-term memory ──────────────────────────────────

    def _archive(self, state: ResearchSession) -> dict:
        ans = (state.get("findings") or {}).get("summary", "")
        if ans:
            self._memory.save(state["question"], ans)
            print("  [6/archive]       saved to session memory")
        return {}

    # ── Graph construction ────────────────────────────────────────────────────

    def _build_graph(self) -> StateMachine:
        m = StateMachine[ResearchSession](ResearchSession)

        entry        = EntryPoint[ResearchSession]()
        probe_step   = Step[ResearchSession]("memory_probe",   self._memory_probe)
        search_step  = Step[ResearchSession]("vault_search",   self._vault_search)
        quality_step = Step[ResearchSession]("quality_check",  self._quality_check)
        web_step     = Step[ResearchSession]("web_lookup",     self._web_lookup)
        synth_step   = Step[ResearchSession]("synthesise",     self._synthesise)
        archive_step = Step[ResearchSession]("archive",        self._archive)
        end          = Termination[ResearchSession]()

        # keep reference for _route closure
        self._web_step   = web_step
        self._synth_step = synth_step

        m.add_steps([entry, probe_step, search_step, quality_step,
                     web_step, synth_step, archive_step, end])

        m.connect(entry,        probe_step)
        m.connect(probe_step,   search_step)
        m.connect(search_step,  quality_step)
        m.connect(quality_step, [web_step, synth_step], self._route)
        m.connect(web_step,     synth_step)
        m.connect(synth_step,   archive_step)
        m.connect(archive_step, end)

        return m

    # ── Public API ────────────────────────────────────────────────────────────

    def ask(self, question: str, session_id: str = "default") -> Run:
        self._session.create_session(session_id)
        initial: ResearchSession = {
            "question":      question,
            "session_id":    session_id,
            "prior_context": None,
            "vault_hits":    None,
            "quality":       None,
            "web_data":      None,
            "findings":      None,
        }
        run = self._workflow.run(initial)
        self._session.add(run, session_id)
        return run

    def history(self, session_id: str = "default") -> List[Run]:
        return self._session.get_all_objects(session_id)


# ── Report display ────────────────────────────────────────────────────────────

def display(idx: int, question: str, run: Run) -> None:
    fs = run.get_final_state()
    f  = fs.get("findings") or {}
    q  = fs.get("quality")  or {}

    width = 70
    print(f"\n{'━'*width}")
    print(f"  #{idx}  {question}")
    print(f"{'━'*width}")
    print(f"\n  {f.get('summary', 'N/A')}\n")

    pts = f.get("data_points", [])
    if pts:
        print("  Key facts:")
        for pt in pts:
            print(f"    › {pt}")

    srcs = f.get("sources", [])
    if srcs:
        print("\n  Sources / citations:")
        for s in srcs:
            print(f"    ⎯ {s}")

    print(f"\n  Reliability : {f.get('reliability','?')}  |  "
          f"Web search : {'Yes' if f.get('web_assisted') else 'No'}  |  "
          f"Vault score : {q.get('score','?')}/10")
    print(f"  Quality note: {q.get('reasoning','')}")
    print(f"{'━'*width}")


# ── Demo queries ──────────────────────────────────────────────────────────────

QUERIES = [
    # Q1 — direct vault hit (Pokémon Gold/Silver is in the index)
    "What year was Pokémon Gold and Silver released and on which handheld?",

    # Q2 — vault will miss (God of War Ragnarök not indexed) → web fallback
    "Who is the developer and publisher of God of War Ragnarök?",

    # Q3 — multi-record vault query + benefits from prior memory
    "Give me a timeline of Mario platform games available in the UdaPlay catalogue.",
]


def main():
    print("\n" + "━"*70)
    print("  UdaPlay — Game Research Assistant")
    print("━"*70 + "\n")

    assistant = GameResearchAssistant()

    for i, q in enumerate(QUERIES, 1):
        print(f"\n{'─'*70}")
        print(f"  Running query {i}: {q}")
        print("─"*70)
        run = assistant.ask(q, session_id="demo")
        display(i, q, run)

    # ── Session summary table ─────────────────────────────────────────────────
    print(f"\n{'━'*70}")
    print("  Session Summary")
    print(f"{'━'*70}")
    for i, run in enumerate(assistant.history("demo"), 1):
        fs = run.get_final_state()
        f  = (fs.get("findings") or {})
        q  = (fs.get("quality")  or {})
        print(
            f"  [{i}] {'WEB' if f.get('web_assisted') else 'INT':3}  "
            f"score={q.get('score','?'):>2}/10  "
            f"rel={f.get('reliability','?'):6}  "
            f"{fs['question'][:55]}"
        )
    print()


if __name__ == "__main__":
    main()
