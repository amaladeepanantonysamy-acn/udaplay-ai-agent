"""
UdaPlay - Part 01: Game Knowledge Vault
========================================
Builds a persistent semantic search index from the local game catalogue.
Each game JSON is transformed into a rich embedding document that captures
platform, genre, publisher, year, and description together — so queries
like "open-world crime sandbox" surface the right titles naturally.

Author : Amala Deepan Antony Samy
"""

import importlib.util, sys

if importlib.util.find_spec("pysqlite3"):
    import pysqlite3
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import os, json
from dataclasses import dataclass
from typing import List
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

# ── Credentials ───────────────────────────────────────────────────────────────

load_dotenv("config.env") if os.path.exists("config.env") else load_dotenv()

assert os.getenv("OPENAI_API_KEY"),  "Set OPENAI_API_KEY in config.env"
assert os.getenv("TAVILY_API_KEY"),  "Set TAVILY_API_KEY in config.env"

OPENAI_KEY  = os.environ["OPENAI_API_KEY"]
TAVILY_KEY  = os.environ["TAVILY_API_KEY"]
OPENAI_BASE = os.getenv("OPENAI_BASE_URL", "https://openai.vocareum.com/v1")

print(f"[config] base_url = {OPENAI_BASE}")


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class GameRecord:
    """Typed wrapper for a single game entry from the JSON catalogue."""
    doc_id:    str
    name:      str
    platform:  str
    year:      int
    genre:     str
    publisher: str
    description: str

    @classmethod
    def from_file(cls, path: str) -> "GameRecord":
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        stem = os.path.splitext(os.path.basename(path))[0]
        return cls(
            doc_id      = stem,
            name        = raw["Name"],
            platform    = raw["Platform"],
            year        = int(raw["YearOfRelease"]),
            genre       = raw["Genre"],
            publisher   = raw["Publisher"],
            description = raw["Description"],
        )

    def to_embedding_text(self) -> str:
        """Produces a single string that captures all searchable dimensions."""
        return (
            f"Title: {self.name}\n"
            f"Platform: {self.platform} | Year: {self.year} | Genre: {self.genre}\n"
            f"Publisher: {self.publisher}\n"
            f"About: {self.description}"
        )

    def to_metadata(self) -> dict:
        return {
            "Name": self.name, "Platform": self.platform,
            "YearOfRelease": self.year, "Genre": self.genre,
            "Publisher": self.publisher, "Description": self.description,
        }


# ── Knowledge Vault ───────────────────────────────────────────────────────────

class GameKnowledgeVault:
    """
    Wraps ChromaDB to provide a clean API for building and querying
    the game semantic index.
    """

    VAULT_NAME = "udaplay"
    STORE_PATH = "chromadb"

    def __init__(self):
        self._client = chromadb.PersistentClient(path=self.STORE_PATH)
        self._embed  = embedding_functions.OpenAIEmbeddingFunction(
            api_key    = OPENAI_KEY,
            api_base   = OPENAI_BASE,
            model_name = "text-embedding-ada-002",
        )
        self._col = None

    def reset(self) -> "GameKnowledgeVault":
        """Drop any existing vault and start fresh."""
        try:
            self._client.delete_collection(self.VAULT_NAME)
        except Exception:
            pass
        self._col = self._client.create_collection(
            self.VAULT_NAME, embedding_function=self._embed
        )
        print(f"[vault] Created fresh collection '{self.VAULT_NAME}'")
        return self

    def load(self) -> "GameKnowledgeVault":
        """Open an existing vault (created by reset+index)."""
        self._col = self._client.get_collection(
            self.VAULT_NAME, embedding_function=self._embed
        )
        return self

    def index(self, games: List[GameRecord]) -> int:
        """Batch-add GameRecord objects to the vault."""
        assert self._col is not None, "Call reset() or load() first"
        self._col.add(
            ids       = [g.doc_id       for g in games],
            documents = [g.to_embedding_text() for g in games],
            metadatas = [g.to_metadata()       for g in games],
        )
        return len(games)

    def query(self, text: str, top_k: int = 3) -> List[dict]:
        """Return the top-k most semantically similar games."""
        assert self._col is not None, "Vault not initialised"
        raw = self._col.query(
            query_texts=[text],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for doc, meta, dist in zip(
            raw["documents"][0], raw["metadatas"][0], raw["distances"][0]
        ):
            hits.append({**meta, "distance": round(dist, 4), "content": doc})
        return hits

    @property
    def size(self) -> int:
        assert self._col is not None
        return self._col.count()


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_catalogue(folder: str) -> List[GameRecord]:
    games = []
    for fname in sorted(os.listdir(folder)):
        if fname.endswith(".json"):
            games.append(GameRecord.from_file(os.path.join(folder, fname)))
    return games


def print_search_results(query: str, hits: List[dict]) -> None:
    print(f"\n  Search: \"{query}\"")
    for i, h in enumerate(hits, 1):
        bar = "█" * max(1, int((1 - h["distance"]) * 20))
        print(f"  {i}. [{bar:<20}] {h['Name']} ({h['Platform']}, {h['YearOfRelease']})")
        print(f"       {h['Description'][:90]}...")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n━━━  UdaPlay Knowledge Vault Builder  ━━━\n")

    # 1. Load raw game files
    catalogue = load_catalogue("games")
    print(f"[loader] {len(catalogue)} game files found in games/")

    # Genre breakdown
    genres: dict = {}
    for g in catalogue:
        genres[g.genre] = genres.get(g.genre, 0) + 1
    print("[loader] Genre breakdown: " +
          ", ".join(f"{k}({v})" for k, v in sorted(genres.items())))

    # 2. Build the vector vault
    vault = GameKnowledgeVault().reset()
    added = vault.index(catalogue)
    print(f"[vault]  Indexed {added} games  —  vault size: {vault.size}")

    # 3. Smoke-test: semantic search across varied query types
    print("\n━━━  Semantic Search Smoke Tests  ━━━")
    test_queries = [
        "kart racing game on Nintendo",
        "crime sandbox open world",
        "pocket monster role-playing handheld",
        "military first person shooter futuristic",
        "physics sports motion controller party game",
    ]
    for q in test_queries:
        print_search_results(q, vault.query(q, top_k=2))

    print(f"\n✓  Vault ready at '{GameKnowledgeVault.STORE_PATH}/'  "
          f"({vault.size} documents)\n")


if __name__ == "__main__":
    main()
