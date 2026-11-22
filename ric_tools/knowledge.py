"""Read-only knowledge search of the RIC agents.

It does no actuation, so it stays a local LangChain tool: it is neither
served by the MCP servers nor gated by governance.
"""

from pathlib import Path

from langchain_core.tools import tool

# Resolved from the project root so the tool works from any working directory.
# The runner can point it at another file (--knowledge-file).
KNOWLEDGE_FILE = Path(__file__).resolve().parent.parent / "data" / "ran_knowledge.txt"


@tool
async def search_ran_knowledge(query: str) -> str:
    """
    Search the 6G RAN knowledge base for information relevant
    to the current RIC problem.
    """
    text = KNOWLEDGE_FILE.read_text(encoding="utf-8")
    query_words = {word.lower() for word in query.split() if len(word) > 2}
    chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]

    results = []
    for chunk in chunks:
        chunk_words = {word.lower().strip(".,:;()[]{}") for word in chunk.split()}
        score = len(query_words & chunk_words)
        if score > 0:
            results.append((score, chunk))

    results.sort(reverse=True)
    if not results:
        return "No relevant RAN knowledge was found."
    return "\n\n".join(chunk for _, chunk in results[:3])
