"""Smoke test of the LLM gateway: send one prompt through create_llm().

poetry run python scripts/check_llm_connection.py
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.core.llm import create_llm


def main() -> None:
    response = create_llm().invoke("Explain network slicing in one sentence.")
    print(response.content)


if __name__ == "__main__":
    main()
