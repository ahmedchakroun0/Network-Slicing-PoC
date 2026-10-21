"""Create a chat model routed through the LiteLLM Gateway."""

import os

from dotenv import load_dotenv
from langchain_litellm import ChatLiteLLM

load_dotenv()


def create_llm() -> ChatLiteLLM:
    """Create a chat model routed through the LiteLLM Gateway."""

    model_name = os.environ["MODEL_NAME"]

    return ChatLiteLLM(
        model=f"litellm_proxy/{model_name}",
        api_base=os.environ["LITELLM_API_BASE"],
        api_key=os.environ["LITELLM_API_KEY"],
        temperature=0,
    )
