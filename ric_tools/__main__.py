"""Run the MCP server of one RIC role (streamable HTTP, 127.0.0.1).

    poetry run python -m ric_tools --role near_rt --port 8201
    poetry run python -m ric_tools --role non_rt  --port 8202

LiteLLM exposes them to the agents (`mcp_servers:` in litellm_config.yaml).
"""

import argparse
import asyncio

from ran_emulator.client import EmulatorClient
from ric_tools.server import ROLE_OF_SERVER, build_server


async def _serve(role: str, host: str, port: int, emulator_url: str) -> None:
    async with EmulatorClient(emulator_url) as client:
        server = build_server(ROLE_OF_SERVER[role], client)
        await server.run_streamable_http_async(host=host, port=port)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--role", required=True, choices=sorted(ROLE_OF_SERVER))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--emulator-url", default="http://127.0.0.1:8100")
    args = parser.parse_args()
    asyncio.run(_serve(args.role, args.host, args.port, args.emulator_url))


if __name__ == "__main__":
    main()
