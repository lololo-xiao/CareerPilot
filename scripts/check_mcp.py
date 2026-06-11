"""Scratch: verify the Phoenix MCP server is reachable from the Python runtime.

Run:  .venv/bin/python scripts/check_mcp.py
Reads PHOENIX_COLLECTOR_ENDPOINT (or PHOENIX_BASE_URL) and PHOENIX_API_KEY from .env.
Spawns `npx -y @arizeai/phoenix-mcp@latest` over stdio and lists its tools.
"""

from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main() -> None:
    load_dotenv()
    base_url = (
        os.getenv("PHOENIX_COLLECTOR_ENDPOINT")
        or os.getenv("PHOENIX_BASE_URL")
        or "https://app.phoenix.arize.com"
    ).strip().strip("'\"")
    api_key = os.getenv("PHOENIX_API_KEY", "").strip()

    print(f"baseUrl = {base_url}")
    print(f"apiKey  = {'set (' + str(len(api_key)) + ' chars)' if api_key else 'MISSING'}")

    params = StdioServerParameters(
        command="npx",
        args=[
            "-y",
            "@arizeai/phoenix-mcp@latest",
            "--baseUrl",
            base_url,
            "--apiKey",
            api_key,
        ],
        env=os.environ.copy(),
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            names = [t.name for t in tools.tools]
            print(f"\n{len(names)} tools exposed:")
            for n in names:
                print(f"  - {n}")

            # Try a low-risk read to confirm the key actually authenticates.
            if any(t.name == "list-projects" for t in tools.tools):
                print("\nCalling list-projects ...")
                res = await session.call_tool("list-projects", {})
                for c in res.content:
                    text = getattr(c, "text", None)
                    if text:
                        print(text[:1500])


if __name__ == "__main__":
    asyncio.run(main())
