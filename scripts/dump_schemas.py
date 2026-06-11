"""Dump input schemas for the phoenix-mcp tools we plan to use."""

from __future__ import annotations

import asyncio
import json
import os

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

WANT = {
    "list-datasets",
    "get-dataset-examples",
    "add-dataset-examples",
    "get-spans",
    "get-span-annotations",
    "upsert-prompt",
    "add-prompt-version-tag",
    "get-project",
}


async def main() -> None:
    load_dotenv()
    base_url = (
        os.getenv("PHOENIX_COLLECTOR_ENDPOINT") or "https://app.phoenix.arize.com"
    ).strip().strip("'\"")
    api_key = os.getenv("PHOENIX_API_KEY", "").strip()
    params = StdioServerParameters(
        command="npx",
        args=["-y", "@arizeai/phoenix-mcp@latest", "--baseUrl", base_url, "--apiKey", api_key],
        env=os.environ.copy(),
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            for t in tools.tools:
                if t.name in WANT:
                    print(f"\n===== {t.name} =====")
                    print((t.description or "").strip()[:300])
                    print(json.dumps(t.inputSchema, indent=2)[:2000])


if __name__ == "__main__":
    asyncio.run(main())
