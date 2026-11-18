#!/usr/bin/env python3
"""Call vLLM directly to verify its OpenAI-compatible Chat Completions API."""

import asyncio
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()


async def main() -> None:
    client = AsyncOpenAI(
        base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
        api_key=os.getenv("VLLM_API_KEY", "local-vllm-key"),
        timeout=120.0,
    )
    model = os.getenv("SERVED_MODEL_NAME", "qwen2.5-1.5b")
    try:
        result = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "你好，请介绍一下 vLLM 的作用。"}],
            max_tokens=128,
            temperature=0.2,
        )
        print(result.choices[0].message.content)
        print("\nUsage:", result.usage.model_dump() if result.usage else None)
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
