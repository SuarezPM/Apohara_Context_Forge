"""Entry point - starts ContextForge server and metrics collector."""
import asyncio
import logging
import sys

import uvicorn

from apohara_context_forge.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


async def main():
    """Start ContextForge server (requires the [serve] extra)."""
    from apohara_context_forge.mcp.server import app, metrics_loop

    logger.info("Starting ContextForge...")
    logger.info(f"Host: {settings.contextforge_host}:{settings.contextforge_port}")
    logger.info(f"vLLM: {settings.vllm_base_url}")
    logger.info(f"Model: {settings.vllm_model}")

    # Start background metrics collector
    metrics_task = asyncio.create_task(metrics_loop())
    
    try:
        config = uvicorn.Config(
            app,
            host=settings.contextforge_host,
            port=settings.contextforge_port,
            log_level="info",
        )
        server = uvicorn.Server(config)
        await server.serve()
    finally:
        metrics_task.cancel()


def run() -> None:
    """``python -m apohara_context_forge.main`` entry point.

    The server stack (mcp.server → compression → llmlingua) lives in the
    ``[serve]`` extra. Probe it up front and fail fast with an actionable
    message instead of surfacing a raw ModuleNotFoundError mid-startup.

    There is intentionally no ``apohara`` console script: a console script
    can't be gated behind an extra, so it would register unconditionally
    and break on a slim install. The slim package stays library + safety
    only; run the server with ``python -m apohara_context_forge.main``.
    """
    try:
        import apohara_context_forge.mcp.server  # noqa: F401 — probe [serve] deps
    except ModuleNotFoundError:
        sys.exit(
            "apohara server requires the [serve] extra: "
            "pip install apohara-context-forge[serve]"
        )
    asyncio.run(main())


if __name__ == "__main__":
    run()