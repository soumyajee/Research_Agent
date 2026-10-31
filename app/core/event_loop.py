import asyncio
import sys


def selector_loop_factory() -> asyncio.AbstractEventLoop:
    """
    Event-loop factory used by Uvicorn.

    Psycopg AsyncConnection is not compatible with the default
    Windows ProactorEventLoop.

    On Windows:
        use SelectorEventLoop.

    On Linux/macOS:
        use the normal asyncio event loop.
    """

    if sys.platform == "win32":

        loop = asyncio.SelectorEventLoop()

        return loop

    return asyncio.new_event_loop()