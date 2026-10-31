import asyncio
import sys


# ============================================================
# WINDOWS EVENT LOOP
# ============================================================

if sys.platform == "win32":
    asyncio.set_event_loop_policy(
        asyncio.WindowsSelectorEventLoopPolicy()
    )


# ============================================================
# PRELOAD PYTORCH
# ============================================================
#
# Load PyTorch very early in the process so its Windows DLLs
# are initialized before the rest of the application imports.
# ============================================================

print("[BOOT] Loading PyTorch...")

import torch

print(
    "[BOOT] PyTorch ready:",
    torch.__version__,
)

print(
    "[BOOT] CUDA available:",
    torch.cuda.is_available(),
)


# ============================================================
# UVICORN
# ============================================================

import uvicorn


if __name__ == "__main__":

    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        loop="app.core.event_loop:selector_loop_factory",
        reload=False,
        workers=1,
        log_level="info",
    )