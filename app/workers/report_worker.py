import asyncio
import json

import aio_pika

from app.queue.consumer import consume
from app.queue.events import REPORT_QUEUE
from app.storage.object_storage import save_text
from app.storage.postgres import init_db, update_job


async def handler(message: aio_pika.IncomingMessage):
    async with message.process():
        payload = json.loads(message.body.decode("utf-8"))
        job_id = payload["job_id"]

        try:
            await update_job(job_id, "running")

            session_id = payload["session_id"]
            content = payload["content"]

            path = await save_text(
                filename=f"reports/{session_id}.md",
                content=content,
            )

            await update_job(job_id, "completed", result={"path": path})
        except Exception as exc:
            await update_job(job_id, "failed", error=str(exc))
            raise


async def main():
    await init_db()
    await consume(REPORT_QUEUE, handler)
    print("Report worker running")
    await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
