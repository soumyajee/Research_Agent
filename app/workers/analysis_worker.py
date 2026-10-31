import asyncio
import json

import aio_pika

from app.agents.analysis import analysis_agent
from app.queue.consumer import consume
from app.queue.events import ANALYSIS_QUEUE
from app.storage.postgres import init_db, update_job


async def handler(message: aio_pika.IncomingMessage):
    async with message.process():
        payload = json.loads(message.body.decode("utf-8"))
        job_id = payload["job_id"]

        try:
            await update_job(job_id, "running")
            result = await analysis_agent(payload["state"])
            await update_job(job_id, "completed", result=result)
        except Exception as exc:
            await update_job(job_id, "failed", error=str(exc))
            raise


async def main():
    await init_db()
    await consume(ANALYSIS_QUEUE, handler)
    print("Analysis worker running")
    await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
