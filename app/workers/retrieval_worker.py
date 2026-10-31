import asyncio
import json

import aio_pika

from app.queue.consumer import consume
from app.queue.events import RETRIEVAL_QUEUE
from app.retrieval.hybrid import hybrid_search
from app.storage.postgres import init_db, update_job


async def handler(message: aio_pika.IncomingMessage):
    async with message.process():
        payload = json.loads(message.body.decode("utf-8"))
        job_id = payload["job_id"]

        try:
            await update_job(job_id, "running")

            results = await hybrid_search(
                tenant_id=payload["tenant_id"],
                query=payload["query"],
            )

            await update_job(
                job_id,
                "completed",
                result={"results": results},
            )
        except Exception as exc:
            await update_job(job_id, "failed", error=str(exc))
            raise


async def main():
    await init_db()
    await consume(RETRIEVAL_QUEUE, handler)
    print("Retrieval worker running")
    await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
