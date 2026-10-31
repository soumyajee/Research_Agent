import asyncio
import json

import aio_pika

from app.queue.consumer import consume
from app.queue.events import INGESTION_QUEUE
from app.services.document_service import ingest_url
from app.storage.postgres import init_db, update_job


async def handler(message: aio_pika.IncomingMessage):
    async with message.process():
        payload = json.loads(message.body.decode("utf-8"))
        job_id = payload.get("job_id")

        try:
            if job_id:
                await update_job(job_id, "running")

            result = await ingest_url(
                tenant_id=payload["tenant_id"],
                url=payload["url"],
                title=payload.get("title"),
            )

            if job_id:
                await update_job(job_id, "completed", result=result)
        except Exception as exc:
            if job_id:
                await update_job(job_id, "failed", error=str(exc))
            raise


async def main():
    await init_db()
    await consume(INGESTION_QUEUE, handler)
    print("Ingestion worker running")
    await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
