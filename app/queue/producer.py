import json

import aio_pika

from app.queue.rabbitmq import get_channel


async def publish_job(queue_name: str, payload: dict):
    channel = await get_channel()
    queue = await channel.declare_queue(queue_name, durable=True)

    message = aio_pika.Message(
        body=json.dumps(payload).encode("utf-8"),
        delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
    )

    await channel.default_exchange.publish(
        message,
        routing_key=queue.name,
    )
