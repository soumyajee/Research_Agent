from app.queue.rabbitmq import get_channel


async def consume(queue_name: str, handler):
    channel = await get_channel()
    queue = await channel.declare_queue(queue_name, durable=True)
    await queue.consume(handler)
