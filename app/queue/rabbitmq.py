import aio_pika

from app.config import settings


_connection = None
_channel = None


async def get_channel():
    global _connection
    global _channel

    if _connection is None or _connection.is_closed:
        _connection = await aio_pika.connect_robust(settings.rabbitmq_url)
        _channel = await _connection.channel()
        await _channel.set_qos(prefetch_count=5)

    return _channel


async def close_rabbitmq():
    global _connection
    global _channel

    if _channel is not None and not _channel.is_closed:
        await _channel.close()

    if _connection is not None and not _connection.is_closed:
        await _connection.close()

    _channel = None
    _connection = None
