import os
import tempfile

from app.ingestion.loader import load_file, load_url
from app.ingestion.pipeline import ingest_text


async def ingest_raw_text(
    tenant_id: str,
    title: str,
    text: str,
):
    return await ingest_text(
        tenant_id=tenant_id,
        title=title,
        text=text,
        source_type="text",
    )


async def ingest_url(
    tenant_id: str,
    url: str,
    title: str | None = None,
):
    content = await load_url(url)
    return await ingest_text(
        tenant_id=tenant_id,
        title=title or url,
        text=content,
        source_type="web",
        source_url=url,
    )


async def ingest_upload(
    tenant_id: str,
    filename: str,
    data: bytes,
):
    extension = os.path.splitext(filename)[1]

    with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as temp_file:
        temp_file.write(data)
        temp_path = temp_file.name

    try:
        text, source_type = load_file(temp_path)
        return await ingest_text(
            tenant_id=tenant_id,
            title=filename,
            text=text,
            source_type=source_type,
        )
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass
