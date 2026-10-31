from app.ingestion.chunker import hybrid_chunk
from app.ingestion.cleaner import clean_text
from app.ingestion.embeddings import embed_texts
from app.ingestion.metadata import build_metadata
from app.storage.postgres import (
    create_document,
    find_document_by_url,
    insert_chunk,
)


async def ingest_text(
    tenant_id: str,
    title: str,
    text: str,
    source_type: str,
    source_url: str | None = None,
    extra_metadata: dict | None = None,
):
    if source_url:
        existing = await find_document_by_url(tenant_id, source_url)
        if existing:
            return {
                "document_id": str(existing["id"]),
                "chunks": 0,
                "skipped": True,
            }

    cleaned_text = clean_text(text)
    if not cleaned_text:
        raise ValueError("Document contains no text")

    metadata = build_metadata(
        source_type=source_type,
        source_url=source_url,
        extra=extra_metadata,
    )

    document_id = await create_document(
        tenant_id=tenant_id,
        title=title,
        source_type=source_type,
        source_url=source_url,
        metadata=metadata,
    )

    chunks = await hybrid_chunk(cleaned_text, source_type)
    if not chunks:
        return {"document_id": document_id, "chunks": 0, "skipped": False}

    embeddings = await embed_texts(chunks)

    for index, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        await insert_chunk(
            document_id=document_id,
            tenant_id=tenant_id,
            content=chunk,
            embedding=embedding,
            metadata={**metadata, "chunk_index": index},
        )

    return {
        "document_id": document_id,
        "chunks": len(chunks),
        "skipped": False,
    }
