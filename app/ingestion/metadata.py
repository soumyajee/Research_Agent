from datetime import datetime, timezone


def build_metadata(
    source_type: str,
    source_url: str | None = None,
    extra: dict | None = None,
):
    metadata = {
        "source_type": source_type,
        "source_url": source_url,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }

    if extra:
        metadata.update(extra)

    return metadata
