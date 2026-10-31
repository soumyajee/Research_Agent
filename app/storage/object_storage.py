import asyncio
import uuid
from pathlib import Path
from typing import Any

from app.config import settings


# ============================================================
# STORAGE ROOT
# ============================================================

STORAGE_ROOT = Path(
    settings.object_storage_path
).resolve()


# ============================================================
# INITIALIZE OBJECT STORAGE
# ============================================================

async def init_object_storage() -> None:
    """
    Create the root storage directory if it does not exist.

    Called during FastAPI startup.
    """

    await asyncio.to_thread(
        STORAGE_ROOT.mkdir,
        parents=True,
        exist_ok=True,
    )

    print(
        "[OBJECT STORAGE] Ready:",
        str(STORAGE_ROOT),
    )


# ============================================================
# SAFE FILENAME
# ============================================================

def _safe_filename(
    filename: str,
) -> str:
    """
    Remove directory traversal information from filenames.

    Example:

        ../../secret/file.pdf

    becomes:

        file.pdf
    """

    filename = (
        filename
        or
        "file"
    )

    safe_name = Path(
        filename
    ).name

    if not safe_name:
        safe_name = "file"

    return safe_name


# ============================================================
# SAFE FOLDER
# ============================================================

def _safe_folder(
    folder: str,
) -> Path:
    """
    Resolve a folder safely inside STORAGE_ROOT.

    Prevents paths such as:

        ../../outside
    """

    folder = (
        folder
        or
        "uploads"
    )

    clean_parts = []

    for part in Path(folder).parts:

        if part in {
            "",
            ".",
            "..",
            "/",
            "\\",
        }:

            continue

        clean_parts.append(
            part
        )


    destination = STORAGE_ROOT

    for part in clean_parts:

        destination = (
            destination
            /
            part
        )


    resolved = destination.resolve()


    # --------------------------------------------------------
    # Ensure resolved folder remains inside STORAGE_ROOT
    # --------------------------------------------------------

    try:

        resolved.relative_to(
            STORAGE_ROOT
        )

    except ValueError as exc:

        raise ValueError(
            "Invalid storage folder."
        ) from exc


    return resolved


# ============================================================
# SAFE EXISTING PATH
# ============================================================

def _safe_existing_path(
    path: str,
) -> Path:
    """
    Validate that a path belongs to our object-storage root.
    """

    file_path = Path(
        path
    )


    if not file_path.is_absolute():

        file_path = (
            STORAGE_ROOT
            /
            file_path
        )


    resolved = file_path.resolve()


    try:

        resolved.relative_to(
            STORAGE_ROOT
        )

    except ValueError as exc:

        raise ValueError(
            "Requested path is outside "
            "the object storage directory."
        ) from exc


    return resolved


# ============================================================
# SAVE FILE
# ============================================================

async def save_file(
    data: bytes,
    filename: str,
    folder: str = "uploads",
    metadata: dict[str, Any] | None = None,
) -> dict:
    """
    Save binary data into local object storage.

    Returns metadata describing the stored object.
    """

    if not data:

        raise ValueError(
            "Cannot save an empty file."
        )


    safe_filename = (
        _safe_filename(
            filename
        )
    )


    destination_folder = (
        _safe_folder(
            folder
        )
    )


    await asyncio.to_thread(
        destination_folder.mkdir,
        parents=True,
        exist_ok=True,
    )


    file_id = str(
        uuid.uuid4()
    )


    stored_filename = (
        f"{file_id}_{safe_filename}"
    )


    file_path = (
        destination_folder
        /
        stored_filename
    )


    await asyncio.to_thread(
        file_path.write_bytes,
        data,
    )


    relative_path = (
        file_path
        .relative_to(
            STORAGE_ROOT
        )
    )


    return {

        "file_id":
            file_id,

        "filename":
            safe_filename,

        "stored_filename":
            stored_filename,

        "path":
            str(file_path),

        "relative_path":
            str(relative_path),

        "folder":
            str(
                relative_path.parent
            ),

        "size":
            len(data),

        "metadata":
            metadata
            or {},
    }


# ============================================================
# READ FILE
# ============================================================

async def read_file(
    path: str,
) -> bytes:
    """
    Read a file from object storage.
    """

    file_path = (
        _safe_existing_path(
            path
        )
    )


    exists = await asyncio.to_thread(
        file_path.exists
    )


    if not exists:

        raise FileNotFoundError(
            f"Stored file not found: {path}"
        )


    is_file = await asyncio.to_thread(
        file_path.is_file
    )


    if not is_file:

        raise ValueError(
            f"Path is not a file: {path}"
        )


    return await asyncio.to_thread(
        file_path.read_bytes
    )


# ============================================================
# DELETE FILE
# ============================================================

async def delete_file(
    path: str,
) -> bool:
    """
    Delete a stored file.

    Returns:
        True  -> deleted
        False -> file did not exist
    """

    file_path = (
        _safe_existing_path(
            path
        )
    )


    exists = await asyncio.to_thread(
        file_path.exists
    )


    if not exists:

        return False


    is_file = await asyncio.to_thread(
        file_path.is_file
    )


    if not is_file:

        raise ValueError(
            f"Path is not a file: {path}"
        )


    await asyncio.to_thread(
        file_path.unlink
    )


    return True


# ============================================================
# FILE EXISTS
# ============================================================

async def file_exists(
    path: str,
) -> bool:
    """
    Check whether a stored file exists.
    """

    file_path = (
        _safe_existing_path(
            path
        )
    )


    return await asyncio.to_thread(
        file_path.is_file
    )


# ============================================================
# FILE INFO
# ============================================================

async def get_file_info(
    path: str,
) -> dict | None:
    """
    Return metadata about one stored file.
    """

    file_path = (
        _safe_existing_path(
            path
        )
    )


    exists = await asyncio.to_thread(
        file_path.exists
    )


    if not exists:

        return None


    if not await asyncio.to_thread(
        file_path.is_file
    ):

        return None


    stat = await asyncio.to_thread(
        file_path.stat
    )


    relative_path = (
        file_path.relative_to(
            STORAGE_ROOT
        )
    )


    return {

        "filename":
            file_path.name,

        "path":
            str(file_path),

        "relative_path":
            str(relative_path),

        "size":
            stat.st_size,

        "exists":
            True,
    }


# ============================================================
# LIST FILES
# ============================================================

async def list_files(
    folder: str = "",
) -> list[dict]:
    """
    List files stored inside a folder.
    """

    target_folder = (
        STORAGE_ROOT
        if not folder
        else _safe_folder(folder)
    )


    exists = await asyncio.to_thread(
        target_folder.exists
    )


    if not exists:

        return []


    def _list():

        results = []

        for file_path in (
            target_folder.rglob("*")
        ):

            if not file_path.is_file():
                continue


            stat = file_path.stat()


            relative_path = (
                file_path.relative_to(
                    STORAGE_ROOT
                )
            )


            results.append(
                {
                    "filename":
                        file_path.name,

                    "path":
                        str(file_path),

                    "relative_path":
                        str(relative_path),

                    "size":
                        stat.st_size,
                }
            )


        return results


    return await asyncio.to_thread(
        _list
    )


# ============================================================
# SAVE UPLOADED DOCUMENT
# ============================================================

async def save_upload(
    filename: str,
    data: bytes,
    tenant_id: str = "default",
) -> dict:
    """
    Save an uploaded source document.

    Structure:

        data/
          uploads/
            <tenant_id>/
    """

    folder = (
        f"uploads/"
        f"{tenant_id}"
    )


    return await save_file(
        data=data,
        filename=filename,
        folder=folder,
        metadata={
            "type":
                "upload",

            "tenant_id":
                tenant_id,
        },
    )


# ============================================================
# SAVE URL CONTENT
# ============================================================

async def save_url_content(
    tenant_id: str,
    filename: str,
    data: bytes,
    source_url: str,
) -> dict:
    """
    Store raw content downloaded from a URL.
    """

    folder = (
        f"url_sources/"
        f"{tenant_id}"
    )


    return await save_file(
        data=data,
        filename=filename,
        folder=folder,
        metadata={
            "type":
                "url_source",

            "tenant_id":
                tenant_id,

            "source_url":
                source_url,
        },
    )


# ============================================================
# SAVE RESEARCH ARTIFACT
# ============================================================

async def save_research_artifact(
    session_id: str,
    filename: str,
    data: bytes,
) -> dict:
    """
    Store files produced by the research pipeline.

    Example:
        JSON result
        markdown report
        evidence export
    """

    folder = (
        f"research/"
        f"{session_id}"
    )


    return await save_file(
        data=data,
        filename=filename,
        folder=folder,
        metadata={
            "type":
                "research_artifact",

            "session_id":
                session_id,
        },
    )


# ============================================================
# SAVE REPORT
# ============================================================

async def save_report(
    session_id: str,
    filename: str,
    data: bytes,
) -> dict:
    """
    Store generated research reports.
    """

    folder = (
        f"reports/"
        f"{session_id}"
    )


    return await save_file(
        data=data,
        filename=filename,
        folder=folder,
        metadata={
            "type":
                "research_report",

            "session_id":
                session_id,
        },
    )


# ============================================================
# SAVE TEMPORARY FILE
# ============================================================

async def save_temp_file(
    filename: str,
    data: bytes,
) -> dict:
    """
    Save temporary processing data.

    Temporary cleanup can be added later if needed.
    """

    return await save_file(
        data=data,
        filename=filename,
        folder="temp",
        metadata={
            "type":
                "temporary",
        },
    )