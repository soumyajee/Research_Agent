# app/services/stt_service.py

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Any

import assemblyai as aai

from app.config import settings


# ============================================================
# API KEY
# ============================================================


def _get_api_key() -> str:
    """
    Resolve AssemblyAI API key from project settings or env.
    """

    candidates = [
        getattr(
            settings,
            "assemblyai_api_key",
            None,
        ),
        getattr(
            settings,
            "ASSEMBLYAI_API_KEY",
            None,
        ),
        os.getenv(
            "ASSEMBLYAI_API_KEY"
        ),
    ]

    for value in candidates:

        if value:

            return str(
                value
            ).strip()

    raise RuntimeError(
        "ASSEMBLYAI_API_KEY is not configured."
    )


# ============================================================
# CONFIGURE SDK
# ============================================================


def _configure_assemblyai() -> None:

    api_key = (
        _get_api_key()
    )

    aai.settings.api_key = (
        api_key
    )


# ============================================================
# LANGUAGE NORMALIZATION
# ============================================================


def _normalize_language(
    language: str | None,
) -> str | None:

    if not language:

        return None

    language = (
        str(
            language
        )
        .strip()
        .lower()
    )

    if not language:

        return None

    mapping = {
        "english":
            "en",

        "hindi":
            "hi",

        "tamil":
            "ta",

        "telugu":
            "te",
    }

    return (
        mapping.get(
            language,
            language,
        )
    )


# ============================================================
# AUDIO SUFFIX
# ============================================================


def _guess_suffix(
    filename: str | None,
    content_type: str | None = None,
) -> str:

    # --------------------------------------------------------
    # From filename
    # --------------------------------------------------------

    if filename:

        suffix = (
            Path(
                filename
            ).suffix
        )

        if suffix:

            return suffix

    # --------------------------------------------------------
    # From content type
    # --------------------------------------------------------

    content_type = (
        content_type
        or
        ""
    ).lower()

    mapping = {
        "audio/wav":
            ".wav",

        "audio/x-wav":
            ".wav",

        "audio/mpeg":
            ".mp3",

        "audio/mp3":
            ".mp3",

        "audio/mp4":
            ".m4a",

        "audio/x-m4a":
            ".m4a",

        "audio/ogg":
            ".ogg",

        "audio/webm":
            ".webm",
    }

    return (
        mapping.get(
            content_type,
            ".wav",
        )
    )


# ============================================================
# SYNC TRANSCRIPTION IMPLEMENTATION
# ============================================================


def _transcribe_sync(
    audio_bytes: bytes,
    filename: str,
    language: str | None,
) -> dict[str, Any]:
    """
    Execute AssemblyAI's normal prerecorded transcription API.

    Uses a temporary file because this works across more
    AssemblyAI SDK versions than relying on direct byte support.
    """

    if not audio_bytes:

        raise ValueError(
            "Audio data is empty."
        )

    _configure_assemblyai()

    language = (
        _normalize_language(
            language
        )
    )

    suffix = (
        _guess_suffix(
            filename
        )
    )

    temp_path: str | None = None

    try:

        # ====================================================
        # WRITE TEMP AUDIO
        # ====================================================

        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False,
        ) as temp_file:

            temp_file.write(
                audio_bytes
            )

            temp_file.flush()

            temp_path = (
                temp_file.name
            )

        # ====================================================
        # CONFIG
        # ====================================================

        config_kwargs: dict[
            str,
            Any,
        ] = {}

        if language:

            # Older SDK-compatible option.
            config_kwargs[
                "language_code"
            ] = language

        config = (
            aai.TranscriptionConfig(
                **config_kwargs
            )
        )

        # ====================================================
        # TRANSCRIBE
        # ====================================================

        transcriber = (
            aai.Transcriber(
                config=config
            )
        )

        transcript = (
            transcriber.transcribe(
                temp_path
            )
        )

        # ====================================================
        # ERROR STATUS
        # ====================================================

        transcript_status = (
            getattr(
                transcript,
                "status",
                None,
            )
        )

        transcript_error = (
            getattr(
                transcript,
                "error",
                None,
            )
        )

        # ----------------------------------------------------
        # Standard AssemblyAI error status check
        # ----------------------------------------------------

        error_status = getattr(
            aai.TranscriptStatus,
            "error",
            None,
        )

        if (
            error_status is not None
            and
            transcript_status
            ==
            error_status
        ):

            raise RuntimeError(
                (
                    "AssemblyAI transcription failed: "
                    f"{transcript_error}"
                )
            )

        # ----------------------------------------------------
        # Compatibility string check
        # ----------------------------------------------------

        if (
            str(
                transcript_status
            ).lower()
            in {
                "error",
                "transcriptstatus.error",
            }
        ):

            raise RuntimeError(
                (
                    "AssemblyAI transcription failed: "
                    f"{transcript_error}"
                )
            )

        # ====================================================
        # EXTRACT TEXT
        # ====================================================

        text = (
            getattr(
                transcript,
                "text",
                None,
            )
            or
            ""
        )

        text = (
            str(
                text
            ).strip()
        )

        if not text:

            raise RuntimeError(
                "AssemblyAI returned an empty transcript."
            )

        # ====================================================
        # RESULT
        # ====================================================

        return {
            "status":
                "success",

            "provider":
                "assemblyai",

            "text":
                text,

            "transcript":
                text,

            "transcript_id":
                getattr(
                    transcript,
                    "id",
                    None,
                ),

            "language":
                language,

            "audio_filename":
                filename,
        }

    finally:

        # ====================================================
        # REMOVE TEMP FILE
        # ====================================================

        if temp_path:

            try:

                os.remove(
                    temp_path
                )

            except OSError:

                pass


# ============================================================
# PUBLIC SERVICE CLASS
# ============================================================


class STTService:

    async def transcribe(
        self,
        audio_bytes: bytes,
        filename: str = "voice.wav",
        language: str | None = None,
    ) -> dict[str, Any]:
        """
        Async wrapper for FastAPI.

        AssemblyAI Transcriber is synchronous, so run it in
        a worker thread instead of blocking FastAPI's event loop.
        """

        return await asyncio.to_thread(
            _transcribe_sync,
            audio_bytes,
            filename,
            language,
        )


# ============================================================
# SERVICE INSTANCE
# ============================================================

stt_service = (
    STTService()
)


# ============================================================
# COMPATIBILITY FUNCTION
# ============================================================


async def transcribe_audio(
    audio_bytes: bytes,
    filename: str = "voice.wav",
    language: str | None = None,
) -> str:
    """
    Compatibility function for main.py.

    Returns transcript text because your current
    /audio/transcribe route expects a string.
    """

    result = (
        await stt_service.transcribe(
            audio_bytes=
                audio_bytes,

            filename=
                filename,

            language=
                language,
        )
    )

    return (
        result.get(
            "text"
        )
        or
        ""
    )