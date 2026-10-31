# app/services/tts_service.py

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid

from pathlib import Path
from typing import Any, Literal

import pyttsx3


# ============================================================
# CONFIGURATION
# ============================================================

AUDIO_OUTPUT_DIR = Path(
    os.getenv(
        "AUDIO_OUTPUT_DIR",
        "generated_audio",
    )
).resolve()

AUDIO_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


DEFAULT_RATE = int(
    os.getenv(
        "TTS_RATE",
        "175",
    )
)

DEFAULT_VOLUME = float(
    os.getenv(
        "TTS_VOLUME",
        "1.0",
    )
)

DEFAULT_MP3_BITRATE = os.getenv(
    "TTS_MP3_BITRATE",
    "192k",
)


# ============================================================
# PYTTSX3 LOCK
#
# pyttsx3 / Windows SAPI should not be driven concurrently
# by multiple threads.
# ============================================================

_TTS_LOCK = threading.Lock()


# ============================================================
# BASIC HELPERS
# ============================================================


def _clean_text(
    text: str | None,
) -> str:

    value = (
        text
        or
        ""
    ).strip()

    if not value:

        raise ValueError(
            "TTS text cannot be empty."
        )

    return value


def _normalize_output_format(
    output_format: str | None,
) -> Literal[
    "wav",
    "mp3",
    "both",
]:

    value = (
        output_format
        or
        "both"
    ).strip().lower()

    if value not in {
        "wav",
        "mp3",
        "both",
    }:

        raise ValueError(
            (
                "output_format must be "
                "'wav', 'mp3', or 'both'."
            )
        )

    return value  # type: ignore[return-value]


def _normalize_volume(
    volume: float,
) -> float:

    return max(
        0.0,
        min(
            1.0,
            float(
                volume
            ),
        ),
    )


def _normalize_rate(
    rate: int,
) -> int:

    return max(
        80,
        min(
            300,
            int(
                rate
            ),
        ),
    )


# ============================================================
# OUTPUT PATH
# ============================================================


def _new_audio_id() -> str:

    return str(
        uuid.uuid4()
    )


def _new_wav_path(
    audio_id: str,
) -> Path:

    return (
        AUDIO_OUTPUT_DIR
        /
        f"{audio_id}.wav"
    )


def _new_mp3_path(
    audio_id: str,
) -> Path:

    return (
        AUDIO_OUTPUT_DIR
        /
        f"{audio_id}.mp3"
    )


# ============================================================
# PUBLIC AUDIO PATH RESOLVER
#
# Used by:
#
# GET /audio/output/{filename}
# ============================================================


def get_audio_path(
    filename: str,
) -> Path:

    safe_filename = (
        Path(
            filename
        ).name
    )

    if not safe_filename:

        raise ValueError(
            "Audio filename is empty."
        )

    path = (
        AUDIO_OUTPUT_DIR
        /
        safe_filename
    ).resolve()

    if (
        path.parent
        !=
        AUDIO_OUTPUT_DIR
    ):

        raise ValueError(
            "Invalid audio filename."
        )

    return path


# ============================================================
# FFMPEG DISCOVERY
# ============================================================


def _find_ffmpeg() -> Path | None:
    """
    Find ffmpeg without importing pydub.

    Search order:

    1. FFMPEG_BINARY environment variable
    2. System PATH
    3. Active Conda environment
    4. sys.prefix/Library/bin on Windows
    """

    # ========================================================
    # ENV OVERRIDE
    # ========================================================

    configured = (
        os.getenv(
            "FFMPEG_BINARY"
        )
        or
        ""
    ).strip()

    if configured:

        configured_path = Path(
            configured
        ).expanduser()

        if configured_path.exists():

            return configured_path.resolve()

    # ========================================================
    # PATH
    # ========================================================

    discovered = shutil.which(
        "ffmpeg"
    )

    if discovered:

        return Path(
            discovered
        ).resolve()

    # ========================================================
    # CONDA_PREFIX
    # ========================================================

    conda_prefix = (
        os.getenv(
            "CONDA_PREFIX"
        )
        or
        ""
    ).strip()

    if conda_prefix:

        if sys.platform == "win32":

            candidate = (
                Path(
                    conda_prefix
                )
                /
                "Library"
                /
                "bin"
                /
                "ffmpeg.exe"
            )

        else:

            candidate = (
                Path(
                    conda_prefix
                )
                /
                "bin"
                /
                "ffmpeg"
            )

        if candidate.exists():

            return candidate.resolve()

    # ========================================================
    # PYTHON ENV PREFIX
    # ========================================================

    if sys.platform == "win32":

        candidate = (
            Path(
                sys.prefix
            )
            /
            "Library"
            /
            "bin"
            /
            "ffmpeg.exe"
        )

    else:

        candidate = (
            Path(
                sys.prefix
            )
            /
            "bin"
            /
            "ffmpeg"
        )

    if candidate.exists():

        return candidate.resolve()

    return None


# ============================================================
# FFMPEG STATUS
# ============================================================


def ffmpeg_status() -> dict[str, Any]:

    ffmpeg = (
        _find_ffmpeg()
    )

    if ffmpeg is None:

        return {
            "available":
                False,

            "path":
                None,

            "message":
                (
                    "FFmpeg was not found. "
                    "WAV generation is available, "
                    "but MP3 conversion requires FFmpeg."
                ),
        }

    return {
        "available":
            True,

        "path":
            str(
                ffmpeg
            ),

        "message":
            "FFmpeg is available.",
    }


# ============================================================
# VOICE DISCOVERY
# ============================================================


def _find_voice_id(
    engine,
    requested_voice: str | None,
) -> str | None:

    if not requested_voice:

        return None

    requested = (
        requested_voice
        .strip()
        .lower()
    )

    voices = (
        engine.getProperty(
            "voices"
        )
        or
        []
    )

    # ========================================================
    # EXACT ID
    # ========================================================

    for voice in voices:

        voice_id = str(
            getattr(
                voice,
                "id",
                "",
            )
        )

        if (
            voice_id.lower()
            ==
            requested
        ):

            return voice_id

    # ========================================================
    # NAME SEARCH
    # ========================================================

    for voice in voices:

        voice_id = str(
            getattr(
                voice,
                "id",
                "",
            )
        )

        voice_name = str(
            getattr(
                voice,
                "name",
                "",
            )
        )

        if (
            requested
            in voice_name.lower()
        ):

            return voice_id

    return None


# ============================================================
# PYTTSX3 WAV GENERATION
# ============================================================


def _generate_wav_sync(
    *,
    text: str,
    wav_path: Path,
    voice_id: str | None,
    rate: int,
    volume: float,
) -> None:

    with _TTS_LOCK:

        engine = None

        try:

            engine = (
                pyttsx3.init()
            )

            # =================================================
            # RATE
            # =================================================

            engine.setProperty(
                "rate",
                _normalize_rate(
                    rate
                ),
            )

            # =================================================
            # VOLUME
            # =================================================

            engine.setProperty(
                "volume",
                _normalize_volume(
                    volume
                ),
            )

            # =================================================
            # VOICE
            # =================================================

            selected_voice = (
                _find_voice_id(
                    engine,
                    voice_id,
                )
            )

            if selected_voice:

                engine.setProperty(
                    "voice",
                    selected_voice,
                )

            # =================================================
            # CLEAN OLD FILE
            # =================================================

            if wav_path.exists():

                wav_path.unlink()

            # =================================================
            # GENERATE
            # =================================================

            engine.save_to_file(
                text,
                str(
                    wav_path
                ),
            )

            engine.runAndWait()

        finally:

            if engine is not None:

                try:

                    engine.stop()

                except Exception:
                    pass

    # ========================================================
    # WINDOWS SAPI SOMETIMES RETURNS JUST BEFORE THE FILE IS
    # FULLY VISIBLE. WAIT VERY BRIEFLY.
    # ========================================================

    for _ in range(
        20
    ):

        if (
            wav_path.exists()
            and
            wav_path.stat().st_size > 44
        ):

            break

        time.sleep(
            0.05
        )

    if not wav_path.exists():

        raise RuntimeError(
            (
                "pyttsx3 completed but no "
                "WAV file was created."
            )
        )

    if wav_path.stat().st_size <= 44:

        raise RuntimeError(
            (
                "pyttsx3 created an invalid "
                "or empty WAV file."
            )
        )


# ============================================================
# WAV -> MP3
# ============================================================


def _convert_wav_to_mp3_sync(
    *,
    wav_path: Path,
    mp3_path: Path,
) -> None:

    ffmpeg = (
        _find_ffmpeg()
    )

    if ffmpeg is None:

        raise RuntimeError(
            (
                "FFmpeg is not installed or could "
                "not be found. Install it with:\n"
                "conda install -c conda-forge ffmpeg"
            )
        )

    if not wav_path.exists():

        raise FileNotFoundError(
            (
                "Input WAV file was not found: "
                f"{wav_path}"
            )
        )

    if mp3_path.exists():

        mp3_path.unlink()

    command = [
        str(
            ffmpeg
        ),

        "-y",

        "-hide_banner",

        "-loglevel",
        "error",

        "-i",
        str(
            wav_path
        ),

        "-codec:a",
        "libmp3lame",

        "-b:a",
        DEFAULT_MP3_BITRATE,

        str(
            mp3_path
        ),
    ]

    process = subprocess.run(
        command,

        stdout=
            subprocess.PIPE,

        stderr=
            subprocess.PIPE,

        text=
            True,

        check=
            False,
    )

    if process.returncode != 0:

        error_message = (
            process.stderr
            or
            process.stdout
            or
            "Unknown FFmpeg error."
        )

        raise RuntimeError(
            (
                "FFmpeg MP3 conversion failed: "
                f"{error_message.strip()}"
            )
        )

    if not mp3_path.exists():

        raise RuntimeError(
            (
                "FFmpeg completed but MP3 "
                "file was not created."
            )
        )

    if mp3_path.stat().st_size <= 0:

        raise RuntimeError(
            "Generated MP3 file is empty."
        )


# ============================================================
# RESPONSE BUILDERS
# ============================================================


def _wav_response(
    path: Path,
) -> dict[str, Any]:

    return {
        "filename":
            path.name,

        "path":
            str(
                path
            ),

        "url":
            (
                f"/audio/output/"
                f"{path.name}"
            ),

        "media_type":
            "audio/wav",

        "size_bytes":
            path.stat().st_size,
    }


def _mp3_response(
    path: Path,
) -> dict[str, Any]:

    return {
        "filename":
            path.name,

        "path":
            str(
                path
            ),

        "url":
            (
                f"/audio/output/"
                f"{path.name}"
            ),

        "media_type":
            "audio/mpeg",

        "size_bytes":
            path.stat().st_size,
    }


# ============================================================
# MAIN SYNTHESIS FUNCTION
# ============================================================


async def synthesize_speech(
    text: str,

    voice_id: str | None = None,

    voice: str | None = None,

    rate: int = DEFAULT_RATE,

    volume: float = DEFAULT_VOLUME,

    output_format: Literal[
        "wav",
        "mp3",
        "both",
    ] = "both",

    **_: Any,
) -> dict[str, Any]:
    """
    Convert text to WAV and/or MP3.

    Main public entry point used by main.py.
    """

    cleaned_text = (
        _clean_text(
            text
        )
    )

    output_format = (
        _normalize_output_format(
            output_format
        )
    )

    selected_voice = (
        voice_id
        or
        voice
    )

    rate = (
        _normalize_rate(
            rate
        )
    )

    volume = (
        _normalize_volume(
            volume
        )
    )

    audio_id = (
        _new_audio_id()
    )

    wav_path = (
        _new_wav_path(
            audio_id
        )
    )

    mp3_path = (
        _new_mp3_path(
            audio_id
        )
    )

    # ========================================================
    # TEXT -> WAV
    #
    # WAV is always generated first because pyttsx3 outputs WAV.
    # ========================================================

    await asyncio.to_thread(
        _generate_wav_sync,

        text=
            cleaned_text,

        wav_path=
            wav_path,

        voice_id=
            selected_voice,

        rate=
            rate,

        volume=
            volume,
    )

    wav_payload = (
        _wav_response(
            wav_path
        )
    )

    mp3_payload: (
        dict[str, Any]
        |
        None
    ) = None

    mp3_error: str | None = None

    # ========================================================
    # WAV -> MP3
    # ========================================================

    if output_format in {
        "mp3",
        "both",
    }:

        try:

            await asyncio.to_thread(
                _convert_wav_to_mp3_sync,

                wav_path=
                    wav_path,

                mp3_path=
                    mp3_path,
            )

            mp3_payload = (
                _mp3_response(
                    mp3_path
                )
            )

        except Exception as exc:

            mp3_error = str(
                exc
            )

            # If the caller specifically asked for only MP3,
            # MP3 failure means the whole operation failed.
            if output_format == "mp3":

                raise RuntimeError(
                    mp3_error
                ) from exc

    # ========================================================
    # RESPONSE
    # ========================================================

    response = {
        "status":
            "success",

        "generated":
            True,

        "engine":
            "pyttsx3",

        "audio_id":
            audio_id,

        "output_format":
            output_format,

        "text_length":
            len(
                cleaned_text
            ),

        "voice_id":
            selected_voice,

        "rate":
            rate,

        "volume":
            volume,

        "wav":
            (
                wav_payload

                if output_format
                in {
                    "wav",
                    "both",
                }

                else
                None
            ),

        "mp3":
            mp3_payload,

        "wav_url":
            (
                wav_payload[
                    "url"
                ]

                if output_format
                in {
                    "wav",
                    "both",
                }

                else
                None
            ),

        "mp3_url":
            (
                mp3_payload[
                    "url"
                ]

                if mp3_payload
                else
                None
            ),

        "mp3_error":
            mp3_error,

        "ffmpeg":
            ffmpeg_status(),
    }

    # ========================================================
    # MP3-ONLY:
    # The intermediate WAV is not needed after conversion.
    # ========================================================

    if (
        output_format == "mp3"
        and
        mp3_payload is not None
    ):

        try:

            wav_path.unlink(
                missing_ok=True
            )

        except Exception:
            pass

        response[
            "wav"
        ] = None

        response[
            "wav_url"
        ] = None

    return response


# ============================================================
# TRANSCRIPT -> AUDIO
#
# Despite the historical name, this supports wav/mp3/both.
# ============================================================


async def transcript_to_mp3(
    text: str,

    voice_id: str | None = None,

    voice: str | None = None,

    rate: int = DEFAULT_RATE,

    volume: float = DEFAULT_VOLUME,

    output_format: Literal[
        "wav",
        "mp3",
        "both",
    ] = "both",

    **kwargs: Any,
) -> dict[str, Any]:

    return await synthesize_speech(
        text=
            text,

        voice_id=
            voice_id,

        voice=
            voice,

        rate=
            rate,

        volume=
            volume,

        output_format=
            output_format,

        **kwargs,
    )


# ============================================================
# COMPATIBILITY ALIASES
#
# Your main.py searches several possible function names.
# Keep all of them working.
# ============================================================


async def synthesize_text(
    text: str,
    **kwargs: Any,
) -> dict[str, Any]:

    return await synthesize_speech(
        text=
            text,

        **kwargs,
    )


async def synthesize_text_to_speech(
    text: str,
    **kwargs: Any,
) -> dict[str, Any]:

    return await synthesize_speech(
        text=
            text,

        **kwargs,
    )


async def generate_tts(
    text: str,
    **kwargs: Any,
) -> dict[str, Any]:

    return await synthesize_speech(
        text=
            text,

        **kwargs,
    )


async def text_to_speech(
    text: str,
    **kwargs: Any,
) -> dict[str, Any]:

    return await synthesize_speech(
        text=
            text,

        **kwargs,
    )


async def generate_audio(
    text: str,
    **kwargs: Any,
) -> dict[str, Any]:

    return await synthesize_speech(
        text=
            text,

        **kwargs,
    )


# ============================================================
# LIST VOICES
# ============================================================


def _list_voices_sync() -> list[
    dict[str, Any]
]:

    with _TTS_LOCK:

        engine = None

        try:

            engine = (
                pyttsx3.init()
            )

            voices = (
                engine.getProperty(
                    "voices"
                )
                or
                []
            )

            result: list[
                dict[str, Any]
            ] = []

            for voice in voices:

                languages: list[str] = []

                raw_languages = (
                    getattr(
                        voice,
                        "languages",
                        [],
                    )
                    or
                    []
                )

                for language in raw_languages:

                    if isinstance(
                        language,
                        bytes,
                    ):

                        try:

                            decoded = (
                                language.decode(
                                    "utf-8",
                                    errors="ignore",
                                )
                            )

                        except Exception:

                            decoded = str(
                                language
                            )

                        languages.append(
                            decoded
                        )

                    else:

                        languages.append(
                            str(
                                language
                            )
                        )

                result.append(
                    {
                        "id":
                            str(
                                getattr(
                                    voice,
                                    "id",
                                    "",
                                )
                            ),

                        "name":
                            str(
                                getattr(
                                    voice,
                                    "name",
                                    "",
                                )
                            ),

                        "languages":
                            languages,

                        "gender":
                            str(
                                getattr(
                                    voice,
                                    "gender",
                                    "",
                                )
                                or
                                ""
                            ),

                        "age":
                            str(
                                getattr(
                                    voice,
                                    "age",
                                    "",
                                )
                                or
                                ""
                            ),
                    }
                )

            return result

        finally:

            if engine is not None:

                try:

                    engine.stop()

                except Exception:
                    pass


async def list_voices() -> list[
    dict[str, Any]
]:

    return await asyncio.to_thread(
        _list_voices_sync
    )


async def get_voices() -> list[
    dict[str, Any]
]:

    return await list_voices()


async def available_voices() -> list[
    dict[str, Any]
]:

    return await list_voices()


# ============================================================
# DELETE GENERATED AUDIO
# ============================================================


def delete_audio(
    filename: str,
) -> bool:

    path = (
        get_audio_path(
            filename
        )
    )

    if not path.exists():

        return False

    path.unlink()

    return True


# ============================================================
# CLEANUP OLD FILES
# ============================================================


def cleanup_generated_audio(
    max_age_seconds: int = 86400,
) -> dict[str, Any]:
    """
    Delete generated WAV/MP3 files older than the supplied TTL.

    Default:
        86400 seconds = 24 hours
    """

    now = time.time()

    deleted = 0
    failed = 0

    for path in (
        AUDIO_OUTPUT_DIR.iterdir()
    ):

        if not path.is_file():

            continue

        if path.suffix.lower() not in {
            ".wav",
            ".mp3",
        }:

            continue

        try:

            age = (
                now
                -
                path.stat().st_mtime
            )

            if age >= max_age_seconds:

                path.unlink()

                deleted += 1

        except Exception:

            failed += 1

    return {
        "status":
            "success",

        "deleted":
            deleted,

        "failed":
            failed,

        "directory":
            str(
                AUDIO_OUTPUT_DIR
            ),
    }


# ============================================================
# MODULE STATUS
# ============================================================


def tts_status() -> dict[str, Any]:

    return {
        "status":
            "ok",

        "engine":
            "pyttsx3",

        "output_directory":
            str(
                AUDIO_OUTPUT_DIR
            ),

        "wav_supported":
            True,

        "mp3_supported":
            bool(
                _find_ffmpeg()
            ),

        "ffmpeg":
            ffmpeg_status(),
    }


# ============================================================
# DO NOT CREATE:
#
# tts_service = ...
#
# main.py imports this WHOLE MODULE using:
#
#     from app.services import tts_service
#
# ============================================================