# app/main.py

from __future__ import annotations

import asyncio
import inspect
import json
import math
import sys

from contextlib import asynccontextmanager
from typing import Any, Literal


# ============================================================
# FASTAPI
# ============================================================

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
)

from fastapi.encoders import jsonable_encoder

from fastapi.middleware.cors import (
    CORSMiddleware,
)

from fastapi.responses import (
    FileResponse,
    JSONResponse,
    Response,
)


# ============================================================
# GROQ
# ============================================================

from groq import RateLimitError


# ============================================================
# PYDANTIC
# ============================================================

from pydantic import (
    BaseModel,
    Field,
)


# ============================================================
# CONFIG
# ============================================================

from app.config import settings


# ============================================================
# DOCUMENT SERVICES
# ============================================================

from app.services.document_service import (
    ingest_raw_text,
    ingest_upload,
    ingest_url,
)


# ============================================================
# LLM SERVICE
# ============================================================

from app.services.llm_service import (
    LLMRateLimitError,
    close_llm_client,
    convert_groq_rate_limit,
)


# ============================================================
# RESEARCH SERVICE
#
# IMPORTANT:
# research_service.py now handles:
#
# normal LangGraph execution
# +
# LLM rate-limit retrieval fallback
#
# main.py does NOT implement fallback logic.
# ============================================================

from app.services.research_service import (
    run_research,
)


# ============================================================
# STT SERVICE
#
# voice_service.py REMOVED
# ============================================================

import app.services.stt_service as stt_service


# ============================================================
# TTS SERVICE
# ============================================================

import app.services.tts_service as tts_service


# ============================================================
# RETRIEVAL
#
# Only used for debug endpoint.
# The actual rate-limit fallback belongs to research_service.
# ============================================================

from app.retrieval.context_builder import (
    build_context,
)

from app.retrieval.hybrid import (
    hybrid_search_debug,
)


# ============================================================
# OBJECT STORAGE
# ============================================================

from app.storage.object_storage import (
    init_object_storage,
)


# ============================================================
# POSTGRES
# ============================================================

from app.storage.postgres import (
    close_db,
    complete_review,
    get_document_chunks,
    get_job,
    get_pool,
    get_session,
    init_db,
    list_documents,
    save_session_state,
)


# ============================================================
# REDIS
# ============================================================

from app.storage.redis import (
    close_redis,
    init_redis,
    redis_health,
)


# ============================================================
# REQUEST MODELS
# ============================================================


class ResearchRequest(BaseModel):

    tenant_id: str = "default"

    query: str = Field(
        ...,
        min_length=1,
    )


class TextDocumentRequest(BaseModel):

    tenant_id: str = "default"

    title: str = Field(
        ...,
        min_length=1,
    )

    text: str = Field(
        ...,
        min_length=1,
    )


class URLDocumentRequest(BaseModel):

    tenant_id: str = "default"

    url: str = Field(
        ...,
        min_length=1,
    )

    async_mode: bool = False


class ReviewRequest(BaseModel):

    action: Literal[
        "approve",
        "reject",
        "research_more",
    ]

    feedback: str | None = None


class TTSRequest(BaseModel):

    text: str = Field(
        ...,
        min_length=1,
    )

    voice_id: str | None = None

    rate: int = Field(
        default=175,
        ge=80,
        le=300,
    )

    volume: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
    )

    output_format: Literal[
        "wav",
        "mp3",
        "both",
    ] = "both"


class TranscriptAudioRequest(BaseModel):

    text: str = Field(
        ...,
        min_length=1,
    )

    voice_id: str | None = None

    rate: int = Field(
        default=175,
        ge=80,
        le=300,
    )

    volume: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
    )

    output_format: Literal[
        "wav",
        "mp3",
        "both",
    ] = "both"


# ============================================================
# MAYBE AWAIT
# ============================================================


async def _maybe_await(
    value: Any,
) -> Any:

    if inspect.isawaitable(
        value
    ):

        return await value

    return value


# ============================================================
# CALL ONLY SUPPORTED ARGUMENTS
# ============================================================


async def _call_supported(
    function,
    **kwargs,
) -> Any:

    signature = inspect.signature(
        function
    )

    accepts_kwargs = any(

        parameter.kind
        ==
        inspect.Parameter.VAR_KEYWORD

        for parameter
        in signature.parameters.values()
    )

    if accepts_kwargs:

        selected_kwargs = kwargs

    else:

        selected_kwargs = {

            key:
                value

            for key, value
            in kwargs.items()

            if key
            in signature.parameters
        }

    result = function(
        **selected_kwargs
    )

    return await _maybe_await(
        result
    )


# ============================================================
# JSON SAFE
# ============================================================


def _json_safe(
    value: Any,
) -> Any:

    try:

        return jsonable_encoder(
            value
        )

    except Exception:

        try:

            return json.loads(
                json.dumps(
                    value,
                    default=str,
                )
            )

        except Exception:

            return str(
                value
            )


# ============================================================
# NORMALIZE TENANT
# ============================================================


def _normalize_tenant_id(
    tenant_id: str | None,
) -> str:

    value = (
        tenant_id
        or
        ""
    ).strip()

    return (
        value
        or
        "default"
    )


# ============================================================
# NORMALIZE QUERY
# ============================================================


def _normalize_query(
    query: str | None,
) -> str:

    value = (
        query
        or
        ""
    ).strip()

    if not value:

        raise HTTPException(
            status_code=400,
            detail=(
                "Research query cannot be empty."
            ),
        )

    return value


# ============================================================
# EXTRACT STORED SESSION STATE
# ============================================================


def _extract_session_state(
    session: Any,
) -> dict[str, Any]:

    if session is None:

        return {}

    if not isinstance(
        session,
        dict,
    ):

        try:

            session = dict(
                session
            )

        except Exception:

            return {}

    # ========================================================
    # state
    # ========================================================

    state = session.get(
        "state"
    )

    if isinstance(
        state,
        dict,
    ):

        return dict(
            state
        )

    if isinstance(
        state,
        str,
    ):

        try:

            parsed = json.loads(
                state
            )

            if isinstance(
                parsed,
                dict,
            ):

                return parsed

        except Exception:
            pass

    # ========================================================
    # state_json
    # ========================================================

    state_json = session.get(
        "state_json"
    )

    if isinstance(
        state_json,
        dict,
    ):

        return dict(
            state_json
        )

    if isinstance(
        state_json,
        str,
    ):

        try:

            parsed = json.loads(
                state_json
            )

            if isinstance(
                parsed,
                dict,
            ):

                return parsed

        except Exception:
            pass

    return dict(
        session
    )


# ============================================================
# RESEARCH RESPONSE
#
# Handles BOTH:
#
# 1. normal full-agentic response
# 2. retrieval-only fallback response
# ============================================================


def _research_response(
    session_id: str,
    state: dict[str, Any],
    review_id: str | None = None,
) -> dict[str, Any]:

    payload = dict(
        state
    )

    # ========================================================
    # ANSWER
    # ========================================================

    answer = (
        payload.get(
            "final_answer"
        )
        or
        payload.get(
            "generated_answer"
        )
        or
        payload.get(
            "answer"
        )
        or
        ""
    )

    payload[
        "session_id"
    ] = str(
        session_id
    )

    payload[
        "answer"
    ] = answer

    payload[
        "final_answer"
    ] = answer

    # ========================================================
    # REVIEW
    # ========================================================

    if review_id is not None:

        payload[
            "review_id"
        ] = str(
            review_id
        )

    # ========================================================
    # RESPONSE MODE
    # ========================================================

    payload[
        "response_mode"
    ] = (
        payload.get(
            "response_mode"
        )
        or
        "full_agentic"
    )

    payload[
        "evaluation_mode"
    ] = payload.get(
        "evaluation_mode"
    )

    payload[
        "degraded_mode"
    ] = bool(
        payload.get(
            "degraded_mode",
            False,
        )
    )

    payload[
        "llm_generated"
    ] = bool(
        payload.get(
            "llm_generated",
            False,
        )
    )

    # ========================================================
    # EVALUATION COMPLETION
    # ========================================================

    payload[
        "claim_extraction_completed"
    ] = bool(
        payload.get(
            "claim_extraction_completed",
            False,
        )
    )

    payload[
        "nli_evaluation_completed"
    ] = bool(
        payload.get(
            "nli_evaluation_completed",
            False,
        )
    )

    # ========================================================
    # SCORES
    # ========================================================

    payload[
        "scores"
    ] = {

        "confidence":
            payload.get(
                "confidence_score"
            ),

        "grounding":
            payload.get(
                "grounding_score"
            ),

        "faithfulness":
            payload.get(
                "faithfulness_score"
            ),

        "hallucination":
            payload.get(
                "hallucination_score"
            ),

        "citation":
            payload.get(
                "citation_score"
            ),

        "evidence":
            payload.get(
                "evidence_score"
            ),

        "retrieval_quality":
            payload.get(
                "retrieval_quality"
            ),
    }

    # ========================================================
    # RETRIEVAL FALLBACK INFORMATION
    # ========================================================

    payload[
        "retrieved_context"
    ] = (
        payload.get(
            "retrieved_context"
        )
        or
        []
    )

    payload[
        "rate_limit"
    ] = payload.get(
        "rate_limit"
    )

    payload[
        "next_action"
    ] = payload.get(
        "next_action"
    )

    return _json_safe(
        payload
    )


# ============================================================
# STT ADAPTER
# ============================================================


async def _transcribe_audio(
    *,
    audio_bytes: bytes,
    upload_file: UploadFile,
    filename: str,
    language: str | None = None,
    session_id: str | None = None,
) -> str:

    if not audio_bytes:

        raise ValueError(
            "Audio input is empty."
        )

    function = getattr(
        stt_service,
        "transcribe_audio",
        None,
    )

    if not callable(
        function
    ):

        function = getattr(
            stt_service,
            "transcribe_upload",
            None,
        )

    if not callable(
        function
    ):

        function = getattr(
            stt_service,
            "transcribe_file",
            None,
        )

    if not callable(
        function
    ):

        raise RuntimeError(
            (
                "No compatible STT function was found "
                "in app/services/stt_service.py."
            )
        )

    try:

        await upload_file.seek(
            0
        )

    except Exception:
        pass

    result = await _call_supported(

        function,

        audio_bytes=
            audio_bytes,

        audio_file=
            upload_file,

        file=
            upload_file,

        upload=
            upload_file,

        filename=
            filename,

        language=
            language,

        session_id=
            session_id,
    )

    # ========================================================
    # STRING
    # ========================================================

    if isinstance(
        result,
        str,
    ):

        transcript = result

    # ========================================================
    # DICTIONARY
    # ========================================================

    elif isinstance(
        result,
        dict,
    ):

        transcript = (
            result.get(
                "text"
            )
            or
            result.get(
                "transcript"
            )
            or
            result.get(
                "content"
            )
            or
            ""
        )

    else:

        transcript = str(
            result
            or
            ""
        )

    transcript = (
        transcript.strip()
    )

    if not transcript:

        raise RuntimeError(
            (
                "STT completed but returned "
                "an empty transcript."
            )
        )

    return transcript


# ============================================================
# TTS FUNCTION RESOLVER
# ============================================================


def _get_tts_function(
    names: tuple[str, ...],
):

    for name in names:

        function = getattr(
            tts_service,
            name,
            None,
        )

        if callable(
            function
        ):

            return function

    return None


# ============================================================
# TTS GENERATE
# ============================================================


async def _synthesize_audio(
    *,
    text: str,
    voice_id: str | None = None,
    rate: int = 175,
    volume: float = 1.0,
    output_format: str = "both",
) -> Any:

    text = (
        text
        or
        ""
    ).strip()

    if not text:

        return {

            "status":
                "skipped",

            "generated":
                False,

            "reason":
                "empty_text",

            "wav_url":
                None,

            "mp3_url":
                None,
        }

    function = _get_tts_function(
        (
            "synthesize_speech",
            "synthesize_text",
            "synthesize_text_to_speech",
            "generate_tts",
            "text_to_speech",
            "generate_audio",
        )
    )

    if function is None:

        raise RuntimeError(
            (
                "No compatible TTS generation function "
                "was found in app/services/tts_service.py."
            )
        )

    result = await _call_supported(

        function,

        text=
            text,

        voice_id=
            voice_id,

        voice=
            voice_id,

        rate=
            rate,

        volume=
            volume,

        output_format=
            output_format,
    )

    return result


# ============================================================
# TTS VOICES
# ============================================================


async def _tts_voices() -> Any:

    function = _get_tts_function(
        (
            "list_voices",
            "get_voices",
            "available_voices",
        )
    )

    if function is None:

        return []

    return await _call_supported(
        function
    )


# ============================================================
# DATABASE HEALTH
# ============================================================


async def _database_health() -> dict[str, Any]:

    try:

        pool = await _maybe_await(
            get_pool()
        )

        if pool is None:

            return {
                "status":
                    "error",

                "error":
                    "Database pool is not initialized.",
            }

        # ====================================================
        # PSYCOPG ASYNC CONNECTION POOL
        # ====================================================

        connection_function = getattr(
            pool,
            "connection",
            None,
        )

        if callable(
            connection_function
        ):

            async with connection_function() as connection:

                result = connection.execute(
                    "SELECT 1"
                )

                await _maybe_await(
                    result
                )

            return {
                "status":
                    "ok"
            }

        # ====================================================
        # ASYNCPG-STYLE POOL
        # ====================================================

        acquire_function = getattr(
            pool,
            "acquire",
            None,
        )

        if callable(
            acquire_function
        ):

            async with acquire_function() as connection:

                execute_function = getattr(
                    connection,
                    "execute",
                )

                result = execute_function(
                    "SELECT 1"
                )

                await _maybe_await(
                    result
                )

            return {
                "status":
                    "ok"
            }

        return {

            "status":
                "unknown",

            "message":
                "Database pool does not expose a known health-check API.",
        }

    except Exception as exc:

        return {

            "status":
                "error",

            "error":
                str(
                    exc
                ),
        }


# ============================================================
# OPTIONAL SERVICE CLOSE
# ============================================================


async def _close_optional_service(
    service: Any,
    names: tuple[str, ...],
) -> None:

    for name in names:

        function = getattr(
            service,
            name,
            None,
        )

        if not callable(
            function
        ):

            continue

        try:

            await _maybe_await(
                function()
            )

        except Exception as exc:

            print(
                f"[SHUTDOWN] {name} warning:",
                repr(
                    exc
                ),
            )

        return


# ============================================================
# LIFESPAN
# ============================================================


@asynccontextmanager
async def lifespan(
    app: FastAPI,
):

    loop = (
        asyncio.get_running_loop()
    )

    print()
    print(
        "============================================"
    )
    print(
        "[STARTUP] Agentic Research RAG"
    )
    print(
        "============================================"
    )

    print(
        "[STARTUP] Event loop:",
        type(
            loop
        ).__name__,
    )

    # ========================================================
    # WINDOWS EVENT LOOP
    # ========================================================

    if (
        sys.platform == "win32"
        and
        "Proactor"
        in type(
            loop
        ).__name__
    ):

        raise RuntimeError(
            (
                "Windows ProactorEventLoop detected. "
                "Start the application using python run_server.py"
            )
        )

    # ========================================================
    # POSTGRES
    # ========================================================

    print(
        "[STARTUP] Initializing PostgreSQL..."
    )

    await _maybe_await(
        init_db()
    )

    print(
        "[STARTUP] PostgreSQL ready."
    )

    # ========================================================
    # REDIS
    # ========================================================

    print(
        "[STARTUP] Initializing Redis..."
    )

    await _maybe_await(
        init_redis()
    )

    print(
        "[STARTUP] Redis ready."
    )

    # ========================================================
    # OBJECT STORAGE
    # ========================================================

    print(
        "[STARTUP] Initializing object storage..."
    )

    await _maybe_await(
        init_object_storage()
    )

    print(
        "[STARTUP] Object storage ready."
    )

    # ========================================================
    # STT
    # ========================================================

    print(
        "[STARTUP] STT module:",
        getattr(
            stt_service,
            "__file__",
            "app.services.stt_service",
        ),
    )

    # ========================================================
    # TTS
    # ========================================================

    try:

        tts_status_function = getattr(
            tts_service,
            "tts_status",
            None,
        )

        if callable(
            tts_status_function
        ):

            tts_status = (
                tts_status_function()
            )

            print(
                "[STARTUP] TTS:",
                tts_status,
            )

    except Exception as exc:

        print(
            "[STARTUP] TTS warning:",
            repr(
                exc
            ),
        )

    print(
        "[STARTUP] API ready."
    )

    print(
        "============================================"
    )

    try:

        yield

    finally:

        print()
        print(
            "============================================"
        )
        print(
            "[SHUTDOWN] Closing services..."
        )
        print(
            "============================================"
        )

        # ====================================================
        # LLM
        # ====================================================

        try:

            await _maybe_await(
                close_llm_client()
            )

        except Exception as exc:

            print(
                "[SHUTDOWN] LLM warning:",
                repr(
                    exc
                ),
            )

        # ====================================================
        # STT
        # ====================================================

        await _close_optional_service(
            stt_service,
            (
                "close_stt_client",
                "close_client",
                "close",
            ),
        )

        # ====================================================
        # TTS
        # ====================================================

        await _close_optional_service(
            tts_service,
            (
                "close_tts_client",
                "close_client",
                "close",
            ),
        )

        # ====================================================
        # REDIS
        # ====================================================

        try:

            await _maybe_await(
                close_redis()
            )

        except Exception as exc:

            print(
                "[SHUTDOWN] Redis warning:",
                repr(
                    exc
                ),
            )

        # ====================================================
        # POSTGRES
        # ====================================================

        try:

            await _maybe_await(
                close_db()
            )

        except Exception as exc:

            print(
                "[SHUTDOWN] PostgreSQL warning:",
                repr(
                    exc
                ),
            )

        print(
            "[SHUTDOWN] Complete."
        )


# ============================================================
# FASTAPI
# ============================================================


app = FastAPI(

    title=(
        "Agentic Research RAG API"
    ),

    description=(
        "Planner -> Discovery -> Retrieval -> Analysis -> "
        "Synthesis -> Claim Extraction -> JEV/NLI -> "
        "Score Generation -> Confidence -> Final/Human Review. "
        "If the LLM is rate-limited, research_service switches "
        "to retrieval-only fallback mode."
    ),

    version="1.3.0",

    lifespan=lifespan,
)


# ============================================================
# CORS
# ============================================================


app.add_middleware(

    CORSMiddleware,

    allow_origins=[
        "*"
    ],

    allow_credentials=False,

    allow_methods=[
        "*"
    ],

    allow_headers=[
        "*"
    ],
)


# ============================================================
# CUSTOM LLM RATE-LIMIT HANDLER
#
# research_service normally catches research LLM rate limits.
# This remains as global protection for unexpected calls.
# ============================================================


@app.exception_handler(
    LLMRateLimitError
)
async def llm_rate_limit_handler(
    request: Request,
    exc: LLMRateLimitError,
):

    headers: dict[
        str,
        str,
    ] = {}

    retry_after = getattr(
        exc,
        "retry_after_seconds",
        None,
    )

    if retry_after is not None:

        headers[
            "Retry-After"
        ] = str(
            max(
                1,
                math.ceil(
                    retry_after
                ),
            )
        )

    return JSONResponse(

        status_code=429,

        headers=headers,

        content={

            "status":
                "rate_limited",

            "provider":
                "groq",

            "quota_type":
                getattr(
                    exc,
                    "quota_type",
                    None,
                ),

            "message":
                getattr(
                    exc,
                    "message",
                    str(
                        exc
                    ),
                ),

            "retry_after_seconds":
                retry_after,
        },
    )


# ============================================================
# RAW GROQ RATE-LIMIT HANDLER
# ============================================================


@app.exception_handler(
    RateLimitError
)
async def raw_groq_rate_limit_handler(
    request: Request,
    exc: RateLimitError,
):

    try:

        converted = (
            convert_groq_rate_limit(
                exc
            )
        )

    except Exception:

        return JSONResponse(
            status_code=429,
            content={
                "status":
                    "rate_limited",

                "provider":
                    "groq",

                "message":
                    str(
                        exc
                    ),
            },
        )

    headers: dict[
        str,
        str,
    ] = {}

    retry_after = getattr(
        converted,
        "retry_after_seconds",
        None,
    )

    if retry_after is not None:

        headers[
            "Retry-After"
        ] = str(
            max(
                1,
                math.ceil(
                    retry_after
                ),
            )
        )

    return JSONResponse(

        status_code=429,

        headers=headers,

        content={

            "status":
                "rate_limited",

            "provider":
                "groq",

            "quota_type":
                getattr(
                    converted,
                    "quota_type",
                    None,
                ),

            "message":
                getattr(
                    converted,
                    "message",
                    str(
                        exc
                    ),
                ),

            "retry_after_seconds":
                retry_after,
        },
    )


# ============================================================
# ROOT
# ============================================================


@app.get("/")
async def root():

    return {

        "name":
            "Agentic Research RAG",

        "status":
            "running",

        "docs":
            "/docs",

        "health":
            "/health",

        "research":
            "/research",

        "voice_research":
            "/voice/research",

        "audio_status":
            "/audio/status",
    }


# ============================================================
# HEALTH
# ============================================================


@app.get(
    "/health"
)
async def health():

    database_status = (
        await _database_health()
    )

    try:

        redis_status = (
            await _maybe_await(
                redis_health()
            )
        )

    except Exception as exc:

        redis_status = {

            "status":
                "error",

            "error":
                str(
                    exc
                ),
        }

    overall_status = (
        "ok"
        if database_status.get(
            "status"
        )
        ==
        "ok"
        else
        "degraded"
    )

    return {

        "status":
            overall_status,

        "database":
            database_status,

        "redis":
            redis_status,

        "llm":
            getattr(
                settings,
                "groq_model",
                None,
            ),

        "embedding":
            getattr(
                settings,
                "embedding_model",
                None,
            ),

        "embedding_dimension":
            getattr(
                settings,
                "embedding_dimension",
                None,
            ),

        "reranker":
            getattr(
                settings,
                "reranker_model",
                None,
            ),

        "nli":
            getattr(
                settings,
                "nli_model",
                None,
            ),
    }


# ============================================================
# AUDIO STATUS
# ============================================================


@app.get(
    "/audio/status"
)
async def audio_status():

    stt_function = (
        getattr(
            stt_service,
            "transcribe_audio",
            None,
        )
        or
        getattr(
            stt_service,
            "transcribe_upload",
            None,
        )
        or
        getattr(
            stt_service,
            "transcribe_file",
            None,
        )
    )

    stt_status = {

        "status":
            (
                "ok"
                if callable(
                    stt_function
                )
                else
                "error"
            ),

        "service":
            "stt_service",

        "module":
            str(
                getattr(
                    stt_service,
                    "__file__",
                    "",
                )
            ),
    }

    try:

        status_function = getattr(
            tts_service,
            "tts_status",
            None,
        )

        if callable(
            status_function
        ):

            tts_status = (
                status_function()
            )

        else:

            tts_status = {

                "status":
                    (
                        "ok"
                        if _get_tts_function(
                            (
                                "synthesize_speech",
                                "generate_tts",
                                "text_to_speech",
                            )
                        )
                        else
                        "error"
                    ),

                "service":
                    "tts_service",
            }

    except Exception as exc:

        tts_status = {

            "status":
                "error",

            "error":
                str(
                    exc
                ),
        }

    return {

        "status":
            "ok",

        "stt":
            stt_status,

        "tts":
            tts_status,
    }


# ============================================================
# STT
#
# AUDIO -> TRANSCRIPT
# ============================================================


@app.post(
    "/audio/transcribe"
)
async def audio_transcribe(

    file: UploadFile = File(...),

    language: str | None = Form(
        default=None
    ),
):

    audio_bytes = (
        await file.read()
    )

    if not audio_bytes:

        raise HTTPException(
            status_code=400,
            detail=(
                "Uploaded audio file is empty."
            ),
        )

    try:

        transcript = (
            await _transcribe_audio(

                audio_bytes=
                    audio_bytes,

                upload_file=
                    file,

                filename=
                    file.filename
                    or
                    "voice.wav",

                language=
                    language,
            )
        )

    except RateLimitError:
        raise

    except HTTPException:
        raise

    except Exception as exc:

        raise HTTPException(

            status_code=500,

            detail=(
                "STT transcription failed: "
                f"{exc}"
            ),
        ) from exc

    return {

        "status":
            "success",

        "text":
            transcript,

        "transcript":
            transcript,

        "filename":
            file.filename,

        "content_type":
            file.content_type,

        "language":
            language,
    }


# ============================================================
# TTS VOICES
# ============================================================


@app.get(
    "/audio/voices"
)
async def audio_voices():

    try:

        voices = (
            await _tts_voices()
        )

        if isinstance(
            voices,
            dict,
        ):

            return _json_safe(
                voices
            )

        return {

            "status":
                "success",

            "voices":
                _json_safe(
                    voices
                    or
                    []
                ),
        }

    except Exception as exc:

        raise HTTPException(

            status_code=500,

            detail=(
                "Unable to list TTS voices: "
                f"{exc}"
            ),
        ) from exc


# ============================================================
# TTS SYNTHESIS
# ============================================================


@app.post(
    "/audio/synthesize"
)
async def audio_synthesize(
    request: TTSRequest,
):

    try:

        result = (
            await _synthesize_audio(

                text=
                    request.text,

                voice_id=
                    request.voice_id,

                rate=
                    request.rate,

                volume=
                    request.volume,

                output_format=
                    request.output_format,
            )
        )

    except Exception as exc:

        raise HTTPException(

            status_code=500,

            detail=(
                "Speech synthesis failed: "
                f"{exc}"
            ),
        ) from exc

    # ========================================================
    # RAW AUDIO BYTES
    # ========================================================

    if isinstance(
        result,
        bytes,
    ):

        media_type = (
            "audio/mpeg"
            if request.output_format == "mp3"
            else
            "audio/wav"
        )

        return Response(
            content=
                result,

            media_type=
                media_type,
        )

    if isinstance(
        result,
        bytearray,
    ):

        media_type = (
            "audio/mpeg"
            if request.output_format == "mp3"
            else
            "audio/wav"
        )

        return Response(
            content=
                bytes(
                    result
                ),

            media_type=
                media_type,
        )

    return _json_safe(
        result
    )


# ============================================================
# TRANSCRIPT -> AUDIO
# ============================================================


@app.post(
    "/audio/transcript-to-audio"
)
async def transcript_to_audio(
    request: TranscriptAudioRequest,
):

    try:

        result = (
            await _synthesize_audio(

                text=
                    request.text,

                voice_id=
                    request.voice_id,

                rate=
                    request.rate,

                volume=
                    request.volume,

                output_format=
                    request.output_format,
            )
        )

        return _json_safe(
            result
        )

    except Exception as exc:

        raise HTTPException(

            status_code=500,

            detail=(
                "Transcript audio generation failed: "
                f"{exc}"
            ),
        ) from exc


# ============================================================
# GENERATED AUDIO FILE
# ============================================================


@app.get(
    "/audio/output/{filename}"
)
async def audio_output(
    filename: str,
):

    function = getattr(
        tts_service,
        "get_audio_path",
        None,
    )

    if not callable(
        function
    ):

        raise HTTPException(
            status_code=501,
            detail=(
                "tts_service.py does not provide "
                "get_audio_path()."
            ),
        )

    try:

        path = function(
            filename
        )

    except Exception as exc:

        raise HTTPException(
            status_code=400,
            detail=str(
                exc
            ),
        ) from exc

    if not path.exists():

        raise HTTPException(
            status_code=404,
            detail=(
                "Generated audio file was not found."
            ),
        )

    suffix = (
        path.suffix.lower()
    )

    if suffix == ".mp3":

        media_type = (
            "audio/mpeg"
        )

    elif suffix == ".wav":

        media_type = (
            "audio/wav"
        )

    else:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported audio format."
            ),
        )

    return FileResponse(

        path=str(
            path
        ),

        media_type=
            media_type,

        filename=
            path.name,
    )


# ============================================================
# VOICE RESEARCH
#
# Audio
#   ↓
# STT
#   ↓
# Transcript
#   ↓
# run_research()
#   ↓
#   ├─ Full Agentic Pipeline
#   │
#   └─ Retrieval-only fallback on LLM 429
#   ↓
# Answer
#   ↓
# TTS
# ============================================================


@app.post(
    "/voice/research"
)
async def voice_research(

    file: UploadFile = File(...),

    tenant_id: str = Form(
        default="default"
    ),

    language: str | None = Form(
        default=None
    ),

    output_format: Literal[
        "wav",
        "mp3",
        "both",
    ] = Form(
        default="both"
    ),

    synthesize_transcript: bool = Form(
        default=False
    ),

    synthesize_answer: bool = Form(
        default=True
    ),
):

    tenant_id = (
        _normalize_tenant_id(
            tenant_id
        )
    )

    # ========================================================
    # AUDIO
    # ========================================================

    audio_bytes = (
        await file.read()
    )

    if not audio_bytes:

        raise HTTPException(
            status_code=400,
            detail=(
                "Uploaded audio file is empty."
            ),
        )

    # ========================================================
    # STT
    # ========================================================

    try:

        transcript = (
            await _transcribe_audio(

                audio_bytes=
                    audio_bytes,

                upload_file=
                    file,

                filename=
                    file.filename
                    or
                    "voice.wav",

                language=
                    language,
            )
        )

    except RateLimitError:
        raise

    except Exception as exc:

        raise HTTPException(

            status_code=500,

            detail=(
                "STT failed: "
                f"{exc}"
            ),
        ) from exc

    # ========================================================
    # OPTIONAL TRANSCRIPT AUDIO
    # ========================================================

    if synthesize_transcript:

        try:

            transcript_audio = (
                await _synthesize_audio(

                    text=
                        transcript,

                    output_format=
                        output_format,
                )
            )

            transcript_audio = (
                _json_safe(
                    transcript_audio
                )
            )

        except Exception as exc:

            transcript_audio = {

                "status":
                    "failed",

                "generated":
                    False,

                "error":
                    str(
                        exc
                    ),
            }

    else:

        transcript_audio = {

            "status":
                "skipped",

            "generated":
                False,

            "reason":
                "disabled",
        }

    # ========================================================
    # RESEARCH
    #
    # research_service now decides:
    #
    # normal mode
    # OR
    # retrieval-only fallback
    # ========================================================

    try:

        (
            session_id,
            result_state,
            review_id,

        ) = await run_research(

            tenant_id=
                tenant_id,

            query=
                transcript,
        )

    # Normally handled inside research_service.
    # Keep these so unexpected unhandled rate limits still
    # reach the global 429 handlers.
    except LLMRateLimitError:
        raise

    except RateLimitError:
        raise

    except Exception as exc:

        raise HTTPException(

            status_code=500,

            detail=(
                "Research pipeline failed: "
                f"{exc}"
            ),
        ) from exc

    research_result = (
        _research_response(

            session_id=
                session_id,

            state=
                result_state,

            review_id=
                review_id,
        )
    )

    answer = (
        research_result.get(
            "answer"
        )
        or
        ""
    ).strip()

    # ========================================================
    # ANSWER TTS
    # ========================================================

    if (
        synthesize_answer
        and
        answer
    ):

        try:

            answer_audio = (
                await _synthesize_audio(

                    text=
                        answer,

                    output_format=
                        output_format,
                )
            )

            answer_audio = (
                _json_safe(
                    answer_audio
                )
            )

        except Exception as exc:

            # Research answer remains valid even if TTS fails.
            answer_audio = {

                "status":
                    "failed",

                "generated":
                    False,

                "error":
                    str(
                        exc
                    ),
            }

    else:

        answer_audio = {

            "status":
                "skipped",

            "generated":
                False,

            "reason":
                (
                    "empty_answer"
                    if not answer
                    else
                    "disabled"
                ),
        }

    # ========================================================
    # RESPONSE
    # ========================================================

    return _json_safe(
        {

            "status":
                research_result.get(
                    "status",
                    "success",
                ),

            "session_id":
                str(
                    session_id
                ),

            # =================================================
            # STT
            # =================================================

            "stt": {

                "status":
                    "success",

                "text":
                    transcript,

                "transcript":
                    transcript,

                "filename":
                    file.filename,

                "content_type":
                    file.content_type,

                "language":
                    language,
            },

            "transcript":
                transcript,

            "transcript_audio":
                transcript_audio,

            # =================================================
            # FULL RESEARCH RESULT
            # =================================================

            "research":
                research_result,

            # =================================================
            # LLM / FALLBACK METADATA
            # =================================================

            "llm": {

                "answer":
                    answer,

                "status":
                    research_result.get(
                        "status"
                    ),

                "response_mode":
                    research_result.get(
                        "response_mode"
                    ),

                "evaluation_mode":
                    research_result.get(
                        "evaluation_mode"
                    ),

                "llm_generated":
                    research_result.get(
                        "llm_generated",
                        False,
                    ),

                "degraded_mode":
                    research_result.get(
                        "degraded_mode",
                        False,
                    ),

                "claim_extraction_completed":
                    research_result.get(
                        "claim_extraction_completed",
                        False,
                    ),

                "nli_evaluation_completed":
                    research_result.get(
                        "nli_evaluation_completed",
                        False,
                    ),

                "scores":
                    research_result.get(
                        "scores"
                    ),

                "requires_human_review":
                    research_result.get(
                        "requires_human_review",
                        False,
                    ),

                "rate_limit":
                    research_result.get(
                        "rate_limit"
                    ),

                "next_action":
                    research_result.get(
                        "next_action"
                    ),
            },

            "answer":
                answer,

            "answer_audio":
                answer_audio,

            # =================================================
            # FALLBACK EVIDENCE
            # =================================================

            "retrieved_context":
                research_result.get(
                    "retrieved_context",
                    [],
                ),
        }
    )


# ============================================================
# TEXT DOCUMENT INGESTION
# ============================================================


@app.post(
    "/documents/text"
)
async def documents_text(
    request: TextDocumentRequest,
):

    tenant_id = (
        _normalize_tenant_id(
            request.tenant_id
        )
    )

    try:

        result = await _call_supported(

            ingest_raw_text,

            tenant_id=
                tenant_id,

            title=
                request.title.strip(),

            text=
                request.text,

            content=
                request.text,
        )

        return _json_safe(
            result
        )

    except Exception as exc:

        raise HTTPException(

            status_code=500,

            detail=(
                "Text ingestion failed: "
                f"{exc}"
            ),
        ) from exc


# ============================================================
# URL DOCUMENT INGESTION
# ============================================================


@app.post(
    "/documents/url"
)
async def documents_url(
    request: URLDocumentRequest,
):

    tenant_id = (
        _normalize_tenant_id(
            request.tenant_id
        )
    )

    url = (
        request.url
        or
        ""
    ).strip()

    if not url:

        raise HTTPException(
            status_code=400,
            detail=(
                "URL cannot be empty."
            ),
        )

    try:

        result = await _call_supported(

            ingest_url,

            tenant_id=
                tenant_id,

            url=
                url,

            async_mode=
                request.async_mode,
        )

        return _json_safe(
            result
        )

    except Exception as exc:

        raise HTTPException(

            status_code=500,

            detail=(
                "URL ingestion failed: "
                f"{exc}"
            ),
        ) from exc


# ============================================================
# FILE DOCUMENT INGESTION
# ============================================================


@app.post(
    "/documents/upload"
)
async def documents_upload(

    file: UploadFile = File(...),

    tenant_id: str = Form(
        default="default"
    ),
):

    tenant_id = (
        _normalize_tenant_id(
            tenant_id
        )
    )

    file_bytes = (
        await file.read()
    )

    if not file_bytes:

        raise HTTPException(
            status_code=400,
            detail=(
                "Uploaded document is empty."
            ),
        )

    try:

        await file.seek(
            0
        )

    except Exception:
        pass

    try:

        result = await _call_supported(

            ingest_upload,

            tenant_id=
                tenant_id,

            file=
                file,

            upload=
                file,

            filename=
                file.filename
                or
                "document",

            data=
                file_bytes,

            file_bytes=
                file_bytes,

            content=
                file_bytes,

            content_type=
                file.content_type,
        )

        return _json_safe(
            result
        )

    except Exception as exc:

        raise HTTPException(

            status_code=500,

            detail=(
                "Document ingestion failed: "
                f"{exc}"
            ),
        ) from exc


# ============================================================
# JOB STATUS
# ============================================================


@app.get(
    "/jobs/{job_id}"
)
async def jobs_get(
    job_id: str,
):

    try:

        result = await _maybe_await(
            get_job(
                job_id
            )
        )

    except Exception as exc:

        raise HTTPException(

            status_code=500,

            detail=(
                "Unable to retrieve job: "
                f"{exc}"
            ),
        ) from exc

    if result is None:

        raise HTTPException(
            status_code=404,
            detail=(
                "Job not found."
            ),
        )

    return _json_safe(
        result
    )


# ============================================================
# DEBUG DOCUMENTS
# ============================================================


@app.get(
    "/debug/documents"
)
async def debug_documents(

    tenant_id: str = Query(
        default="default"
    ),
):

    tenant_id = (
        _normalize_tenant_id(
            tenant_id
        )
    )

    try:

        documents = await _call_supported(

            list_documents,

            tenant_id=
                tenant_id,
        )

        return _json_safe(
            {

                "tenant_id":
                    tenant_id,

                "documents":
                    documents
                    or
                    [],
            }
        )

    except Exception as exc:

        raise HTTPException(

            status_code=500,

            detail=(
                "Unable to list documents: "
                f"{exc}"
            ),
        ) from exc


# ============================================================
# DEBUG DOCUMENT CHUNKS
# ============================================================


@app.get(
    "/debug/documents/{document_id}/chunks"
)
async def debug_document_chunks(
    document_id: str,
):

    try:

        chunks = await _call_supported(

            get_document_chunks,

            document_id=
                document_id,
        )

        chunks = (
            chunks
            or
            []
        )

        return _json_safe(
            {

                "document_id":
                    document_id,

                "count":
                    len(
                        chunks
                    ),

                "chunks":
                    chunks,
            }
        )

    except Exception as exc:

        raise HTTPException(

            status_code=500,

            detail=(
                "Unable to retrieve chunks: "
                f"{exc}"
            ),
        ) from exc


# ============================================================
# DEBUG RETRIEVAL
# ============================================================


@app.get(
    "/debug/retrieval"
)
async def debug_retrieval(

    query: str = Query(
        ...,
        min_length=1,
    ),

    tenant_id: str = Query(
        default="default"
    ),

    limit: int = Query(
        default=5,
        ge=1,
        le=50,
    ),
):

    query = (
        _normalize_query(
            query
        )
    )

    tenant_id = (
        _normalize_tenant_id(
            tenant_id
        )
    )

    try:

        result = await _call_supported(

            hybrid_search_debug,

            tenant_id=
                tenant_id,

            query=
                query,

            limit=
                limit,

            use_cache=
                False,
        )

        if not isinstance(
            result,
            dict,
        ):

            result = {
                "results":
                    result
            }

        final_results = (
            result.get(
                "final_results"
            )
            or
            result.get(
                "reranked_results"
            )
            or
            result.get(
                "results"
            )
            or
            []
        )

        try:

            context = (
                build_context(
                    final_results
                )
            )

        except Exception:

            context = ""

        result[
            "context"
        ] = context

        result[
            "context_length"
        ] = len(
            context
        )

        return _json_safe(
            result
        )

    except Exception as exc:

        raise HTTPException(

            status_code=500,

            detail=(
                "Retrieval debug failed: "
                f"{exc}"
            ),
        ) from exc


# ============================================================
# RESEARCH
#
# Important:
#
# run_research() itself handles:
#
# normal pipeline
# OR
# retrieval-only fallback.
# ============================================================


@app.post(
    "/research"
)
async def research(
    request: ResearchRequest,
):

    tenant_id = (
        _normalize_tenant_id(
            request.tenant_id
        )
    )

    query = (
        _normalize_query(
            request.query
        )
    )

    try:

        (
            session_id,
            result_state,
            review_id,

        ) = await run_research(

            tenant_id=
                tenant_id,

            query=
                query,
        )

        return _research_response(

            session_id=
                session_id,

            state=
                result_state,

            review_id=
                review_id,
        )

    # Usually consumed inside research_service.
    # These are safety fallbacks only.
    except LLMRateLimitError:
        raise

    except RateLimitError:
        raise

    except HTTPException:
        raise

    except Exception as exc:

        raise HTTPException(

            status_code=500,

            detail=(
                "Research pipeline failed: "
                f"{exc}"
            ),
        ) from exc


# ============================================================
# GET RESEARCH SESSION
# ============================================================


@app.get(
    "/research/{session_id}"
)
async def research_get(
    session_id: str,
):

    try:

        session = await _maybe_await(
            get_session(
                session_id
            )
        )

    except Exception as exc:

        raise HTTPException(

            status_code=500,

            detail=(
                "Unable to retrieve research session: "
                f"{exc}"
            ),
        ) from exc

    if session is None:

        raise HTTPException(
            status_code=404,
            detail=(
                "Research session not found."
            ),
        )

    state = (
        _extract_session_state(
            session
        )
    )

    return _research_response(

        session_id=
            session_id,

        state=
            state,

        review_id=
            state.get(
                "review_id"
            ),
    )


# ============================================================
# COMPLETE REVIEW HELPER
# ============================================================


async def _complete_review_safely(

    session_id: str,

    state: dict[str, Any],

    action: str,

    feedback: str | None,

) -> None:

    review_id = (
        state.get(
            "review_id"
        )
    )

    try:

        await _call_supported(

            complete_review,

            review_id=
                review_id,

            session_id=
                session_id,

            action=
                action,

            decision=
                action,

            feedback=
                feedback,

            status=
                action,
        )

    except Exception as exc:

        print(
            "[REVIEW] complete_review warning:",
            repr(
                exc
            ),
        )


# ============================================================
# SAVE REVIEW STATE
# ============================================================


async def _save_review_state(
    session_id: str,
    state: dict[str, Any],
) -> None:

    safe_state = (
        _json_safe(
            state
        )
    )

    await _call_supported(

        save_session_state,

        session_id=
            session_id,

        state=
            safe_state,
    )


# ============================================================
# HUMAN REVIEW
# ============================================================


@app.post(
    "/research/{session_id}/review"
)
async def research_review(

    session_id: str,

    request: ReviewRequest,
):

    # ========================================================
    # LOAD SESSION
    # ========================================================

    try:

        session = await _maybe_await(
            get_session(
                session_id
            )
        )

    except Exception as exc:

        raise HTTPException(

            status_code=500,

            detail=(
                "Unable to retrieve research session: "
                f"{exc}"
            ),
        ) from exc

    if session is None:

        raise HTTPException(
            status_code=404,
            detail=(
                "Research session not found."
            ),
        )

    state = (
        _extract_session_state(
            session
        )
    )

    if not state:

        raise HTTPException(
            status_code=500,
            detail=(
                "Research session has no state."
            ),
        )

    feedback = (
        request.feedback.strip()
        if request.feedback
        else
        None
    )

    # ========================================================
    # APPROVE
    # ========================================================

    if request.action == "approve":

        answer = (
            state.get(
                "final_answer"
            )
            or
            state.get(
                "generated_answer"
            )
            or
            state.get(
                "answer"
            )
            or
            ""
        )

        state[
            "final_answer"
        ] = answer

        state[
            "answer"
        ] = answer

        state[
            "status"
        ] = "completed"

        state[
            "requires_human_review"
        ] = False

        state[
            "human_feedback"
        ] = feedback

        await _save_review_state(
            session_id=
                session_id,

            state=
                state,
        )

        await _complete_review_safely(

            session_id=
                session_id,

            state=
                state,

            action=
                "approve",

            feedback=
                feedback,
        )

        return _research_response(

            session_id=
                session_id,

            state=
                state,

            review_id=
                state.get(
                    "review_id"
                ),
        )

    # ========================================================
    # REJECT
    # ========================================================

    if request.action == "reject":

        state[
            "status"
        ] = "rejected"

        state[
            "requires_human_review"
        ] = False

        state[
            "human_feedback"
        ] = feedback

        await _save_review_state(
            session_id=
                session_id,

            state=
                state,
        )

        await _complete_review_safely(

            session_id=
                session_id,

            state=
                state,

            action=
                "reject",

            feedback=
                feedback,
        )

        return _research_response(

            session_id=
                session_id,

            state=
                state,

            review_id=
                state.get(
                    "review_id"
                ),
        )

    # ========================================================
    # RESEARCH MORE
    # ========================================================

    original_query = (
        state.get(
            "query"
        )
        or
        ""
    ).strip()

    if feedback:

        next_query = (
            f"{original_query}\n\n"
            f"Reviewer feedback: {feedback}"
        )

    else:

        next_query = (
            original_query
        )

    if not next_query:

        raise HTTPException(
            status_code=400,
            detail=(
                "No research query is available."
            ),
        )

    # ========================================================
    # RESET BEFORE RESUME
    # ========================================================

    state[
        "query"
    ] = next_query

    state[
        "active_query"
    ] = next_query

    state[
        "human_feedback"
    ] = feedback

    state[
        "requires_human_review"
    ] = False

    state[
        "review_reason"
    ] = None

    state[
        "generated_answer"
    ] = None

    state[
        "final_answer"
    ] = None

    state[
        "answer"
    ] = None

    state[
        "analysis"
    ] = None

    state[
        "analysis_result"
    ] = None

    state[
        "claims"
    ] = []

    state[
        "claim_evaluations"
    ] = []

    state[
        "claim_extraction_completed"
    ] = False

    state[
        "nli_evaluation_completed"
    ] = False

    state[
        "evidence_score"
    ] = 0.0

    state[
        "retrieval_quality"
    ] = 0.0

    state[
        "grounding_score"
    ] = 0.0

    state[
        "faithfulness_score"
    ] = 0.0

    state[
        "hallucination_score"
    ] = 0.0

    state[
        "citation_score"
    ] = 0.0

    state[
        "confidence_score"
    ] = 0.0

    state[
        "degraded_mode"
    ] = False

    state[
        "response_mode"
    ] = "full_agentic"

    state[
        "evaluation_mode"
    ] = None

    state[
        "rate_limit"
    ] = None

    state[
        "next_action"
    ] = None

    state[
        "status"
    ] = "running"

    state[
        "retry_count"
    ] = (
        int(
            state.get(
                "retry_count",
                0,
            )
            or
            0
        )
        +
        1
    )

    tenant_id = (
        _normalize_tenant_id(
            state.get(
                "tenant_id"
            )
        )
    )

    try:

        (
            resumed_session_id,
            result_state,
            review_id,

        ) = await run_research(

            tenant_id=
                tenant_id,

            query=
                next_query,

            existing_state=
                state,
        )

        await _complete_review_safely(

            session_id=
                session_id,

            state=
                state,

            action=
                "research_more",

            feedback=
                feedback,
        )

        return _research_response(

            session_id=
                resumed_session_id,

            state=
                result_state,

            review_id=
                review_id,
        )

    except LLMRateLimitError:
        raise

    except RateLimitError:
        raise

    except HTTPException:
        raise

    except Exception as exc:

        raise HTTPException(

            status_code=500,

            detail=(
                "Additional research failed: "
                f"{exc}"
            ),
        ) from exc