# streamlit_app.py

from __future__ import annotations

import os
from typing import Any

import httpx
import streamlit as st


# ============================================================
# CONFIG
# ============================================================

BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "http://127.0.0.1:8000",
).rstrip("/")

REQUEST_TIMEOUT = 240.0


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title=(
        "Voice Research Agent"
    ),

    page_icon="🎙️",

    layout="wide",
)


# ============================================================
# SESSION DEFAULTS
# ============================================================

SESSION_DEFAULTS = {
    "last_transcript":
        None,

    "last_transcript_audio":
        None,

    "last_result":
        None,

    "last_answer_audio":
        None,

    "last_voice_response":
        None,
}

for key, value in (
    SESSION_DEFAULTS.items()
):

    if key not in st.session_state:

        st.session_state[
            key
        ] = value


# ============================================================
# HTTP HELPERS
# ============================================================


def api_get(
    path: str,
) -> dict[str, Any]:

    with httpx.Client(
        timeout=REQUEST_TIMEOUT
    ) as client:

        response = client.get(
            f"{BACKEND_URL}{path}"
        )

        response.raise_for_status()

        return response.json()


def api_post_json(
    path: str,
    payload: dict[str, Any],
) -> dict[str, Any]:

    with httpx.Client(
        timeout=REQUEST_TIMEOUT
    ) as client:

        response = client.post(
            f"{BACKEND_URL}{path}",

            json=
                payload,
        )

        response.raise_for_status()

        return response.json()


# ============================================================
# ERROR MESSAGE
# ============================================================


def show_http_error(
    exc: httpx.HTTPStatusError,
) -> None:

    try:

        payload = (
            exc.response.json()
        )

    except Exception:

        payload = {}

    message = (
        payload.get(
            "message"
        )
        or
        payload.get(
            "detail"
        )
        or
        exc.response.text
        or
        "Backend request failed."
    )

    status_code = (
        exc.response.status_code
    )

    if status_code == 429:

        retry_after = (
            payload.get(
                "retry_after_seconds"
            )
        )

        st.warning(
            (
                "Model rate limit reached. "
                f"{message}"
            )
        )

        if retry_after:

            st.caption(
                (
                    "Retry after approximately "
                    f"{retry_after} seconds."
                )
            )

    else:

        st.error(
            (
                f"Backend error "
                f"{status_code}: "
                f"{message}"
            )
        )


# ============================================================
# VOICE RESEARCH API
# ============================================================


def run_voice_research(
    audio_file,
    tenant_id: str,
    language: str | None,
) -> dict[str, Any]:

    filename = (
        getattr(
            audio_file,
            "name",
            "voice.wav",
        )
        or
        "voice.wav"
    )

    media_type = (
        getattr(
            audio_file,
            "type",
            "audio/wav",
        )
        or
        "audio/wav"
    )

    files = {
        "file": (
            filename,
            audio_file.getvalue(),
            media_type,
        )
    }

    data = {
        "tenant_id":
            tenant_id,

        "output_format":
            "both",

        "synthesize_transcript":
            "true",

        "synthesize_answer":
            "true",
    }

    if language:

        data[
            "language"
        ] = language

    with httpx.Client(
        timeout=REQUEST_TIMEOUT
    ) as client:

        response = client.post(
            (
                f"{BACKEND_URL}"
                "/voice/research"
            ),

            files=
                files,

            data=
                data,
        )

        response.raise_for_status()

        return response.json()


# ============================================================
# STANDALONE RESEARCH
# ============================================================


def run_research(
    query: str,
    tenant_id: str,
) -> dict[str, Any]:

    return api_post_json(
        "/research",

        {
            "query":
                query,

            "tenant_id":
                tenant_id,
        },
    )


# ============================================================
# TEXT -> AUDIO
# ============================================================


def generate_text_audio(
    text: str,
) -> dict[str, Any]:

    return api_post_json(
        "/audio/transcript-to-audio",

        {
            "text":
                text,

            "rate":
                175,

            "volume":
                1.0,

            "output_format":
                "both",
        },
    )


# ============================================================
# FULL AUDIO URL
# ============================================================


def full_audio_url(
    url: str | None,
) -> str | None:

    if not url:

        return None

    if (
        url.startswith(
            "http://"
        )
        or
        url.startswith(
            "https://"
        )
    ):

        return url

    return (
        f"{BACKEND_URL}"
        f"{url}"
    )


# ============================================================
# DOWNLOAD AUDIO
# ============================================================


def fetch_audio_bytes(
    url: str,
) -> bytes | None:

    full_url = (
        full_audio_url(
            url
        )
    )

    if not full_url:

        return None

    try:

        with httpx.Client(
            timeout=60.0
        ) as client:

            response = (
                client.get(
                    full_url
                )
            )

            response.raise_for_status()

            return response.content

    except Exception:

        return None


# ============================================================
# RENDER AUDIO
# ============================================================


def render_audio_result(
    title: str,
    result: dict[str, Any] | None,
    key_prefix: str,
) -> None:

    if not isinstance(
        result,
        dict,
    ):

        return

    status = (
        result.get(
            "status"
        )
    )

    if status == "failed":

        st.warning(
            (
                f"{title} generation failed: "
                f"{result.get('error', 'Unknown error')}"
            )
        )

        return

    if status == "skipped":

        return

    wav_url = (
        result.get(
            "wav_url"
        )
    )

    mp3_url = (
        result.get(
            "mp3_url"
        )
    )

    wav = (
        result.get(
            "wav"
        )
    )

    mp3 = (
        result.get(
            "mp3"
        )
    )

    if (
        not wav_url
        and
        isinstance(
            wav,
            dict,
        )
    ):

        wav_url = wav.get(
            "url"
        )

    if (
        not mp3_url
        and
        isinstance(
            mp3,
            dict,
        )
    ):

        mp3_url = mp3.get(
            "url"
        )

    if (
        not wav_url
        and
        not mp3_url
    ):

        return

    st.markdown(
        f"### {title}"
    )

    wav_col, mp3_col = (
        st.columns(2)
    )

    # ========================================================
    # WAV
    # ========================================================

    with wav_col:

        if wav_url:

            st.caption(
                "WAV"
            )

            st.audio(
                full_audio_url(
                    wav_url
                ),
                format="audio/wav",
            )

            wav_bytes = (
                fetch_audio_bytes(
                    wav_url
                )
            )

            if wav_bytes:

                st.download_button(
                    label=(
                        "Download WAV"
                    ),

                    data=
                        wav_bytes,

                    file_name=(
                        f"{key_prefix}.wav"
                    ),

                    mime=
                        "audio/wav",

                    key=(
                        f"{key_prefix}_wav_download"
                    ),
                )

        else:

            st.caption(
                "WAV not generated"
            )

    # ========================================================
    # MP3
    # ========================================================

    with mp3_col:

        if mp3_url:

            st.caption(
                "MP3"
            )

            st.audio(
                full_audio_url(
                    mp3_url
                ),
                format="audio/mpeg",
            )

            mp3_bytes = (
                fetch_audio_bytes(
                    mp3_url
                )
            )

            if mp3_bytes:

                st.download_button(
                    label=(
                        "Download MP3"
                    ),

                    data=
                        mp3_bytes,

                    file_name=(
                        f"{key_prefix}.mp3"
                    ),

                    mime=
                        "audio/mpeg",

                    key=(
                        f"{key_prefix}_mp3_download"
                    ),
                )

        else:

            st.caption(
                "MP3 not generated"
            )


# ============================================================
# SCORES
# ============================================================


def render_scores(
    result: dict[str, Any],
) -> None:

    columns = (
        st.columns(5)
    )

    metrics = [
        (
            "Confidence",
            result.get(
                "confidence_score"
            ),
        ),
        (
            "Grounding",
            result.get(
                "grounding_score"
            ),
        ),
        (
            "Faithfulness",
            result.get(
                "faithfulness_score"
            ),
        ),
        (
            "Hallucination",
            result.get(
                "hallucination_score"
            ),
        ),
        (
            "Citation",
            result.get(
                "citation_score"
            ),
        ),
    ]

    for (
        column,
        (
            label,
            value,
        ),

    ) in zip(
        columns,
        metrics,
    ):

        if isinstance(
            value,
            (
                int,
                float,
            ),
        ):

            column.metric(
                label,
                f"{value:.2%}",
            )

        else:

            column.metric(
                label,
                "—",
            )


# ============================================================
# RATE LIMIT / FALLBACK INFO
# ============================================================


def render_response_mode(
    result: dict[str, Any],
) -> None:

    response_mode = (
        result.get(
            "response_mode"
        )
    )

    degraded = bool(
        result.get(
            "degraded_mode",
            False,
        )
    )

    if degraded:

        st.warning(
            (
                "The LLM pipeline is running "
                "in degraded retrieval-only mode."
            )
        )

        if response_mode:

            st.caption(
                (
                    "Response mode: "
                    f"{response_mode}"
                )
            )

    rate_limit = (
        result.get(
            "rate_limit"
        )
    )

    if isinstance(
        rate_limit,
        dict,
    ):

        retry_after = (
            rate_limit.get(
                "retry_after_seconds"
            )
        )

        if retry_after:

            st.caption(
                (
                    "Full LLM synthesis can be "
                    "retried in approximately "
                    f"{retry_after} seconds."
                )
            )


# ============================================================
# HUMAN REVIEW
# ============================================================


def render_human_review(
    result: dict[str, Any],
) -> None:

    session_id = (
        result.get(
            "session_id"
        )
    )

    if not session_id:

        return

    feedback = st.text_area(
        "Reviewer feedback",

        placeholder=(
            "Add corrections or tell the "
            "agent what to research next."
        ),

        key=(
            f"feedback_{session_id}"
        ),
    )

    approve_col, research_col, reject_col = (
        st.columns(3)
    )

    if approve_col.button(
        "Approve",
        key=f"approve_{session_id}",
    ):

        try:

            review = api_post_json(
                (
                    f"/research/"
                    f"{session_id}/review"
                ),

                {
                    "action":
                        "approve",

                    "feedback":
                        feedback
                        or
                        None,
                },
            )

            st.session_state[
                "last_result"
            ] = review

            st.success(
                "Result approved."
            )

            st.rerun()

        except httpx.HTTPStatusError as exc:

            show_http_error(
                exc
            )

    if research_col.button(
        "Research more",
        key=f"research_{session_id}",
    ):

        try:

            review = api_post_json(
                (
                    f"/research/"
                    f"{session_id}/review"
                ),

                {
                    "action":
                        "research_more",

                    "feedback":
                        feedback
                        or
                        None,
                },
            )

            st.session_state[
                "last_result"
            ] = review

            st.session_state[
                "last_answer_audio"
            ] = None

            st.rerun()

        except httpx.HTTPStatusError as exc:

            show_http_error(
                exc
            )

    if reject_col.button(
        "Reject",
        key=f"reject_{session_id}",
    ):

        try:

            review = api_post_json(
                (
                    f"/research/"
                    f"{session_id}/review"
                ),

                {
                    "action":
                        "reject",

                    "feedback":
                        feedback
                        or
                        None,
                },
            )

            st.session_state[
                "last_result"
            ] = review

            st.error(
                "Result rejected."
            )

            st.rerun()

        except httpx.HTTPStatusError as exc:

            show_http_error(
                exc
            )


# ============================================================
# RESEARCH RESULT
# ============================================================


def render_result(
    result: dict[str, Any],
) -> None:

    if not isinstance(
        result,
        dict,
    ):

        return

    st.subheader(
        "Research answer"
    )

    render_response_mode(
        result
    )

    answer = (
        result.get(
            "answer"
        )
        or
        result.get(
            "final_answer"
        )
        or
        "No answer returned."
    )

    st.markdown(
        answer
    )

    st.write("")

    render_scores(
        result
    )

    st.write("")

    metadata_col1, metadata_col2, metadata_col3 = (
        st.columns(3)
    )

    metadata_col1.caption(
        (
            "Session: "
            f"{result.get('session_id', '—')}"
        )
    )

    metadata_col2.caption(
        (
            "Status: "
            f"{result.get('status', '—')}"
        )
    )

    metadata_col3.caption(
        (
            "Intent: "
            f"{result.get('intent', '—')}"
        )
    )

    active_query = (
        result.get(
            "active_query"
        )
    )

    if active_query:

        with st.expander(
            "Retrieval query"
        ):

            st.code(
                active_query
            )

    evaluations = (
        result.get(
            "claim_evaluations"
        )
        or
        []
    )

    if evaluations:

        with st.expander(
            (
                "Claim-level grounding "
                "/ NLI evaluation"
            )
        ):

            st.dataframe(
                evaluations,
                use_container_width=True,
            )

    if result.get(
        "requires_human_review"
    ):

        st.warning(
            (
                "Confidence gate selected "
                "human review."
            )
        )

        render_human_review(
            result
        )


# ============================================================
# DOCUMENT INGESTION
# ============================================================


def render_document_ingestion(
    tenant_id: str,
) -> None:

    st.sidebar.subheader(
        "Knowledge base"
    )

    upload = (
        st.sidebar.file_uploader(
            (
                "Upload PDF, DOCX, CSV, "
                "TXT or MD"
            ),

            type=[
                "pdf",
                "docx",
                "csv",
                "txt",
                "md",
            ],
        )
    )

    if (
        upload
        and
        st.sidebar.button(
            "Ingest file"
        )
    ):

        try:

            with st.spinner(
                "Ingesting document..."
            ):

                with httpx.Client(
                    timeout=REQUEST_TIMEOUT
                ) as client:

                    response = client.post(
                        (
                            f"{BACKEND_URL}"
                            "/documents/upload"
                        ),

                        files={
                            "file": (
                                upload.name,
                                upload.getvalue(),
                                upload.type
                                or
                                "application/octet-stream",
                            )
                        },

                        data={
                            "tenant_id":
                                tenant_id
                        },
                    )

                    response.raise_for_status()

                    payload = (
                        response.json()
                    )

            st.sidebar.success(
                (
                    "Stored "
                    f"{payload.get('chunks', 0)} "
                    "chunks"
                )
            )

        except httpx.HTTPStatusError as exc:

            show_http_error(
                exc
            )

        except Exception as exc:

            st.sidebar.error(
                str(
                    exc
                )
            )

    url = (
        st.sidebar.text_input(
            "Or ingest a URL"
        )
    )

    if (
        url
        and
        st.sidebar.button(
            "Ingest URL"
        )
    ):

        try:

            with st.spinner(
                "Crawling and indexing URL..."
            ):

                payload = (
                    api_post_json(
                        "/documents/url",

                        {
                            "tenant_id":
                                tenant_id,

                            "url":
                                url,

                            "async_mode":
                                False,
                        },
                    )
                )

            st.sidebar.success(
                (
                    "Stored "
                    f"{payload.get('chunks', 0)} "
                    "chunks"
                )
            )

        except httpx.HTTPStatusError as exc:

            show_http_error(
                exc
            )

        except Exception as exc:

            st.sidebar.error(
                str(
                    exc
                )
            )


# ============================================================
# HEADER
# ============================================================

st.title(
    "🎙️ Agentic Research Assistant"
)

st.caption(
    (
        "Voice → STT → Planner → Discovery → "
        "Retrieval → Analysis → Synthesis → "
        "Claim Extraction → Grounding/NLI → "
        "Scores → Confidence → Human Review → TTS"
    )
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "Settings"
    )

    tenant_id = (
        st.text_input(
            "Tenant ID",
            value="demo",
        )
    )

    language_choice = (
        st.selectbox(
            "Voice language",

            [
                (
                    "Auto detect",
                    "",
                ),
                (
                    "English",
                    "en",
                ),
                (
                    "Hindi",
                    "hi",
                ),
                (
                    "Tamil",
                    "ta",
                ),
                (
                    "Telugu",
                    "te",
                ),
            ],

            format_func=lambda item:
                item[0],
        )
    )

    try:

        health = (
            api_get(
                "/health"
            )
        )

        st.success(
            "Backend connected"
        )

        st.caption(
            (
                f"LLM: "
                f"{health.get('llm')}\n\n"
                f"Embedding: "
                f"{health.get('embedding')}"
            )
        )

    except Exception:

        st.error(
            (
                "Backend is not reachable. "
                "Start FastAPI using "
                "`python run_server.py`."
            )
        )


render_document_ingestion(
    tenant_id
)


# ============================================================
# TABS
# ============================================================

voice_tab, text_tab = (
    st.tabs(
        [
            "Voice query",
            "Text query",
        ]
    )
)


# ============================================================
# VOICE TAB
# ============================================================

with voice_tab:

    st.subheader(
        "Ask your research question by voice"
    )

    if hasattr(
        st,
        "audio_input",
    ):

        # Do not pass sample_rate. Some Streamlit versions
        # do not support that argument.
        audio_value = (
            st.audio_input(
                "Record your question"
            )
        )

    else:

        st.warning(
            (
                "Your Streamlit version does not "
                "support st.audio_input()."
            )
        )

        audio_value = (
            st.file_uploader(
                "Upload voice recording",

                type=[
                    "wav",
                    "mp3",
                    "m4a",
                    "webm",
                ],

                key=
                    "voice_file_fallback",
            )
        )

    if audio_value:

        st.markdown(
            "### Original recording"
        )

        st.audio(
            audio_value
        )

        if st.button(
            "Transcribe and research",
            type="primary",
            key="voice_research_button",
        ):

            try:

                with st.status(
                    (
                        "Processing voice "
                        "research..."
                    ),
                    expanded=True,
                ) as status:

                    st.write(
                        "1. Running speech-to-text..."
                    )

                    st.write(
                        (
                            "2. Generating transcript "
                            "WAV/MP3..."
                        )
                    )

                    st.write(
                        (
                            "3. Running Planner → "
                            "Discovery → Retrieval → "
                            "Analysis → Synthesis..."
                        )
                    )

                    st.write(
                        (
                            "4. Running Claim Extraction → "
                            "Grounding/NLI → Scores → "
                            "Confidence..."
                        )
                    )

                    st.write(
                        (
                            "5. Generating answer "
                            "WAV/MP3..."
                        )
                    )

                    payload = (
                        run_voice_research(
                            audio_value,

                            tenant_id,

                            language_choice[1]
                            or
                            None,
                        )
                    )

                    transcript = (
                        payload.get(
                            "transcript"
                        )
                        or
                        (
                            payload.get(
                                "stt"
                            )
                            or
                            {}
                        ).get(
                            "transcript"
                        )
                        or
                        ""
                    )

                    research_result = (
                        payload.get(
                            "research"
                        )
                        or
                        {}
                    )

                    transcript_audio = (
                        payload.get(
                            "transcript_audio"
                        )
                        or
                        {}
                    )

                    answer_audio = (
                        payload.get(
                            "answer_audio"
                        )
                        or
                        {}
                    )

                    st.session_state[
                        "last_voice_response"
                    ] = payload

                    st.session_state[
                        "last_transcript"
                    ] = transcript

                    st.session_state[
                        "last_transcript_audio"
                    ] = transcript_audio

                    st.session_state[
                        "last_result"
                    ] = research_result

                    st.session_state[
                        "last_answer_audio"
                    ] = answer_audio

                    status.update(
                        label=(
                            "Research complete"
                        ),

                        state="complete",
                    )

            except httpx.HTTPStatusError as exc:

                show_http_error(
                    exc
                )

            except Exception as exc:

                st.error(
                    (
                        "Voice request failed: "
                        f"{exc}"
                    )
                )

    # ========================================================
    # TRANSCRIPT
    # ========================================================

    transcript = (
        st.session_state.get(
            "last_transcript"
        )
    )

    if transcript:

        st.divider()

        st.markdown(
            "## STT output"
        )

        st.text_area(
            "Transcript",

            value=
                transcript,

            height=
                120,

            disabled=
                True,
        )

        render_audio_result(
            title=(
                "Transcript audio"
            ),

            result=
                st.session_state.get(
                    "last_transcript_audio"
                ),

            key_prefix=
                "transcript",
        )


# ============================================================
# TEXT TAB
# ============================================================

with text_tab:

    query = (
        st.text_area(
            "Research question",

            placeholder=(
                "Ask a question about your "
                "indexed documents or a "
                "research topic..."
            ),

            height=140,
        )
    )

    generate_audio = (
        st.checkbox(
            (
                "Generate WAV and MP3 "
                "for the research answer"
            ),

            value=True,
        )
    )

    if st.button(
        "Run research",
        key="run_text",
        type="primary",
    ):

        if not query.strip():

            st.warning(
                (
                    "Enter a research "
                    "question."
                )
            )

        else:

            try:

                with st.status(
                    (
                        "Running agentic "
                        "research pipeline..."
                    ),
                    expanded=True,
                ) as status:

                    st.write(
                        (
                            "Planner → Discovery → "
                            "Retrieval → Analysis"
                        )
                    )

                    st.write(
                        (
                            "Synthesis → Claims → "
                            "Grounding/NLI → Confidence"
                        )
                    )

                    result = (
                        run_research(
                            query.strip(),
                            tenant_id,
                        )
                    )

                    st.session_state[
                        "last_result"
                    ] = result

                    answer = (
                        result.get(
                            "answer"
                        )
                        or
                        ""
                    )

                    if (
                        generate_audio
                        and
                        answer
                    ):

                        st.write(
                            (
                                "Generating answer "
                                "WAV/MP3..."
                            )
                        )

                        answer_audio = (
                            generate_text_audio(
                                answer
                            )
                        )

                        st.session_state[
                            "last_answer_audio"
                        ] = answer_audio

                    else:

                        st.session_state[
                            "last_answer_audio"
                        ] = None

                    status.update(
                        label=(
                            "Research complete"
                        ),

                        state="complete",
                    )

            except httpx.HTTPStatusError as exc:

                show_http_error(
                    exc
                )

            except Exception as exc:

                st.error(
                    (
                        "Research request failed: "
                        f"{exc}"
                    )
                )


# ============================================================
# FINAL RESEARCH RESULT
# ============================================================

last_result = (
    st.session_state.get(
        "last_result"
    )
)

if last_result:

    st.divider()

    render_result(
        last_result
    )

    render_audio_result(
        title=(
            "Research answer audio"
        ),

        result=
            st.session_state.get(
                "last_answer_audio"
            ),

        key_prefix=
            "research_answer",
    )


# ============================================================
# DEBUG JSON
# ============================================================

last_voice_response = (
    st.session_state.get(
        "last_voice_response"
    )
)

if last_voice_response:

    with st.expander(
        "Complete voice pipeline JSON"
    ):

        st.json(
            last_voice_response
        )