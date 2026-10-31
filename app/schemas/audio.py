from pydantic import (
    BaseModel,
    Field,
)


# ============================================================
# SPEECH TO TEXT
# ============================================================

class STTResponse(BaseModel):
    """
    Response returned by the AssemblyAI
    speech-to-text service.
    """

    status: str = Field(
        default="success",
        description="STT processing status",
    )

    transcript: str = Field(
        ...,
        description="Text generated from the input audio",
    )

    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="AssemblyAI transcription confidence",
    )

    audio_duration_ms: int | None = Field(
        default=None,
        ge=0,
        description="Audio duration in milliseconds",
    )

    inference_time_ms: int | None = Field(
        default=None,
        ge=0,
        description="AssemblyAI processing time",
    )

    session_id: str | None = Field(
        default=None,
        description="AssemblyAI session identifier",
    )


# ============================================================
# TEXT TO SPEECH REQUEST
# ============================================================

class TTSRequest(BaseModel):
    """
    Request body used by the pyttsx3
    text-to-speech service.
    """

    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description=(
            "Text that should be converted "
            "into speech"
        ),
    )

    voice_id: str | None = Field(
        default=None,
        description=(
            "Optional pyttsx3 system voice ID. "
            "If omitted, the system default "
            "voice will be used."
        ),
    )

    rate: int | None = Field(
        default=None,
        ge=80,
        le=300,
        description=(
            "Speech rate. "
            "Typical value is around 175."
        ),
    )

    volume: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Speech volume between "
            "0.0 and 1.0"
        ),
    )


# ============================================================
# INDIVIDUAL PYTTSX3 VOICE
# ============================================================

class TTSVoice(BaseModel):
    """
    Information about one installed
    pyttsx3 system voice.
    """

    index: int = Field(
        ...,
        ge=0,
        description="Voice index",
    )

    id: str = Field(
        ...,
        description="System-specific voice ID",
    )

    name: str = Field(
        ...,
        description="Human-readable voice name",
    )

    languages: list[str] = Field(
        default_factory=list,
        description=(
            "Languages reported by the "
            "operating system voice"
        ),
    )


# ============================================================
# AVAILABLE VOICES RESPONSE
# ============================================================

class TTSVoicesResponse(BaseModel):
    """
    Response returned by GET /audio/voices.
    """

    status: str = Field(
        default="success",
        description="Voice listing status",
    )

    voices: list[TTSVoice] = Field(
        default_factory=list,
        description=(
            "Available pyttsx3 system voices"
        ),
    )


# ============================================================
# OPTIONAL AUDIO SERVICE STATUS
# ============================================================

class AudioServiceStatus(BaseModel):
    """
    Optional schema that can be used
    for an audio health-check endpoint.
    """

    stt_provider: str = Field(
        default="AssemblyAI",
    )

    tts_provider: str = Field(
        default="pyttsx3",
    )

    stt_available: bool = Field(
        default=True,
    )

    tts_available: bool = Field(
        default=True,
    )