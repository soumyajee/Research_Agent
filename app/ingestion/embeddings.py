import asyncio
import threading

from app.config import settings


# ============================================================
# GLOBAL MODEL
# ============================================================

_embedding_model = None

_embedding_inference_lock = threading.Lock()


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

def get_embedding_model():
    """
    Load SentenceTransformer lazily.

    IMPORTANT:
    This function should be called from the main/event-loop
    thread BEFORE inference is moved into asyncio.to_thread().

    This prevents the first PyTorch DLL initialization from
    occurring inside a worker thread on Windows.
    """

    global _embedding_model

    if _embedding_model is None:

        print(
            "[EMBEDDING] Loading model:"
        )

        print(
            settings.embedding_model
        )

        # ----------------------------------------------------
        # Import here intentionally.
        #
        # We don't import sentence_transformers when the module
        # itself is imported.
        # ----------------------------------------------------

        from sentence_transformers import (
            SentenceTransformer,
        )

        _embedding_model = SentenceTransformer(
            settings.embedding_model,
            device="cpu",
        )

        print(
            "[EMBEDDING] Model loaded successfully."
        )

    return _embedding_model


# ============================================================
# SYNCHRONOUS ENCODING
# ============================================================

def _encode_sync(
    model,
    texts: list[str],
) -> list[list[float]]:
    """
    Perform CPU model inference.

    The model is already initialized before this function
    is executed in the worker thread.
    """

    if not texts:
        return []

    with _embedding_inference_lock:

        vectors = model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    return vectors.tolist()


# ============================================================
# EMBED MULTIPLE TEXTS
# ============================================================

async def embed_texts(
    texts: list[str],
) -> list[list[float]]:

    if not texts:
        return []

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Initialize/import PyTorch model here BEFORE
    # asyncio.to_thread().
    # --------------------------------------------------------

    model = get_embedding_model()

    vectors = await asyncio.to_thread(
        _encode_sync,
        model,
        texts,
    )

    return vectors


# ============================================================
# EMBED SINGLE TEXT
# ============================================================

async def embed_text(
    text: str,
) -> list[float]:

    text = text.strip()

    if not text:

        raise ValueError(
            "Embedding text cannot be empty."
        )

    vectors = await embed_texts(
        [text]
    )

    if not vectors:

        raise RuntimeError(
            "Embedding model returned no vector."
        )

    vector = vectors[0]

    if (
        len(vector)
        !=
        settings.embedding_dimension
    ):

        raise RuntimeError(
            "Embedding dimension mismatch. "
            f"Expected "
            f"{settings.embedding_dimension}, "
            f"received {len(vector)}."
        )

    return vector