import uuid
from urllib.parse import urlparse

import numpy as np
import psycopg

from pgvector.psycopg import (
    register_vector_async,
)

from psycopg.rows import (
    dict_row,
)

from psycopg.types.json import (
    Jsonb,
)

from psycopg_pool import (
    AsyncConnectionPool,
)

from app.config import settings


# ============================================================
# GLOBAL POOL
# ============================================================

pool: AsyncConnectionPool | None = None


# ============================================================
# POOL CONNECTION CONFIGURATION
# ============================================================

async def configure_connection(
    connection: psycopg.AsyncConnection,
) -> None:

    try:

        print(
            "[DB POOL] Registering pgvector..."
        )

        await register_vector_async(
            connection
        )

        # pgvector type discovery may create a transaction.
        # Pool configuration must finish with the
        # connection in an idle state.
        await connection.commit()

        print(
            "[DB POOL] pgvector registered."
        )

    except Exception as exc:

        print(
            "[DB POOL] Configuration failed:"
        )

        print(
            repr(exc)
        )

        try:

            await connection.rollback()

        except Exception:

            pass

        raise


# ============================================================
# GET POOL
# ============================================================

def get_pool() -> AsyncConnectionPool:

    if pool is None:

        raise RuntimeError(
            "PostgreSQL connection pool "
            "is not initialized."
        )

    if pool.closed:

        raise RuntimeError(
            "PostgreSQL connection pool "
            "is closed."
        )

    return pool


# ============================================================
# INITIALIZE POSTGRES
# ============================================================

async def init_db() -> None:

    global pool

    print(
        "Initializing PostgreSQL..."
    )

    parsed = urlparse(
        settings.database_url
    )

    print(
        "[DB] Host:",
        parsed.hostname
    )

    print(
        "[DB] Port:",
        parsed.port
    )

    print(
        "[DB] Database:",
        parsed.path.lstrip("/")
    )


    # ========================================================
    # BOOTSTRAP
    # ========================================================

    bootstrap = None

    try:

        print(
            "Creating PostgreSQL "
            "bootstrap connection..."
        )

        bootstrap = (
            await psycopg.AsyncConnection.connect(
                settings.database_url,
                connect_timeout=10,
            )
        )

        print(
            "PostgreSQL bootstrap "
            "connection established."
        )

        await bootstrap.execute(
            """
            CREATE EXTENSION
            IF NOT EXISTS vector;
            """
        )

        await bootstrap.commit()

        print(
            "pgvector extension ready."
        )

    except Exception as exc:

        print(
            "PostgreSQL bootstrap failed:"
        )

        print(
            repr(exc)
        )

        raise

    finally:

        if bootstrap is not None:

            try:

                await bootstrap.close()

            except Exception:

                pass


    # ========================================================
    # CREATE ASYNC POOL
    # ========================================================

    try:

        if pool is not None:

            try:

                await pool.close()

            except Exception:

                pass

            pool = None


        pool = AsyncConnectionPool(
            conninfo=settings.database_url,

            min_size=1,
            max_size=10,

            open=False,

            configure=
                configure_connection,

            kwargs={
                "row_factory":
                    dict_row,

                "connect_timeout":
                    10,
            },

            timeout=30,

            reconnect_timeout=15,

            name=
                "research-db-pool",
        )


        print(
            "Opening PostgreSQL async "
            "connection pool..."
        )

        await pool.open()

        print(
            "Waiting for PostgreSQL pool..."
        )

        await pool.wait(
            timeout=30
        )

        print(
            "PostgreSQL async "
            "connection pool ready."
        )

    except Exception as exc:

        print(
            "PostgreSQL connection "
            "pool failed:"
        )

        print(
            repr(exc)
        )

        if pool is not None:

            try:

                await pool.close()

            except Exception:

                pass

            pool = None

        raise


    # ========================================================
    # CREATE SCHEMA
    # ========================================================

    db_pool = get_pool()

    async with db_pool.connection() as conn:


        # ----------------------------------------------------
        # RESEARCH SESSIONS
        # ----------------------------------------------------

        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS
            research_sessions (

                id UUID PRIMARY KEY,

                tenant_id TEXT NOT NULL,

                query TEXT NOT NULL,

                status TEXT NOT NULL,

                state JSONB,

                created_at TIMESTAMPTZ
                    NOT NULL DEFAULT NOW(),

                updated_at TIMESTAMPTZ
                    NOT NULL DEFAULT NOW()
            );
            """
        )


        # ----------------------------------------------------
        # DOCUMENTS
        # ----------------------------------------------------

        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS
            documents (

                id UUID PRIMARY KEY,

                tenant_id TEXT NOT NULL,

                title TEXT,

                source_type TEXT,

                source_url TEXT,

                metadata JSONB,

                created_at TIMESTAMPTZ
                    NOT NULL DEFAULT NOW()
            );
            """
        )


        # ----------------------------------------------------
        # DOCUMENT CHUNKS
        # ----------------------------------------------------

        await conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS
            document_chunks (

                id UUID PRIMARY KEY,

                document_id UUID NOT NULL
                    REFERENCES documents(id)
                    ON DELETE CASCADE,

                tenant_id TEXT NOT NULL,

                content TEXT NOT NULL,

                metadata JSONB,

                embedding VECTOR(
                    {settings.embedding_dimension}
                ),

                created_at TIMESTAMPTZ
                    NOT NULL DEFAULT NOW()
            );
            """
        )


        # ----------------------------------------------------
        # HUMAN REVIEWS
        # ----------------------------------------------------

        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS
            human_reviews (

                id UUID PRIMARY KEY,

                session_id UUID NOT NULL,

                status TEXT
                    NOT NULL
                    DEFAULT 'pending',

                action TEXT,

                feedback TEXT,

                created_at TIMESTAMPTZ
                    NOT NULL DEFAULT NOW(),

                updated_at TIMESTAMPTZ
                    NOT NULL DEFAULT NOW()
            );
            """
        )


        # ----------------------------------------------------
        # JOBS
        # ----------------------------------------------------

        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS
            jobs (

                id UUID PRIMARY KEY,

                session_id UUID,

                job_type TEXT NOT NULL,

                status TEXT NOT NULL,

                payload JSONB,

                result JSONB,

                error TEXT,

                created_at TIMESTAMPTZ
                    NOT NULL DEFAULT NOW(),

                updated_at TIMESTAMPTZ
                    NOT NULL DEFAULT NOW()
            );
            """
        )


        # ----------------------------------------------------
        # INDEXES
        # ----------------------------------------------------

        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_document_chunks_tenant

            ON document_chunks (
                tenant_id
            );
            """
        )


        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_document_chunks_document

            ON document_chunks (
                document_id
            );
            """
        )


        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_documents_source_url

            ON documents (
                tenant_id,
                source_url
            );
            """
        )


        # ----------------------------------------------------
        # FULL TEXT SEARCH
        # ----------------------------------------------------

        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_document_chunks_fts

            ON document_chunks

            USING GIN (
                to_tsvector(
                    'english',
                    content
                )
            );
            """
        )


        # ----------------------------------------------------
        # VECTOR INDEX
        # ----------------------------------------------------

        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_document_chunks_vector

            ON document_chunks

            USING hnsw (
                embedding vector_cosine_ops
            );
            """
        )


        await conn.commit()


    print(
        "PostgreSQL tables "
        "initialized successfully."
    )


# ============================================================
# CLOSE DATABASE
# ============================================================

async def close_db() -> None:

    global pool

    if pool is None:
        return

    try:

        print(
            "[DB] Closing PostgreSQL pool..."
        )

        await pool.close()

        print(
            "[DB] PostgreSQL pool closed."
        )

    finally:

        pool = None


# ============================================================
# SESSION
# ============================================================

async def create_session(
    session_id: str,
    tenant_id: str,
    query: str,
) -> str:

    db_pool = get_pool()

    async with db_pool.connection() as conn:

        await conn.execute(
            """
            INSERT INTO research_sessions (
                id,
                tenant_id,
                query,
                status,
                state
            )

            VALUES (
                %s,
                %s,
                %s,
                %s,
                %s
            );
            """,
            (
                session_id,
                tenant_id,
                query,
                "created",
                Jsonb({}),
            ),
        )

        await conn.commit()

    return session_id


async def save_session_state(
    session_id: str,
    state: dict,
) -> None:

    db_pool = get_pool()

    async with db_pool.connection() as conn:

        await conn.execute(
            """
            UPDATE research_sessions

            SET
                state = %s,
                status = %s,
                updated_at = NOW()

            WHERE id = %s;
            """,
            (
                Jsonb(state),

                state.get(
                    "status",
                    "running",
                ),

                session_id,
            ),
        )

        await conn.commit()


async def get_session(
    session_id: str,
) -> dict | None:

    db_pool = get_pool()

    async with db_pool.connection() as conn:

        cursor = await conn.execute(
            """
            SELECT *

            FROM research_sessions

            WHERE id = %s

            LIMIT 1;
            """,
            (
                session_id,
            ),
        )

        return await cursor.fetchone()


# ============================================================
# DOCUMENTS
# ============================================================

async def find_document_by_url(
    tenant_id: str,
    url: str,
) -> dict | None:

    db_pool = get_pool()

    async with db_pool.connection() as conn:

        cursor = await conn.execute(
            """
            SELECT *

            FROM documents

            WHERE tenant_id = %s
            AND source_url = %s

            LIMIT 1;
            """,
            (
                tenant_id,
                url,
            ),
        )

        return await cursor.fetchone()


async def create_document(
    tenant_id: str,
    title: str,
    source_type: str,
    source_url: str | None,
    metadata: dict | None = None,
) -> str:

    document_id = str(
        uuid.uuid4()
    )

    db_pool = get_pool()

    async with db_pool.connection() as conn:

        await conn.execute(
            """
            INSERT INTO documents (
                id,
                tenant_id,
                title,
                source_type,
                source_url,
                metadata
            )

            VALUES (
                %s,
                %s,
                %s,
                %s,
                %s,
                %s
            );
            """,
            (
                document_id,
                tenant_id,
                title,
                source_type,
                source_url,
                Jsonb(
                    metadata or {}
                ),
            ),
        )

        await conn.commit()

    return document_id


async def delete_document(
    document_id: str,
) -> bool:

    db_pool = get_pool()

    async with db_pool.connection() as conn:

        cursor = await conn.execute(
            """
            DELETE FROM documents

            WHERE id = %s;
            """,
            (
                document_id,
            ),
        )

        await conn.commit()

        return (
            cursor.rowcount > 0
        )


async def list_documents(
    tenant_id: str = "default",
) -> list[dict]:

    db_pool = get_pool()

    async with db_pool.connection() as conn:

        cursor = await conn.execute(
            """
            SELECT

                d.id,
                d.tenant_id,
                d.title,
                d.source_type,
                d.source_url,
                d.metadata,
                d.created_at,

                COUNT(c.id)
                    AS chunk_count

            FROM documents AS d

            LEFT JOIN document_chunks AS c
                ON c.document_id = d.id

            WHERE d.tenant_id = %s

            GROUP BY d.id

            ORDER BY
                d.created_at DESC;
            """,
            (
                tenant_id,
            ),
        )

        rows = await cursor.fetchall()

    return list(
        rows
    )


# ============================================================
# CHUNKS
# ============================================================

async def insert_chunk(
    document_id: str,
    tenant_id: str,
    content: str,
    embedding: list[float],
    metadata: dict | None = None,
) -> str:

    chunk_id = str(
        uuid.uuid4()
    )

    vector = np.asarray(
        embedding,
        dtype=np.float32,
    )

    if (
        len(vector)
        !=
        settings.embedding_dimension
    ):

        raise ValueError(
            "Embedding dimension mismatch. "
            f"Expected "
            f"{settings.embedding_dimension}, "
            f"received {len(vector)}."
        )

    db_pool = get_pool()

    async with db_pool.connection() as conn:

        await conn.execute(
            """
            INSERT INTO document_chunks (
                id,
                document_id,
                tenant_id,
                content,
                metadata,
                embedding
            )

            VALUES (
                %s,
                %s,
                %s,
                %s,
                %s,
                %s
            );
            """,
            (
                chunk_id,
                document_id,
                tenant_id,
                content,
                Jsonb(
                    metadata or {}
                ),
                vector,
            ),
        )

        await conn.commit()

    return chunk_id


async def get_document_chunks(
    document_id: str,
) -> list[dict]:

    db_pool = get_pool()

    async with db_pool.connection() as conn:

        cursor = await conn.execute(
            """
            SELECT

                id,
                document_id,
                tenant_id,
                content,
                metadata,
                created_at

            FROM document_chunks

            WHERE document_id = %s

            ORDER BY
                created_at ASC;
            """,
            (
                document_id,
            ),
        )

        rows = await cursor.fetchall()

    return list(
        rows
    )


async def count_document_chunks(
    document_id: str,
) -> int:

    db_pool = get_pool()

    async with db_pool.connection() as conn:

        cursor = await conn.execute(
            """
            SELECT
                COUNT(*) AS count

            FROM document_chunks

            WHERE document_id = %s;
            """,
            (
                document_id,
            ),
        )

        row = await cursor.fetchone()

    if not row:
        return 0

    return int(
        row["count"]
    )


# ============================================================
# DENSE VECTOR SEARCH
# ============================================================

async def vector_search_db(
    tenant_id: str,
    embedding: list[float],
    limit: int,
) -> list[dict]:

    vector = np.asarray(
        embedding,
        dtype=np.float32,
    )

    if (
        len(vector)
        !=
        settings.embedding_dimension
    ):

        raise ValueError(
            "Query embedding dimension mismatch."
        )

    db_pool = get_pool()

    async with db_pool.connection() as conn:

        cursor = await conn.execute(
            """
            SELECT

                c.id,
                c.document_id,
                c.content,
                c.metadata,

                d.title,
                d.source_type,
                d.source_url,

                1 - (
                    c.embedding
                    <=>
                    %s
                ) AS vector_score

            FROM document_chunks AS c

            JOIN documents AS d
                ON d.id = c.document_id

            WHERE
                c.tenant_id = %s

            AND
                c.embedding IS NOT NULL

            ORDER BY
                c.embedding
                <=>
                %s

            LIMIT %s;
            """,
            (
                vector,
                tenant_id,
                vector,
                limit,
            ),
        )

        rows = await cursor.fetchall()

    return list(
        rows
    )


# ============================================================
# SPARSE FULL-TEXT SEARCH
# ============================================================

async def keyword_search_db(
    tenant_id: str,
    query: str,
    limit: int,
) -> list[dict]:

    db_pool = get_pool()

    async with db_pool.connection() as conn:

        cursor = await conn.execute(
            """
            SELECT

                c.id,
                c.document_id,
                c.content,
                c.metadata,

                d.title,
                d.source_type,
                d.source_url,

                ts_rank_cd(

                    to_tsvector(
                        'english',
                        c.content
                    ),

                    websearch_to_tsquery(
                        'english',
                        %s
                    )

                ) AS keyword_score

            FROM document_chunks AS c

            JOIN documents AS d
                ON d.id = c.document_id

            WHERE
                c.tenant_id = %s

            AND
                to_tsvector(
                    'english',
                    c.content
                )

                @@

                websearch_to_tsquery(
                    'english',
                    %s
                )

            ORDER BY
                keyword_score DESC

            LIMIT %s;
            """,
            (
                query,
                tenant_id,
                query,
                limit,
            ),
        )

        rows = await cursor.fetchall()

    return list(
        rows
    )


# ============================================================
# READ-ONLY SQL
# ============================================================

async def execute_read_only_sql(
    query: str,
) -> list[dict]:

    cleaned_query = (
        query
        .strip()
    )

    if not cleaned_query:

        raise ValueError(
            "SQL query cannot be empty."
        )

    first_word = (
        cleaned_query
        .split(
            maxsplit=1
        )[0]
        .upper()
    )

    allowed = {
        "SELECT",
        "WITH",
        "SHOW",
        "EXPLAIN",
    }

    if first_word not in allowed:

        raise ValueError(
            "Only read-only SQL "
            "queries are allowed."
        )

    db_pool = get_pool()

    async with db_pool.connection() as conn:

        cursor = await conn.execute(
            query
        )

        rows = await cursor.fetchall()

    return list(
        rows
    )


# ============================================================
# HUMAN REVIEW
# ============================================================

async def create_review(
    session_id: str,
) -> str:

    review_id = str(
        uuid.uuid4()
    )

    db_pool = get_pool()

    async with db_pool.connection() as conn:

        await conn.execute(
            """
            INSERT INTO human_reviews (
                id,
                session_id,
                status
            )

            VALUES (
                %s,
                %s,
                'pending'
            );
            """,
            (
                review_id,
                session_id,
            ),
        )

        await conn.commit()

    return review_id


async def complete_review(
    session_id: str,
    action: str,
    feedback: str | None,
) -> None:

    db_pool = get_pool()

    async with db_pool.connection() as conn:

        await conn.execute(
            """
            UPDATE human_reviews

            SET
                status = 'completed',
                action = %s,
                feedback = %s,
                updated_at = NOW()

            WHERE session_id = %s
            AND status = 'pending';
            """,
            (
                action,
                feedback,
                session_id,
            ),
        )

        await conn.commit()


# ============================================================
# JOBS
# ============================================================

async def create_job(
    job_type: str,
    payload: dict,
    session_id: str | None = None,
) -> str:

    job_id = str(
        uuid.uuid4()
    )

    db_pool = get_pool()

    async with db_pool.connection() as conn:

        await conn.execute(
            """
            INSERT INTO jobs (
                id,
                session_id,
                job_type,
                status,
                payload
            )

            VALUES (
                %s,
                %s,
                %s,
                %s,
                %s
            );
            """,
            (
                job_id,
                session_id,
                job_type,
                "queued",
                Jsonb(
                    payload or {}
                ),
            ),
        )

        await conn.commit()

    return job_id


async def update_job(
    job_id: str,
    status: str,
    result: dict | None = None,
    error: str | None = None,
) -> None:

    result_json = (
        Jsonb(result)
        if result is not None
        else None
    )

    db_pool = get_pool()

    async with db_pool.connection() as conn:

        await conn.execute(
            """
            UPDATE jobs

            SET
                status = %s,
                result = %s,
                error = %s,
                updated_at = NOW()

            WHERE id = %s;
            """,
            (
                status,
                result_json,
                error,
                job_id,
            ),
        )

        await conn.commit()


async def get_job(
    job_id: str,
) -> dict | None:

    db_pool = get_pool()

    async with db_pool.connection() as conn:

        cursor = await conn.execute(
            """
            SELECT *

            FROM jobs

            WHERE id = %s

            LIMIT 1;
            """,
            (
                job_id,
            ),
        )

        return await cursor.fetchone()