import logging
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from config import (
    COLLECTION_NAME,
    VECTOR_SIZE,
    EMBEDDING_MODEL,
    INGEST_BATCH_SIZE,
    PAYLOAD_PREVIEW_LENGTH,
)

load_dotenv()

logging.basicConfig(
    filename="logs/pipeline.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_embedding(client: OpenAI, text: str) -> list[float]:
    """Buat vector embedding dari teks menggunakan text-embedding-3-small.

    Args:
        client: Instance OpenAI client.
        text: Teks yang akan di-embed.

    Returns:
        List float berisi vector embedding 1536 dimensi.
    """
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding


def create_collection_if_not_exists(qdrant: QdrantClient) -> None:
    """Buat Qdrant collection jika belum ada.

    Idempoten: tidak melakukan apapun jika collection sudah ada,
    sehingga script aman dijalankan ulang tanpa menghapus data.

    Args:
        qdrant: Instance QdrantClient yang sudah terkoneksi.
    """
    if qdrant.collection_exists(COLLECTION_NAME):
        logger.info(f"Collection '{COLLECTION_NAME}' sudah ada, skip pembuatan.")
        return

    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE,
        ),
    )
    logger.info(f"Collection '{COLLECTION_NAME}' berhasil dibuat.")


def build_payload(row: object) -> dict:
    """Buat payload untuk satu vector point di Qdrant.

    Payload berisi metadata yang ditampilkan di candidate card
    tanpa perlu query tambahan ke SQLite.

    Args:
        row: Satu baris dari DataFrame hasil itertuples().

    Returns:
        Dict berisi id, category, current_position, dan preview.
    """
    # Preview 200 karakter pertama resume_summary untuk candidate card di UI
    preview = str(row.resume_summary)[:PAYLOAD_PREVIEW_LENGTH]

    return {
        "id": str(row.id),
        "category": row.category,
        "current_position": (
            None if pd.isna(row.current_position) else row.current_position
        ),
        "preview": preview,
    }


def main() -> None:
    """Jalankan pipeline ingestion resume ke Qdrant Cloud.

    Membaca resumes_structured.csv, membuat embedding untuk setiap
    resume_summary, lalu mengupload vector dan payload ke Qdrant.
    Script ini idempoten karena hanya membuat collection jika belum ada.
    """
    logger.info("Script 04 - Vector DB Ingestion dimulai")

    input_path = Path("data/processed/resumes_structured.csv")

    if not input_path.exists():
        raise FileNotFoundError(f"Input tidak ditemukan: {input_path}")

    df = pd.read_csv(input_path)
    total = len(df)
    logger.info(f"Membaca {total} resume dari {input_path}")

    openai_api_key = os.getenv("OPENAI_API_KEY")
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    if not all([openai_api_key, qdrant_url, qdrant_api_key]):
        raise ValueError("OPENAI_API_KEY, QDRANT_URL, atau QDRANT_API_KEY tidak ditemukan.")

    openai_client = OpenAI(api_key=openai_api_key)
    qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    create_collection_if_not_exists(qdrant_client)

    points = []
    failed = 0

    for i, row in enumerate(df.itertuples(), start=1):
        try:
            embedding = get_embedding(openai_client, row.resume_summary)
            payload = build_payload(row)

            points.append(
                PointStruct(
                    # ID di Qdrant harus integer atau UUID, konversi dari string
                    id=i,
                    vector=embedding,
                    payload=payload,
                )
            )
        except Exception as e:
            logger.warning(f"Gagal embed resume ID {row.id}: {e}")
            failed += 1

        # Upload per batch untuk menghindari payload terlalu besar dalam satu request
        if len(points) == INGEST_BATCH_SIZE:
            qdrant_client.upsert(
                collection_name=COLLECTION_NAME,
                points=points,
            )
            logger.info(f"Progress: {i}/{total} resume diupload")
            points = []

    # Upload sisa points yang belum mencapai BATCH_SIZE
    if points:
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
        )
        logger.info(f"Progress: {total}/{total} resume diupload")

    success = total - failed
    logger.info(f"Berhasil: {success} | Gagal: {failed}")
    logger.info("Script 04 - Vector DB Ingestion selesai")
    print(f"[04_ingest_vectordb] Selesai. Berhasil: {success} | Gagal: {failed}")


if __name__ == "__main__":
    main()