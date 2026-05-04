"""Service layer untuk operasi Qdrant di RecruitMind AI.

Membungkus Qdrant client sehingga node-node agent tidak perlu tahu
nama collection, format payload, atau detail API Qdrant.

Client diinisialisasi di dalam fungsi untuk konsistensi dengan pattern
keseluruhan codebase: st.secrets hanya diakses saat runtime Streamlit,
bukan saat module diimport.
"""

import streamlit as st
from qdrant_client import QdrantClient

from config import COLLECTION_NAME, TOP_K_RETRIEVAL


def search(query_vector: list[float], top_k: int = TOP_K_RETRIEVAL) -> list[dict]:
    """Cari kandidat paling relevan berdasarkan kedekatan vector.

    Mengambil top_k kandidat dari Qdrant menggunakan cosine similarity,
    lalu mengembalikan data dari payload yang tersimpan bersama vector.
    Hasil ini menjadi input untuk reranker_service sebelum masuk ke
    active_candidates di state.

    Payload yang disimpan di Qdrant per vector: id, category,
    current_position, preview (200 karakter pertama resume_summary).
    Field lain seperti resume_summary lengkap dan top_skills diambil
    dari SQLite setelah reranking, sehingga SQLite fetch hanya
    dilakukan untuk 5 kandidat final, bukan 15.

    Args:
        query_vector: Vector embedding dari query user, 1536 dimensi.
        top_k: Jumlah kandidat yang diambil sebelum reranking.
            Default TOP_K_RETRIEVAL (15) dari config.

    Returns:
        List dict kandidat, diurutkan dari skor tertinggi, masing-masing berisi:
            - id (str): ID resume, sama dengan primary key di SQLite.
            - category (str): Kategori pekerjaan dari dataset.
            - current_position (str atau None): Job title terakhir.
            - preview (str): 200 karakter pertama resume_summary.
            - qdrant_score (float): Skor cosine similarity mentah,
                dipakai untuk monitoring, tidak ditampilkan ke user.

    Raises:
        qdrant_client.http.exceptions.UnexpectedResponse: Jika terjadi
            error dari API Qdrant.
    """
    # Client diinisialisasi di dalam fungsi karena st.secrets hanya
    # tersedia saat runtime Streamlit, bukan saat module diimport
    client = QdrantClient(
        url=st.secrets["QDRANT_URL"],
        api_key=st.secrets["QDRANT_API_KEY"],
    )

    hits = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    ).points

    hasil = []
    for hit in hits:
        payload = hit.payload or {}
        hasil.append(
            {
                "id": payload.get("id", str(hit.id)),
                "category": payload.get("category", ""),
                "current_position": payload.get("current_position"),
                "preview": payload.get("preview", ""),
                "qdrant_score": float(hit.score),
            }
        )

    return hasil