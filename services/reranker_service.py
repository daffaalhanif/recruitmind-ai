"""Service layer untuk reranking di RecruitMind AI.

Membungkus FlashRank sehingga node-node agent tidak perlu tahu
detail implementasi cross-encoder atau format input FlashRank.

Ranker diinisialisasi sekali saat modul diimport karena proses loading
model weights cukup berat. Di-reuse untuk semua call berikutnya.
"""

from flashrank import Ranker, RerankRequest

from config import TOP_K_RERANK

# Ranker diinisialisasi sekali, model weights di-load saat import
_ranker = Ranker(model_name="rank-T5-flan")


def rerank(query: str, candidates: list[dict], top_k: int = TOP_K_RERANK) -> list[dict]:
    """Urutkan ulang kandidat menggunakan cross-encoder model.

    Cross-encoder membaca pasangan (query, dokumen) secara bersamaan
    sehingga menghasilkan skor relevansi yang lebih akurat dibanding
    cosine similarity dari tahap embedding. Urutan kandidat bisa berubah
    signifikan setelah reranking.

    Input candidates tidak dimodifikasi. Dict yang dikembalikan adalah
    object baru yang berisi semua field original ditambah rerank_score.

    Field 'preview' dari candidates dipakai sebagai teks dokumen untuk
    reranking karena di tahap ini resume_summary lengkap belum diambil
    dari SQLite. Preview 200 karakter sudah cukup untuk cross-encoder
    menilai relevansi.

    Args:
        query: Query user yang dipakai untuk menilai relevansi.
        candidates: List dict kandidat dari qdrant_service.search().
            Setiap dict harus punya field 'id' dan 'preview'.
        top_k: Jumlah kandidat yang dikembalikan setelah reranking.
            Default TOP_K_RERANK (5) dari config.

    Returns:
        List dict kandidat, diurutkan dari rerank_score tertinggi,
        masing-masing berisi semua field original ditambah:
            - rerank_score (float): Skor dari cross-encoder model.
        List kosong jika candidates kosong.
    """
    if not candidates:
        return []

    # Buat lookup by ID untuk merge score kembali ke candidate dict setelah reranking
    id_to_candidate = {c["id"]: c for c in candidates}

    passages = [
        {"id": c["id"], "text": c.get("resume_summary") or c["preview"]}
        for c in candidates
    ]

    request = RerankRequest(query=query, passages=passages)
    results = _ranker.rerank(request)

    reranked = []
    for result in results[:top_k]:
        candidate_id = result["id"]
        original = id_to_candidate[candidate_id]
        reranked.append(
            {
                **original,
                "rerank_score": float(result["score"]),
            }
        )

    return reranked