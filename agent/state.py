"""Definisi AgentState untuk RecruitMind AI.

State adalah satu-satunya sumber kebenaran yang dibagi oleh semua node.
Tidak ada node yang berkomunikasi langsung satu sama lain. Semua
komunikasi terjadi melalui baca dan tulis ke state.

Perubahan pada struktur ini berdampak ke seluruh sistem karena
semua node bergantung pada field-field yang didefinisikan di sini.
"""

from typing import Annotated, Optional

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """State tunggal yang dibaca dan ditulis oleh semua node dalam graph.

    Attributes:
        messages: Riwayat percakapan lengkap. Menggunakan reducer
            add_messages sehingga setiap node cukup append pesan baru,
            tidak perlu overwrite seluruh list. Yang dikirim ke LLM
            hanya 10 pesan terakhir (sliding window), tapi full history
            tetap tersimpan di checkpointer.
        intent: Hasil klasifikasi Router Node. Nilai yang valid:
            RAG, SQL, EVALUATOR, GENERATOR, CLARIFY, MULTI_STEP.
            Di-overwrite setiap invocation baru.
        clarification_type: Jenis klarifikasi yang dideteksi Router.
            Nilai yang valid: missing_candidates, no_prior_search,
            ambiguous_comparison, search_overwrite_confirmation,
            unclear_intent. None jika tidak ada klarifikasi atau
            jika butuh LLM fallback di Clarification Node.
        clarification_context: Reasoning Router tentang mengapa query
            ambigu. Diisi hanya jika clarification_type adalah None
            dan Clarification Node perlu LLM fallback.
        task_queue: Antrian task untuk multi-step query. Dikonsumsi
            satu per satu oleh conditional edge setelah setiap node
            selesai. Default list kosong.
        processed_query: Query yang sudah diproses Router untuk dipakai
            RAG Node saat embedding. Berisi query asli untuk input
            singkat, atau key requirements hasil ekstraksi untuk input
            yang terdeteksi sebagai Job Description. Di-set setiap
            kali intent RAG terdeteksi.
        candidate_refs: Referensi kandidat yang disebutkan user secara
            eksplisit dalam bentuk angka, contoh ["1", "3"]. Di-set
            oleh Router, dibaca oleh Evaluator dan Generator untuk
            resolve kandidat dari active_candidates berdasarkan index.
        active_candidates: List kandidat aktif dari pencarian terakhir.
            Di-overwrite setiap pencarian baru dengan konfirmasi.
            Kosong hanya saat sesi baru dibuat.
        sql_result: Hasil eksekusi SQL Node. Di-overwrite setiap SQL
            Node jalan. Dipakai untuk LangFuse metadata dan logging.
        session_title: Judul sesi yang di-generate LLM dari query
            pertama. Di-set sekali pada invocation pertama sesi baru,
            tidak pernah berubah setelah itu.
    """

    messages: Annotated[list, add_messages]
    intent: Optional[str]
    clarification_type: Optional[str]
    clarification_context: Optional[str]
    task_queue: list[dict]
    processed_query: Optional[str]
    candidate_refs: list[str]
    active_candidates: list[dict]
    sql_result: Optional[dict]
    session_title: Optional[str]