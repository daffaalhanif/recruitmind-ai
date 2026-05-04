"""Service layer untuk operasi SQLite di RecruitMind AI.

Semua query ke database melewati fungsi-fungsi di modul ini sehingga
node-node agent tidak perlu tahu detail koneksi, nama tabel, atau
format penyimpanan data (misalnya top_skills sebagai JSON string).

Setiap fungsi membuka dan menutup koneksinya sendiri menggunakan
context manager. Ini aman di model eksekusi Streamlit yang single-threaded
per session dan menghindari shared state antar invocation.
"""

import json
import sqlite3
from contextlib import contextmanager
from typing import Generator, Optional

from config import DB_PATH


# ---------------------------------------------------------------------------
# Manajemen koneksi
# ---------------------------------------------------------------------------

@contextmanager
def _get_connection() -> Generator[sqlite3.Connection, None, None]:
    """Buka koneksi SQLite, yield, lalu tutup otomatis.

    Menggunakan row_factory = sqlite3.Row agar baris hasil query
    bisa diakses seperti dict, tidak perlu indexing posisi kolom.

    Yields:
        Koneksi SQLite yang aktif.

    Raises:
        sqlite3.Error: Jika file database tidak bisa dibuka.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Query resume
# ---------------------------------------------------------------------------

def get_resume_by_id(resume_id: str) -> Optional[dict]:
    """Ambil satu resume beserta skillnya berdasarkan primary key.

    Args:
        resume_id: ID resume, sama dengan kolom 'id' di tabel resumes
            dan point ID di Qdrant.

    Returns:
        Dict berisi semua kolom resume dengan top_skills sudah berupa
        list Python (bukan string JSON), atau None jika ID tidak ada.
    """
    with _get_connection() as conn:
        row = conn.execute(
            """
            SELECT id, category, current_position, years_experience,
                   education_level, top_skills, resume_summary
            FROM resumes
            WHERE id = ?
            """,
            (resume_id,),
        ).fetchone()

    if row is None:
        return None

    hasil = dict(row)
    # top_skills disimpan sebagai JSON array string, parse ke list Python
    # agar caller selalu terima list, bukan string mentah
    hasil["top_skills"] = json.loads(hasil["top_skills"] or "[]")
    return hasil


def get_resumes_by_ids(resume_ids: list[str]) -> list[dict]:
    """Ambil beberapa resume sekaligus dalam satu query.

    Urutan hasil dikembalikan sesuai urutan resume_ids agar caller
    bisa mengkorelasikan hasil dengan index di active_candidates
    (dibutuhkan saat resolve candidate_refs di Evaluator dan Generator).

    Args:
        resume_ids: List ID resume yang ingin diambil.

    Returns:
        List dict resume sesuai urutan resume_ids. ID yang tidak
        ditemukan di database di-skip tanpa error.
    """
    if not resume_ids:
        return []

    placeholders = ",".join("?" * len(resume_ids))
    with _get_connection() as conn:
        rows = conn.execute(
            f"""
            SELECT id, category, current_position, years_experience,
                   education_level, top_skills, resume_summary
            FROM resumes
            WHERE id IN ({placeholders})
            """,
            resume_ids,
        ).fetchall()

    # Buat lookup by ID agar hasil bisa dikembalikan sesuai urutan input
    lookup = {}
    for row in rows:
        record = dict(row)
        record["top_skills"] = json.loads(record["top_skills"] or "[]")
        lookup[record["id"]] = record

    return [lookup[rid] for rid in resume_ids if rid in lookup]


# ---------------------------------------------------------------------------
# Query analytics
# ---------------------------------------------------------------------------

def execute_analytics_query(sql: str, params: tuple = ()) -> list[dict]:
    """Jalankan query analytics read-only dan kembalikan hasilnya sebagai list dict.

    Validasi keamanan SQL (blokir DROP, DELETE, dll.) ada di SQL Node,
    bukan di sini. Fungsi ini hanya bertanggung jawab pada eksekusi.

    Args:
        sql: Query SQL yang akan dijalankan. Harus berupa SELECT.
        params: Tuple parameter untuk bind ke query (opsional).

    Returns:
        List dict, satu dict per baris hasil. List kosong jika tidak ada hasil.

    Raises:
        sqlite3.Error: Jika terjadi error database (syntax error, dll.).
    """
    with _get_connection() as conn:
        rows = conn.execute(sql, params).fetchall()

    return [dict(row) for row in rows]


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

def save_session(thread_id: str, title: str) -> None:
    """Simpan satu record sesi baru ke tabel chat_sessions.

    Dipanggil tepat sekali per sesi, saat Response Node meng-generate
    session title di invocation pertama. thread_id harus unik karena
    merupakan primary key.

    Args:
        thread_id: UUID string, sama dengan thread_id di LangGraph.
        title: Judul yang di-generate LLM, maksimal 5 kata Bahasa Indonesia.

    Raises:
        sqlite3.IntegrityError: Jika thread_id sudah ada.
        sqlite3.Error: Untuk error database lainnya.
    """
    with _get_connection() as conn:
        conn.execute(
            "INSERT INTO chat_sessions (thread_id, title) VALUES (?, ?)",
            (thread_id, title),
        )
        conn.commit()


def get_all_sessions() -> list[dict]:
    """Ambil semua sesi chat, diurutkan dari yang terbaru.

    Digunakan sidebar Streamlit untuk menampilkan daftar sesi.

    Returns:
        List dict dengan key: thread_id, title, created_at.
    """
    with _get_connection() as conn:
        rows = conn.execute(
            "SELECT thread_id, title, created_at FROM chat_sessions ORDER BY created_at DESC"
        ).fetchall()

    return [dict(row) for row in rows]


def delete_session(thread_id: str) -> None:
    """Hapus record sesi dari tabel chat_sessions.

    Cleanup checkpoint LangGraph dari checkpoints.db dilakukan
    terpisah di layer Streamlit karena membutuhkan graph object,
    bukan hanya koneksi SQLite.

    Args:
        thread_id: UUID string dari sesi yang akan dihapus.

    Raises:
        sqlite3.Error: Jika terjadi error database.
    """
    with _get_connection() as conn:
        conn.execute(
            "DELETE FROM chat_sessions WHERE thread_id = ?",
            (thread_id,),
        )
        conn.commit()