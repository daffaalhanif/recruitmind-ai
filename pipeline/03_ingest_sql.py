import os
import sys

# Tambahkan root project ke sys.path agar import config bisa resolve
# saat script dijalankan sebagai subprocess dari app/main.py
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import json
import logging
import sqlite3
from pathlib import Path

import pandas as pd

from config import DB_PATH

logging.basicConfig(
    filename="logs/pipeline.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def create_tables(conn: sqlite3.Connection) -> None:
    """Buat semua tabel yang dibutuhkan aplikasi.

    Menggunakan IF NOT EXISTS agar script aman dijalankan
    berulang kali tanpa menghapus data yang sudah ada.

    Args:
        conn: Koneksi SQLite yang aktif.
    """
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS resumes (
            id TEXT PRIMARY KEY,
            category TEXT NOT NULL,
            current_position TEXT,
            years_experience INTEGER,
            education_level TEXT,
            top_skills TEXT NOT NULL DEFAULT '[]',
            resume_summary TEXT NOT NULL
        )
    """)

    # Tabel terpisah untuk operasi SQL per skill seperti GROUP BY dan INTERSECT
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS resume_skills (
            resume_id TEXT NOT NULL,
            skill TEXT NOT NULL,
            FOREIGN KEY (resume_id) REFERENCES resumes(id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_sessions (
            thread_id TEXT PRIMARY KEY,
            title TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pipeline_runs (
            run_id TEXT PRIMARY KEY,
            run_at TIMESTAMP,
            script TEXT,
            status TEXT,
            summary TEXT
        )
    """)

    # Index untuk mempercepat query analytics yang sering filter dan group by skill
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_resume_skills_skill
        ON resume_skills(skill)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_resume_skills_resume_id
        ON resume_skills(resume_id)
    """)

    conn.commit()
    logger.info("Semua tabel dan index berhasil dibuat.")


def populate_resumes(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    """Isi tabel resumes dari DataFrame hasil ekstraksi.

    Args:
        conn: Koneksi SQLite yang aktif.
        df: DataFrame dari resumes_structured.csv.

    Returns:
        Jumlah baris yang berhasil diinsert.
    """
    cursor = conn.cursor()
    inserted = 0

    for row in df.itertuples():
        try:
            cursor.execute(
                """
                INSERT OR IGNORE INTO resumes (
                    id, category, current_position,
                    years_experience, education_level,
                    top_skills, resume_summary
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(row.id),
                    row.category,
                    # None untuk field nullable jika pandas membaca sebagai NaN
                    None if pd.isna(row.current_position) else row.current_position,
                    None if pd.isna(row.years_experience) else int(row.years_experience),
                    None if pd.isna(row.education_level) else row.education_level,
                    row.top_skills,
                    row.resume_summary,
                ),
            )
            inserted += cursor.rowcount
        except Exception as e:
            logger.warning(f"Gagal insert resume ID {row.id}: {e}")

    conn.commit()
    return inserted


def populate_resume_skills(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    """Isi tabel resume_skills dari kolom top_skills di DataFrame.

    Setiap skill per resume menjadi satu baris tersendiri untuk
    memungkinkan operasi SQL per skill seperti GROUP BY dan INTERSECT.

    Args:
        conn: Koneksi SQLite yang aktif.
        df: DataFrame dari resumes_structured.csv.

    Returns:
        Jumlah baris yang berhasil diinsert.
    """
    cursor = conn.cursor()
    inserted = 0

    for row in df.itertuples():
        try:
            skills = json.loads(row.top_skills)
        except (json.JSONDecodeError, TypeError):
            # Skip resume dengan top_skills yang tidak bisa di-parse
            logger.warning(f"Gagal parse top_skills untuk ID {row.id}: {row.top_skills}")
            continue

        for skill in skills:
            if skill and skill.strip():
                cursor.execute(
                    "INSERT INTO resume_skills (resume_id, skill) VALUES (?, ?)",
                    (str(row.id), skill.strip()),
                )
                inserted += 1

    conn.commit()
    return inserted


def main() -> None:
    """Jalankan pipeline ingestion data ke SQLite.

    Membuat semua tabel, mengisi resumes dan resume_skills
    dari resumes_structured.csv. Script ini idempoten dan aman
    dijalankan berulang kali karena menggunakan INSERT OR IGNORE
    dan CREATE IF NOT EXISTS.
    """
    logger.info("Script 03 - SQL Ingestion dimulai")

    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = Path(ROOT_DIR) / "data/processed/resumes_structured.csv"
    db_path = Path(ROOT_DIR) / DB_PATH

    if not input_path.exists():
        raise FileNotFoundError(f"Input tidak ditemukan: {input_path}")

    df = pd.read_csv(input_path)
    logger.info(f"Membaca {len(df)} baris dari {input_path}")

    conn = sqlite3.connect(db_path)

    try:
        create_tables(conn)

        inserted_resumes = populate_resumes(conn, df)
        logger.info(f"Resumes diinsert: {inserted_resumes}")

        inserted_skills = populate_resume_skills(conn, df)
        logger.info(f"Resume skills diinsert: {inserted_skills}")

    finally:
        # Pastikan koneksi selalu ditutup meski ada error
        conn.close()

    logger.info(f"Database disimpan di {db_path}")
    logger.info("Script 03 - SQL Ingestion selesai")
    print(f"[03_ingest_sql] Selesai. Resumes: {inserted_resumes} | Skills: {inserted_skills}")


if __name__ == "__main__":
    main()