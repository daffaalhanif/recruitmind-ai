import json
import logging
import os
import sqlite3
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, field_validator

from config import (
    LLM_MODEL,
    LLM_TEMPERATURE,
    EXTRACTION_BATCH_SIZE,
    EXTRACTION_BATCH_DELAY,
)


load_dotenv()

logging.basicConfig(
    filename="logs/pipeline.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Error per resume dicatat terpisah agar mudah di-audit tanpa noise log utama
error_logger = logging.getLogger("extraction_errors")
error_handler = logging.FileHandler("logs/extraction_errors.log")
error_handler.setFormatter(
    logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
)
error_logger.addHandler(error_handler)
error_logger.setLevel(logging.ERROR)


class ResumeExtraction(BaseModel):
    """Schema hasil ekstraksi LLM untuk satu resume.

    Attributes:
        current_position: Job title eksplisit dari resume, bukan placeholder.
        years_experience: Tahun pengalaman, hanya jika ada angka eksplisit.
        education_level: Tingkat pendidikan terakhir.
        top_skills: Daftar skill eksplisit, maksimal 8 item.
        resume_summary: Ringkasan 100-300 kata berdasarkan isi resume.
    """

    current_position: Optional[str] = Field(
        default=None,
        description="Job title eksplisit dari resume. None jika tidak tersedia.",
    )
    years_experience: Optional[int] = Field(
        default=None,
        description="Tahun pengalaman kerja. Hanya isi jika ada angka eksplisit. Range valid 0-60.",
        ge=0,
        le=60,
    )
    education_level: Optional[str] = Field(
        default=None,
        description="Tingkat pendidikan tertinggi. Nilai valid: SMA, D3, S1, S2, S3.",
    )
    top_skills: list[str] = Field(
        default_factory=list,
        description="Daftar skill eksplisit dari resume. Maksimal 8 item.",
    )
    resume_summary: str = Field(
        description="Ringkasan resume dalam 100-300 kata. Wajib ada. Jangan mengarang.",
    )

    @field_validator("education_level")
    @classmethod
    def validate_education_level(cls, v: Optional[str]) -> Optional[str]:
        """Validasi education_level hanya berisi nilai enum yang valid.

        LLM kadang mengembalikan variasi seperti 'Bachelor', 'Sarjana',
        string 'None', atau sistem pendidikan asing seperti 'GED', 'AAS'.
        Semua nilai di luar enum di-set None agar resume tetap diproses.

        Args:
            v: Nilai education_level dari LLM.

        Returns:
            Nilai yang sudah divalidasi atau None.
        """
        valid_values = {"SMA", "D3", "S1", "S2", "S3"}
        # Konversi string 'None' dan nilai di luar enum ke None Python
        if v is None or v == "None" or v not in valid_values:
            return None
        return v

    @field_validator("top_skills")
    @classmethod
    def validate_top_skills(cls, v: list[str]) -> list[str]:
        """Batasi top_skills maksimal 8 item.

        Args:
            v: List skill dari LLM.

        Returns:
            List skill yang sudah dibatasi maksimal 8 item.
        """
        return v[:8]

    @field_validator("resume_summary")
    @classmethod
    def validate_resume_summary(cls, v: str) -> str:
        """Validasi resume_summary tidak kosong.

        Args:
            v: Nilai resume_summary dari LLM.

        Returns:
            Nilai resume_summary yang sudah di-strip.

        Raises:
            ValueError: Jika resume_summary kosong atau hanya whitespace.
        """
        if not v or not v.strip():
            raise ValueError("resume_summary tidak boleh kosong")
        return v.strip()


def extract_single_resume(
    llm: ChatOpenAI,
    resume_id: int,
    resume_text: str,
) -> Optional[dict]:
    """Ekstrak informasi terstruktur dari satu resume menggunakan LLM.

    Args:
        llm: Instance ChatOpenAI dengan structured output.
        resume_id: ID resume untuk keperluan logging error.
        resume_text: Teks resume yang sudah dibersihkan.

    Returns:
        Dict berisi hasil ekstraksi, atau None jika gagal.
    """
    prompt = f"""Kamu adalah sistem ekstraksi informasi resume yang akurat.
Ekstrak informasi berikut dari resume di bawah ini.

ATURAN PENTING:
- current_position: job title eksplisit yang tercantum di resume. None jika tidak ada.
- years_experience: hanya isi jika ada angka eksplisit tahun pengalaman. Jangan estimasi.
- education_level: pilih dari SMA, D3, S1, S2, S3 saja. None jika tidak jelas.
- top_skills: maksimal 8 skill eksplisit dari bagian Skills atau pengalaman kerja.
- resume_summary: ringkasan 100-300 KATA berdasarkan isi resume. Jangan mengarang.

RESUME:
{resume_text}"""

    try:
        result = llm.invoke(prompt)
        return result.model_dump()
    except Exception as e:
        error_logger.error(
            f"ID {resume_id} | Error: {type(e).__name__} | {str(e)}"
        )
        return None


def log_pipeline_run(status: str, summary: str) -> None:
    """Catat metadata run ke tabel pipeline_runs di SQLite.

    Dipanggil di akhir script. Di-skip tanpa error jika database
    belum ada karena script 03 belum dijalankan.

    Args:
        status: Status akhir run, nilai valid adalah 'success' atau 'partial'.
        summary: Ringkasan hasil run dalam satu string.
    """
    db_path = Path("data/recruitmind.db")

    # Database baru ada setelah script 03 dijalankan
    if not db_path.exists():
        logger.info("recruitmind.db belum ada, skip pipeline_runs logging.")
        return

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO pipeline_runs (run_id, run_at, script, status, summary)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                datetime.now(),
                "02_extract.py",
                status,
                summary,
            ),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"Gagal insert ke pipeline_runs: {e}")


def main(test_mode: bool = False) -> None:
    """Jalankan pipeline LLM extraction untuk semua resume.

    Membaca resumes_clean.csv, mengirim setiap resume ke gpt-4o-mini
    untuk ekstraksi field terstruktur, lalu menyimpan hasilnya ke
    resumes_structured.csv. Resume yang sudah berhasil diproses
    sebelumnya di-skip agar script aman dijalankan ulang.

    Args:
        test_mode: Jika True, hanya proses 10 resume pertama untuk verifikasi.
    """
    logger.info("Script 02 - LLM Extraction dimulai")

    input_path = Path("data/processed/resumes_clean.csv")
    output_path = Path("data/processed/resumes_structured.csv")

    df = pd.read_csv(input_path)

    if test_mode:
        df = df.head(10)
        logger.info("TEST MODE: hanya memproses 10 resume pertama")

    # Muat ID yang sudah berhasil diproses agar tidak diproses ulang
    already_done_ids = set()
    if output_path.exists():
        df_existing = pd.read_csv(output_path)
        already_done_ids = set(df_existing["id"].astype(str))
        logger.info(f"Resume yang sudah diproses sebelumnya: {len(already_done_ids)}")

    df = df[~df["ID"].astype(str).isin(already_done_ids)]
    logger.info(f"Resume yang akan diproses: {len(df)}")

    if len(df) == 0:
        logger.info("Semua resume sudah diproses, tidak ada yang perlu dijalankan.")
        print("[02_extract] Semua resume sudah diproses.")
        return

    total = len(df)
    logger.info(f"Total resume yang akan diproses: {total}")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY tidak ditemukan di environment.")

    # temperature=0 untuk hasil ekstraksi yang deterministik dan konsisten
    llm = ChatOpenAI(
        model=LLM_MODEL,
        api_key=api_key,
        temperature=LLM_TEMPERATURE,
    ).with_structured_output(ResumeExtraction)

    results = []
    failed_ids = []

    for i, row in enumerate(df.itertuples(), start=1):
        result = extract_single_resume(llm, row.ID, row.Resume_str)

        if result is not None:
            # Gabungkan hasil ekstraksi dengan metadata asli dari dataset
            result["id"] = str(row.ID)
            result["category"] = row.Category
            # Simpan sebagai JSON string agar kompatibel dengan format CSV
            result["top_skills"] = json.dumps(result["top_skills"])
            results.append(result)
        else:
            failed_ids.append(row.ID)

        if i % EXTRACTION_BATCH_SIZE == 0 or i == total:
            logger.info(f"Progress: {i}/{total} resume diproses")

        if i % EXTRACTION_BATCH_SIZE == 0 and i < total:
            time.sleep(EXTRACTION_BATCH_DELAY)

    success = len(results)
    failed = len(failed_ids)

    if failed_ids:
        logger.warning(f"Resume yang gagal diekstrak: {failed_ids}")

    if results:
        df_new = pd.DataFrame(results)
        column_order = [
            "id",
            "category",
            "current_position",
            "years_experience",
            "education_level",
            "top_skills",
            "resume_summary",
        ]
        df_new = df_new[column_order]

        # Gabungkan dengan hasil sebelumnya agar output tetap lengkap
        if output_path.exists() and not test_mode:
            df_existing = pd.read_csv(output_path)
            df_output = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_output = df_new

        df_output.to_csv(output_path, index=False)
        logger.info(f"Output disimpan ke {output_path}, total baris: {len(df_output)}")

    # Quality report untuk audit kualitas ekstraksi sebelum ingest ke database
    logger.info("--- Extraction Quality Report ---")
    logger.info(f"Total diproses: {total}")
    logger.info(f"Berhasil: {success} | Gagal: {failed}")
    logger.info(f"Error rate: {failed / total * 100:.2f}%")

    if results:
        df_results = pd.DataFrame(results)
        for field in ["current_position", "years_experience", "education_level"]:
            filled = df_results[field].notna().sum()
            rate = filled / len(df_results) * 100
            logger.info(
                f"Field '{field}' completeness: {filled}/{len(df_results)} ({rate:.1f}%)"
            )
        # top_skills tersimpan sebagai JSON string, parse dulu sebelum hitung panjangnya
        avg_skills = df_results["top_skills"].apply(
            lambda x: len(json.loads(x))
        ).mean()
        logger.info(f"Rata-rata top_skills per resume: {avg_skills:.1f}")

    logger.info("--- End of Quality Report ---")

    summary = (
        f"Total: {total} | "
        f"Berhasil: {success} | "
        f"Gagal: {failed} | "
        f"Error rate: {failed / total * 100:.2f}%"
    )
    logger.info(f"Summary: {summary}")
    logger.info("Script 02 - LLM Extraction selesai")

    print(f"[02_extract] Selesai. {summary}")


if __name__ == "__main__":
    # Jalankan dengan argumen 'test' untuk memproses 10 resume pertama saja
    # Contoh: python pipeline/02_extract.py test
    test_mode = len(sys.argv) > 1 and sys.argv[1] == "test"
    main(test_mode=test_mode)