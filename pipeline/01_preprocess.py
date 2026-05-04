import pandas as pd
import re
import sqlite3
import logging
import uuid
from datetime import datetime
from pathlib import Path


logging.basicConfig(
    filename="logs/pipeline.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def strip_html(text: str) -> str:
    """Hapus semua HTML tags dari teks resume.

    Args:
        text: Teks mentah yang mungkin mengandung HTML tags.

    Returns:
        Teks bersih tanpa HTML tags, dengan spasi menggantikan tags.
    """
    # Ganti tag dengan spasi agar kata-kata di sekitar tag tidak menyatu
    return re.sub(r"<[^>]+>", " ", text)


def normalize_whitespace(text: str) -> str:
    """Normalisasi whitespace berlebih pada teks.

    Args:
        text: Teks yang akan dinormalisasi.

    Returns:
        Teks dengan whitespace yang sudah dinormalisasi.
    """
    # Tab tidak konsisten antar resume, samakan ke spasi
    text = text.replace("\t", " ")
    # Lebih dari dua newline berturut tidak menambah informasi struktural
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Spasi berlebih hasil strip HTML tags
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def clean_resume(text: str) -> str:
    """Jalankan pipeline pembersihan lengkap untuk satu teks resume.

    Args:
        text: Teks resume mentah.

    Returns:
        Teks resume yang sudah bersih dari HTML dan whitespace berlebih.
    """
    text = strip_html(text)
    text = normalize_whitespace(text)
    return text


def log_pipeline_run(status: str, summary: str) -> None:
    """Catat metadata run ke tabel pipeline_runs di SQLite.

    Dipanggil di akhir script. Di-skip tanpa error jika database
    belum ada karena script 03 belum dijalankan.

    Args:
        status: Status akhir run, nilai valid adalah 'success' atau 'failed'.
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
                "01_preprocess.py",
                status,
                summary,
            ),
        )
        conn.commit()
        conn.close()
        logger.info("Pipeline run tercatat di pipeline_runs.")
    except Exception as e:
        logger.warning(f"Gagal insert ke pipeline_runs: {e}")


def main() -> None:
    """Jalankan pipeline preprocessing dataset resume.

    Membaca dataset mentah, menghapus duplikat, membersihkan
    kolom Resume_str dari HTML dan whitespace berlebih, lalu
    menyimpan hasilnya ke data/processed/resumes_clean.csv.
    """
    logger.info("Script 01 - Preprocessing dimulai")

    input_path = Path("data/raw/resume_dataset.csv")
    output_path = Path("data/processed/resumes_clean.csv")

    logger.info(f"Membaca dataset dari {input_path}")
    df = pd.read_csv(input_path)
    total_input = len(df)
    logger.info(f"Total baris input: {total_input}")

    # ID adalah identifier unik per kandidat di seluruh sistem
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["ID"])
    duplicates_removed = before_dedup - len(df)
    logger.info(f"Duplikat dihapus: {duplicates_removed}")

    logger.info("Membersihkan kolom Resume_str...")
    df["Resume_str"] = df["Resume_str"].astype(str).apply(clean_resume)

    # Resume kosong setelah cleaning tidak bisa diproses LLM maupun di-embed
    before_empty_drop = len(df)
    df = df[df["Resume_str"].str.strip() != ""]
    empty_dropped = before_empty_drop - len(df)
    if empty_dropped > 0:
        logger.info(f"Baris dengan Resume_str kosong di-drop: {empty_dropped}")

    # Resume_html hanya dipakai untuk rendering web, bukan untuk NLP pipeline
    df = df.drop(columns=["Resume_html"])

    # Pastikan folder processed ada sebelum menulis output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    total_output = len(df)
    logger.info(f"Output disimpan ke {output_path}, total baris: {total_output}")

    summary = (
        f"Input: {total_input} baris | "
        f"Duplikat dihapus: {duplicates_removed} | "
        f"Resume kosong di-drop: {empty_dropped} | "
        f"Output: {total_output} baris"
    )
    logger.info(f"Summary: {summary}")
    logger.info("Script 01 - Preprocessing selesai")

    log_pipeline_run("success", summary)
    print(f"[01_preprocess] Selesai. {summary}")


if __name__ == "__main__":
    main()