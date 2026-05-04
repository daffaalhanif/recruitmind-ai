"""Setup logging terpusat untuk RecruitMind AI."""

import logging
import os

_LOG_FILE = "logs/app.log"


def setup_logging() -> None:
    """Konfigurasi root logger untuk menulis ke logs/app.log.

    Cek apakah FileHandler ke app.log sudah ada sebelum menambahkan
    yang baru, agar tidak duplikat saat Streamlit re-run.
    """
    os.makedirs("logs", exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Cek apakah sudah ada FileHandler ke file yang sama
    for handler in root.handlers:
        if isinstance(handler, logging.FileHandler):
            if handler.baseFilename.endswith("app.log"):
                return

    file_handler = logging.FileHandler(_LOG_FILE)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    root.addHandler(file_handler)