"""Entry point aplikasi Streamlit RecruitMind AI.

Menangani initialization flow, graph caching, session state,
dan orkestrasi antara sidebar dan area chat utama.
"""

import os
import sys

# Tambahkan root project ke sys.path agar semua import bisa resolve
# di Streamlit Community Cloud yang tidak punya .envrc atau pyproject.toml
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import logging
import sqlite3
import subprocess
import time
import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from app.logger import setup_logging

setup_logging()
logger = logging.getLogger("app.main")

from config import (
    CHECKPOINT_DB_PATH,
    DB_PATH,
    MAX_ACTIVE_SESSIONS,
    MAX_QUERY_LENGTH,
)
from agent.graph import build_graph
from services.sqlite_service import get_all_sessions, delete_session
from app.components.sidebar import render_sidebar
from app.components.chat_display import render_messages


# ---------------------------------------------------------------------------
# Initialization flow
# ---------------------------------------------------------------------------

def _run_sql_init() -> None:
    """Jalankan script 03_ingest_sql.py untuk mengisi SQLite dari CSV.

    Dipanggil otomatis saat recruitmind.db tidak ditemukan. Ini terjadi
    di fresh environment seperti Streamlit Community Cloud setelah redeploy,
    karena SQLite bersifat ephemeral di sana.
    """
    script_path = os.path.join(
        os.path.dirname(__file__), "..", "pipeline", "03_ingest_sql.py"
    )
    result = subprocess.run(
        [sys.executable, os.path.abspath(script_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error("SQL init gagal: %s", result.stderr)
        st.error("Inisialisasi database gagal. Silakan refresh halaman.")
        st.stop()
    logger.info("SQL init selesai.")


def _ensure_db() -> None:
    """Pastikan recruitmind.db sudah ada sebelum UI dirender.

    Jika belum ada, tampilkan loading state dan jalankan 03_ingest_sql.py
    secara otomatis. Proses ini tidak melibatkan LLM sehingga selesai cepat.
    """
    if not os.path.exists(DB_PATH):
        with st.spinner("Mempersiapkan database... Ini hanya terjadi sekali."):
            logger.info("DB tidak ditemukan, menjalankan SQL init.")
            _run_sql_init()


# ---------------------------------------------------------------------------
# Graph caching
# ---------------------------------------------------------------------------

@st.cache_resource
def _get_graph():
    """Bangun dan cache graph LangGraph.

    Di-cache agar graph hanya diinisialisasi sekali per instance Streamlit,
    bukan di setiap re-run. SqliteSaver di dalam build_graph() juga
    hanya dibuat sekali sehingga koneksi ke checkpoints.db efisien.

    Returns:
        Graph LangGraph yang sudah di-compile.
    """
    logger.info("Menginisialisasi graph LangGraph.")
    return build_graph()


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def _init_session_state() -> None:
    """Inisialisasi key-key di st.session_state yang dibutuhkan UI.

    Hanya set nilai default untuk key yang belum ada,
    sehingga tidak menimpa state yang sudah ada saat re-run.
    """
    defaults = {
        "thread_id": None,
        "messages": [],
        "session_list_version": 0,
        "pending_delete_thread_id": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ---------------------------------------------------------------------------
# Session actions
# ---------------------------------------------------------------------------

def _handle_new_session() -> None:
    """Buat sesi baru dengan thread_id random.

    Cek limit sesi sebelum membuat, lalu reset messages
    agar area chat kosong untuk sesi baru.
    """
    sessions = get_all_sessions()
    if len(sessions) >= MAX_ACTIVE_SESSIONS:
        st.warning(
            f"Batas maksimal {MAX_ACTIVE_SESSIONS} sesi sudah tercapai. "
            "Hapus sesi yang tidak digunakan untuk membuat sesi baru."
        )
        return

    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.session_list_version += 1
    logger.info("Sesi baru dibuat: thread_id=%s", st.session_state.thread_id)


def _handle_load_session(thread_id: str) -> None:
    """Muat sesi yang sudah ada dari checkpointer.

    Jika sesi yang diklik sudah aktif, tidak perlu reload.

    Args:
        thread_id: ID unik sesi yang akan di-load.
    """
    if st.session_state.thread_id == thread_id:
        return

    st.session_state.thread_id = thread_id
    st.session_state.messages = _load_messages_from_checkpoint(thread_id)
    logger.info("Sesi di-load: thread_id=%s", thread_id)


def _handle_delete_session(thread_id: str) -> None:
    """Hapus sesi dari chat_sessions dan checkpoint terkait.

    Jika sesi yang dihapus adalah sesi aktif, reset UI ke state kosong.

    Args:
        thread_id: ID sesi yang akan dihapus.
    """
    delete_session(thread_id)
    _delete_checkpoint(thread_id)

    if st.session_state.thread_id == thread_id:
        st.session_state.thread_id = None
        st.session_state.messages = []

    st.session_state.session_list_version += 1
    logger.info("Sesi dihapus: thread_id=%s", thread_id)


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------

def _load_messages_from_checkpoint(thread_id: str) -> list:
    """Ambil riwayat pesan dari LangGraph checkpoint untuk ditampilkan di UI.

    Args:
        thread_id: ID sesi yang riwayatnya akan diambil.

    Returns:
        List of {"role": str, "content": str}. List kosong jika gagal.
    """
    graph = _get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    try:
        state = graph.get_state(config)
        if state and state.values and "messages" in state.values:
            ui_messages = []
            for msg in state.values["messages"]:
                if hasattr(msg, "type"):
                    role = "user" if msg.type == "human" else "assistant"
                    ui_messages.append({"role": role, "content": msg.content})
            return ui_messages
    except Exception as exc:
        logger.warning("Gagal load messages dari checkpoint: %s", exc)
    return []


def _delete_checkpoint(thread_id: str) -> None:
    """Hapus checkpoint dari checkpoints.db untuk thread_id tertentu.

    SqliteSaver tidak menyediakan API delete per thread, sehingga
    penghapusan dilakukan langsung via SQL ke tabel yang dikelolanya.

    Args:
        thread_id: ID thread yang checkpointnya akan dihapus.
    """
    if not os.path.exists(CHECKPOINT_DB_PATH):
        return
    try:
        with sqlite3.connect(CHECKPOINT_DB_PATH, check_same_thread=False) as conn:
            for table in ["checkpoints", "checkpoint_writes", "checkpoint_blobs"]:
                try:
                    conn.execute(
                        f"DELETE FROM {table} WHERE thread_id = ?", (thread_id,)
                    )
                except sqlite3.OperationalError:
                    pass
            conn.commit()
    except Exception as exc:
        logger.warning("Gagal hapus checkpoint thread_id=%s: %s", thread_id, exc)


# ---------------------------------------------------------------------------
# Query processing
# ---------------------------------------------------------------------------

def _get_langfuse_callbacks() -> list:
    """Buat LangFuse callback handler jika credentials tersedia di st.secrets.

    LangFuse 3.x membaca credentials dari environment variable.
    Semua credentials di-set ke env var sebelum handler dibuat.

    Returns:
        List berisi satu CallbackHandler, atau list kosong.
    """
    try:
        from langfuse.langchain import CallbackHandler

        public_key = st.secrets.get("LANGFUSE_PUBLIC_KEY", "")
        secret_key = st.secrets.get("LANGFUSE_SECRET_KEY", "")
        host = st.secrets.get("LANGFUSE_HOST", "https://cloud.langfuse.com")

        if not public_key or not secret_key:
            return []

        os.environ["LANGFUSE_PUBLIC_KEY"] = public_key
        os.environ["LANGFUSE_SECRET_KEY"] = secret_key
        os.environ["LANGFUSE_HOST"] = host

        handler = CallbackHandler()
        return [handler]
    except ImportError:
        return []
    except Exception as exc:
        logger.warning("LangFuse callback gagal diinisialisasi: %s", exc)
        return []


def _process_query(query: str, graph) -> str:
    """Kirim query ke graph dan kembalikan response sebagai string.

    Untuk single-step, kembalikan content dari AIMessage terakhir.
    Untuk multi-step, gabungkan hanya AIMessage baru dari invocation ini
    dengan separator agar setiap output node tampil terpisah.

    Args:
        query: Input teks dari user (sudah divalidasi panjangnya).
        graph: Graph LangGraph yang sudah di-compile.

    Returns:
        Konten response dari agent. Pesan error jika graph gagal.
    """
    thread_id = st.session_state.thread_id
    config = {"configurable": {"thread_id": thread_id}}

    callbacks = _get_langfuse_callbacks()
    if callbacks:
        config["callbacks"] = callbacks

    # Catat jumlah messages sebelum invoke untuk isolasi output invocation ini
    state_before = graph.get_state(config)
    count_before = len(state_before.values.get("messages", [])) if state_before and state_before.values else 0

    start_time = time.time()
    try:
        result = graph.invoke(
            {"messages": [HumanMessage(content=query)]},
            config=config,
        )
        elapsed = time.time() - start_time
        logger.info("Query selesai: thread_id=%s, durasi=%.2fs", thread_id, elapsed)
    except Exception as exc:
        elapsed = time.time() - start_time
        logger.error(
            "Error invoke graph: thread_id=%s, error=%s, durasi=%.2fs",
            thread_id, exc, elapsed,
        )
        return "Maaf, terjadi kesalahan saat memproses permintaan. Silakan coba lagi."

    all_messages = result.get("messages", [])
    if not all_messages:
        return "Maaf, tidak ada response yang dihasilkan."

    # Ambil hanya messages yang baru ditambahkan di invocation ini.
    # +1 untuk skip HumanMessage user yang baru di-append oleh input_node.
    new_messages = all_messages[count_before + 1:]
    new_ai_messages = [m for m in new_messages if isinstance(m, AIMessage)]

    if not new_ai_messages:
        return "Maaf, tidak ada response yang dihasilkan."

    if len(new_ai_messages) == 1:
        return new_ai_messages[0].content

    # Multi-step: gabungkan output semua node dengan separator
    return "\n\n---\n\n".join(m.content for m in new_ai_messages)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point utama aplikasi.

    Urutan eksekusi di setiap Streamlit re-run:
    1. Pastikan DB ada
    2. Load graph dari cache
    3. Inisialisasi session state
    4. Render sidebar
    5. Render area chat
    6. Handle input user
    """
    st.set_page_config(
        page_title="RecruitMind AI",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    _ensure_db()
    graph = _get_graph()
    _init_session_state()

    render_sidebar(
        on_new_session=_handle_new_session,
        on_load_session=_handle_load_session,
        on_delete_session=_handle_delete_session,
    )

    st.title("RecruitMind AI")
    st.caption("Asisten rekrutmen berbasis AI untuk pencarian dan analisis kandidat.")

    if st.session_state.thread_id is None:
        st.info("Buat sesi baru atau pilih sesi dari sidebar untuk mulai.")
        return

    render_messages(st.session_state.messages)

    query = st.chat_input(
        placeholder="Cari kandidat, bandingkan, atau minta generate pertanyaan interview...",
    )

    if not query:
        return

    if len(query) > MAX_QUERY_LENGTH:
        st.warning(
            f"Query terlalu panjang ({len(query)} karakter). "
            f"Di-truncate ke {MAX_QUERY_LENGTH} karakter."
        )
        query = query[:MAX_QUERY_LENGTH]

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Sedang memproses..."):
            response = _process_query(query, graph)

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.session_list_version += 1
    st.rerun()


if __name__ == "__main__":
    main()