"""Response Node untuk menangani session title di setiap akhir invokasi graph.

Node terminal sebelum END di semua alur. Tugasnya:
- Invokasi pertama sesi baru: generate session title via LLM, simpan ke SQLite
- Invokasi berikutnya: tidak melakukan apapun

Node ini tidak menulis ke messages karena output narasi sudah ditulis
oleh execution node sebelumnya.
"""

import logging

import streamlit as st
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.types import RunnableConfig

from agent.state import AgentState
from config import LLM_MODEL, LLM_TEMPERATURE
from services.sqlite_service import save_session

logger = logging.getLogger(__name__)

SESSION_TITLE_SYSTEM_PROMPT = """Kamu adalah asisten yang membuat judul ringkas untuk sesi percakapan.

Buat judul MAKSIMAL 5 kata dalam Bahasa Indonesia yang merangkum topik utama dari pesan pengguna.

Aturan:
- Keluarkan judul saja, tanpa penjelasan atau tanda kutip
- Gunakan topik yang spesifik
- Hindari kata generik seperti "Percakapan tentang" atau "Diskusi mengenai"

Contoh yang baik: "Pencarian Data Scientist Python", "Analitik Kandidat Per Kategori"
"""


def _get_first_user_message(state: AgentState) -> str:
    """Ambil teks pesan pertama dari user dalam sesi.

    Pesan pertama paling merepresentasikan topik sesi karena belum
    ada konteks percakapan sebelumnya.

    Args:
        state: State LangGraph yang berisi messages.

    Returns:
        Teks pesan user pertama, atau string kosong jika tidak ada.
    """
    for message in state.get("messages", []):
        if isinstance(message, HumanMessage):
            return message.content
    return ""


def _generate_session_title(first_message: str) -> str:
    """Generate judul sesi via LLM berdasarkan pesan pertama user.

    Dipanggil hanya sekali per sesi saat session_title masih None.

    Args:
        first_message: Pesan pertama dari user sebagai konteks judul.

    Returns:
        Judul sesi dalam Bahasa Indonesia, maksimal 5 kata.
    """
    # LLM diinisialisasi di dalam fungsi karena st.secrets hanya tersedia
    # saat runtime Streamlit, bukan saat module diimport
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        api_key=st.secrets["OPENAI_API_KEY"],
    )

    result = llm.invoke([
        {"role": "system", "content": SESSION_TITLE_SYSTEM_PROMPT},
        {"role": "user", "content": first_message},
    ])

    return result.content.strip()


def response_node(state: AgentState, config: RunnableConfig) -> dict:
    """Tangani session title di setiap akhir invokasi graph.

    Membaca thread_id dari config yang di-pass oleh graph.invoke()
    di Streamlit. Session title hanya di-generate dan disimpan sekali
    per sesi, yaitu saat session_title di state masih None.

    Args:
        state: State LangGraph yang berisi messages dan session_title.
        config: LangGraph config yang berisi thread_id di configurable.

    Returns:
        Dict kosong jika session_title sudah ada, atau dict berisi
        session_title baru jika ini invokasi pertama sesi.
    """
    if state.get("session_title") is not None:
        logger.debug("Response Node: session_title sudah ada, skip generate")
        return {}

    thread_id = config["configurable"]["thread_id"]
    first_message = _get_first_user_message(state)

    if not first_message:
        logger.warning("Response Node: tidak ada pesan user di state, pakai judul default")
        title = "Sesi Rekrutmen Baru"
    else:
        logger.info("Response Node: generate session title untuk thread_id=%s", thread_id)
        title = _generate_session_title(first_message)

    save_session(thread_id=thread_id, title=title)
    logger.info("Response Node: session title disimpan: '%s'", title)

    return {"session_title": title}