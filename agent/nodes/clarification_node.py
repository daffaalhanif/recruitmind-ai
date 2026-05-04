"""Clarification Node untuk menangani query ambigu dan kondisi yang butuh konfirmasi.

Beroperasi dalam lima mode:
- Mode CHITCHAT: sapaan dan pertanyaan umum tentang sistem, gunakan template statis.
- Mode CONVERSATION: follow-up dari riwayat percakapan, jawab via LLM hanya dari history.
- Mode GENERAL: pertanyaan umum rekrutmen dan HR, jawab via LLM dari knowledge umum.
- Mode template: clarification_type dikenal, kembalikan string statis tanpa LLM call.
- Mode LLM fallback: semua kasus lain, jawab via LLM secara natural dengan awareness
  penuh tentang kapabilitas dan keterbatasan sistem. Tidak boleh halusinasi data.
"""

import logging

import streamlit as st
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from agent.nodes.utils import get_windowed_messages
from agent.prompts.clarification_fallback_prompt import (
    build_clarification_fallback_prompt,
    build_conversation_prompt,
    build_general_prompt,
)
from agent.state import AgentState
from config import LLM_MODEL, LLM_TEMPERATURE

logger = logging.getLogger(__name__)


# Template statis hanya untuk kasus terprediksi yang tidak butuh konteks LLM.
# unclear_intent tidak ada di sini agar ditangani LLM fallback secara natural.
_TEMPLATES: dict[str, str] = {
    "missing_candidates": (
        "Mohon sebutkan nomor kandidat yang ingin kamu proses, "
        "misalnya: 'bandingkan kandidat 1 dan 3' atau 'buat pertanyaan untuk kandidat 2'. "
        "Kamu bisa lihat nomor kandidat di hasil pencarian sebelumnya."
    ),
    "no_prior_search": (
        "Sepertinya kamu belum melakukan pencarian kandidat di sesi ini. "
        "Coba mulai dengan mencari kandidat terlebih dahulu, "
        "misalnya: 'cari kandidat dengan pengalaman Python dan machine learning'. "
        "Setelah itu kamu bisa membandingkan atau membuat pertanyaan wawancara."
    ),
    "ambiguous_comparison": (
        "Sepertinya kamu merujuk kandidat dari dua pencarian yang berbeda. "
        "Sistem hanya bisa memproses kandidat dari satu daftar aktif. "
        "Apakah kamu ingin melakukan pencarian baru, "
        "atau tetap menggunakan kandidat dari hasil pencarian terakhir?"
    ),
    "chitchat": (
        "Halo! Aku RecruitMind AI, asisten rekrutmen berbasis AI. "
        "Aku bisa membantu kamu untuk:\n"
        "- Mencari kandidat berdasarkan skill, posisi, atau pengalaman\n"
        "- Menganalisis data talent pool seperti distribusi skill atau jumlah kandidat\n"
        "- Membandingkan 2-3 kandidat secara berdampingan\n"
        "- Membuat pertanyaan wawancara yang spesifik untuk satu kandidat\n"
        "- Menjawab pertanyaan umum seputar rekrutmen dan HR\n\n"
        "Apa yang ingin kamu lakukan?"
    ),
}


def _build_llm() -> ChatOpenAI:
    """Inisialisasi LLM untuk clarification node.

    Diinisialisasi di dalam fungsi karena st.secrets hanya tersedia
    saat runtime Streamlit, bukan saat module diimport.

    Returns:
        Instance ChatOpenAI yang siap dipakai.
    """
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        api_key=st.secrets["OPENAI_API_KEY"],
    )


def _build_llm_input(system_prompt: str, state: AgentState) -> list:
    """Bangun input LLM dengan system prompt dan sliding window messages.

    Args:
        system_prompt: System prompt yang akan dipakai.
        state: State yang berisi messages.

    Returns:
        List of dict siap dikirim ke LLM.
    """
    messages = get_windowed_messages(state)
    return [{"role": "system", "content": system_prompt}] + [
        {
            "role": m.type if hasattr(m, "type") else m["role"],
            "content": m.content if hasattr(m, "content") else m["content"],
        }
        for m in messages
    ]


def _handle_conversation(state: AgentState) -> str:
    """Jawab pertanyaan follow-up hanya dari riwayat percakapan.

    Tidak boleh menambahkan informasi di luar yang ada di history.
    Jika informasi tidak ada di history, LLM harus mengakuinya.

    Args:
        state: State yang berisi messages.

    Returns:
        String response dari LLM.
    """
    system_prompt = build_conversation_prompt()
    result = _build_llm().invoke(_build_llm_input(system_prompt, state))
    return result.content


def _handle_general(state: AgentState) -> str:
    """Jawab pertanyaan umum rekrutmen dan HR dari knowledge LLM.

    Args:
        state: State yang berisi messages.

    Returns:
        String response dari LLM.
    """
    system_prompt = build_general_prompt()
    result = _build_llm().invoke(_build_llm_input(system_prompt, state))
    return result.content


def _handle_fallback(state: AgentState, clarification_context: str) -> str:
    """Tangani semua kasus yang tidak masuk template dengan LLM yang cerdas.

    LLM menjawab secara natural dengan awareness penuh tentang kapabilitas
    dan keterbatasan sistem. Tidak boleh mengarang data apapun.

    Args:
        state: State yang berisi messages.
        clarification_context: Konteks dari Router tentang mengapa query
            tidak dapat diproses langsung.

    Returns:
        String response dari LLM.
    """
    system_prompt = build_clarification_fallback_prompt(clarification_context)
    result = _build_llm().invoke(_build_llm_input(system_prompt, state))
    return result.content


def clarification_node(state: AgentState) -> dict:
    """Hasilkan response berdasarkan intent dan jenis ambiguitas dari Router.

    Mode eksekusi berdasarkan prioritas:
    1. CHITCHAT: template statis, tidak butuh LLM
    2. CONVERSATION: LLM dengan history sebagai satu-satunya sumber informasi
    3. GENERAL: LLM dengan knowledge rekrutmen umum, dilarang mengarang data sistem
    4. clarification_type dikenal: template statis untuk kasus terprediksi
    5. Semua kasus lain: LLM fallback yang natural dan jujur tentang keterbatasan

    Args:
        state: State LangGraph yang berisi intent, clarification_type,
            dan clarification_context dari Router Node.

    Returns:
        Dict dengan key messages berisi satu AIMessage.
    """
    intent = state.get("intent")
    clarification_type = state.get("clarification_type")
    clarification_context = state.get("clarification_context") or ""

    if intent == "CHITCHAT":
        logger.info("Clarification Node: CHITCHAT")
        response_text = _TEMPLATES["chitchat"]

    elif intent == "CONVERSATION":
        logger.info("Clarification Node: CONVERSATION")
        response_text = _handle_conversation(state)

    elif intent == "GENERAL":
        logger.info("Clarification Node: GENERAL")
        response_text = _handle_general(state)

    elif clarification_type in _TEMPLATES:
        logger.info("Clarification Node: template '%s'", clarification_type)
        response_text = _TEMPLATES[clarification_type]

    else:
        logger.info(
            "Clarification Node: LLM fallback, intent=%s, type=%s",
            intent,
            clarification_type,
        )
        response_text = _handle_fallback(state, clarification_context)

    return {"messages": [AIMessage(content=response_text)]}