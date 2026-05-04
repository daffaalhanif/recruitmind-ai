"""Generator Node untuk membuat pertanyaan wawancara spesifik per kandidat.

Menerima candidate_refs dari state, me-resolve satu kandidat dari active_candidates,
lalu menghasilkan daftar pertanyaan wawancara yang spesifik terhadap profil via LLM.
"""

import logging

import streamlit as st
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from agent.nodes.utils import get_windowed_messages
from agent.prompts.generator_prompt import build_generator_prompt
from agent.state import AgentState
from config import LLM_MODEL, LLM_TEMPERATURE

logger = logging.getLogger(__name__)


def _resolve_single_candidate(candidate_refs: list[str], active_candidates: list[dict]) -> dict | None:
    """Resolve satu referensi kandidat ke profil di active_candidates.

    Mengambil ref pertama dari candidate_refs. Router sudah memvalidasi
    bahwa Generator hanya menerima satu referensi kandidat.

    Args:
        candidate_refs: List string angka. Hanya index 0 yang digunakan.
        active_candidates: List profil kandidat aktif dari state.

    Returns:
        Dict profil kandidat jika resolve berhasil, None jika tidak.
    """
    if not candidate_refs:
        logger.warning("Generator Node: candidate_refs kosong")
        return None

    ref = candidate_refs[0]
    try:
        idx = int(ref) - 1
        if 0 <= idx < len(active_candidates):
            return active_candidates[idx]
        logger.warning(
            "Generator Node: ref '%s' di luar rentang (total active_candidates: %d)",
            ref,
            len(active_candidates),
        )
        return None
    except ValueError:
        logger.warning("Generator Node: ref tidak valid: '%s'", ref)
        return None


def generator_node(state: AgentState) -> dict:
    """Buat pertanyaan wawancara yang spesifik untuk satu kandidat.

    Membaca candidate_refs dan active_candidates dari state. Data profil
    diambil langsung dari active_candidates tanpa query SQLite tambahan
    karena RAG Node sudah menyimpan semua field yang dibutuhkan ke state.

    Args:
        state: State LangGraph yang berisi candidate_refs dan active_candidates.

    Returns:
        Dict dengan key messages berisi satu AIMessage berisi daftar
        pertanyaan wawancara yang dinomori.
    """
    candidate_refs = state.get("candidate_refs", [])
    active_candidates = state.get("active_candidates", [])

    candidate = _resolve_single_candidate(candidate_refs, active_candidates)

    if candidate is None:
        logger.error(
            "Generator Node: kandidat tidak bisa di-resolve (refs=%s, total active: %d)",
            candidate_refs,
            len(active_candidates),
        )
        return {
            "messages": [
                AIMessage(
                    content=(
                        "Maaf, tidak bisa menemukan kandidat yang kamu maksud. "
                        "Pastikan nomor kandidat yang kamu sebutkan ada di "
                        "hasil pencarian aktif."
                    )
                )
            ]
        }

    logger.info(
        "Generator Node: generate pertanyaan untuk ref '%s' (id=%s, posisi=%s)",
        candidate_refs[0],
        candidate.get("id"),
        candidate.get("current_position"),
    )

    system_prompt = build_generator_prompt(candidate)
    windowed_messages = get_windowed_messages(state)

    # LLM diinisialisasi di dalam fungsi karena st.secrets hanya tersedia
    # saat runtime Streamlit, bukan saat module diimport
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        api_key=st.secrets["OPENAI_API_KEY"],
    )

    llm_input = [{"role": "system", "content": system_prompt}] + [
        {
            "role": m.type if hasattr(m, "type") else m["role"],
            "content": m.content if hasattr(m, "content") else m["content"],
        }
        for m in windowed_messages
    ]

    result = llm.invoke(llm_input)

    return {"messages": [AIMessage(content=result.content)]}