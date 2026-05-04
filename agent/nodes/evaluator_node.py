"""Evaluator Node untuk membandingkan 2-3 kandidat dan memberikan rekomendasi.

Menerima candidate_refs dari state, me-resolve ke profil aktual di active_candidates,
lalu menghasilkan tabel perbandingan dan rekomendasi via LLM.
"""

import logging

import streamlit as st
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from agent.nodes.utils import get_windowed_messages
from agent.prompts.evaluator_prompt import build_evaluator_prompt
from agent.state import AgentState
from config import LLM_MODEL, LLM_TEMPERATURE

logger = logging.getLogger(__name__)


def _resolve_candidates(candidate_refs: list[str], active_candidates: list[dict]) -> list[dict]:
    """Petakan referensi angka dari user ke profil kandidat di active_candidates.

    Menggunakan 1-based index karena RAG Node menampilkan kandidat sebagai [1][2][3].
    Index yang di luar rentang dilewati dengan log warning.

    Args:
        candidate_refs: List string angka eksplisit, misalnya ["1", "3"].
        active_candidates: List profil kandidat aktif dari state.

    Returns:
        List dict profil kandidat yang berhasil di-resolve.
    """
    resolved = []
    for ref in candidate_refs:
        try:
            idx = int(ref) - 1
            if 0 <= idx < len(active_candidates):
                resolved.append(active_candidates[idx])
            else:
                logger.warning(
                    "Evaluator Node: ref '%s' di luar rentang (total active_candidates: %d)",
                    ref,
                    len(active_candidates),
                )
        except ValueError:
            # Seharusnya tidak terjadi karena Router sudah memfilter ke angka saja
            logger.warning("Evaluator Node: ref tidak valid: '%s'", ref)

    return resolved


def evaluator_node(state: AgentState) -> dict:
    """Bandingkan 2-3 kandidat dan hasilkan tabel perbandingan serta rekomendasi.

    Membaca candidate_refs dan active_candidates dari state. Data profil diambil
    langsung dari active_candidates tanpa query SQLite tambahan karena RAG Node
    sudah menyimpan semua field yang dibutuhkan ke state.

    Args:
        state: State LangGraph yang berisi candidate_refs dan active_candidates.

    Returns:
        Dict dengan key messages berisi satu AIMessage berisi tabel perbandingan
        dan paragraf rekomendasi dalam format Markdown.
    """
    candidate_refs = state.get("candidate_refs", [])
    active_candidates = state.get("active_candidates", [])

    resolved = _resolve_candidates(candidate_refs, active_candidates)

    if len(resolved) < 2:
        logger.error(
            "Evaluator Node: kandidat ter-resolve kurang dari 2 (refs=%s, total active: %d)",
            candidate_refs,
            len(active_candidates),
        )
        return {
            "messages": [
                AIMessage(
                    content=(
                        "Maaf, tidak bisa membandingkan karena jumlah kandidat yang valid "
                        "kurang dari 2. Pastikan nomor kandidat yang kamu sebutkan ada di "
                        "hasil pencarian aktif."
                    )
                )
            ]
        }

    logger.info(
        "Evaluator Node: membandingkan %d kandidat (refs=%s)",
        len(resolved),
        candidate_refs,
    )

    system_prompt = build_evaluator_prompt(resolved)
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