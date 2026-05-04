"""Router Node untuk RecruitMind AI.

Mengklasifikasikan intent dari query user dan conversation context,
lalu menulis hasilnya ke state untuk dikonsumsi conditional edge
dan node-node berikutnya.
"""

import logging
from typing import Optional

import streamlit as st
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from agent.nodes.utils import get_windowed_messages
from agent.prompts.router_prompt import build_router_prompt
from agent.state import AgentState
from config import LLM_MODEL, LLM_TEMPERATURE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema structured output
# ---------------------------------------------------------------------------

class TaskItem(BaseModel):
    """Satu task dalam antrian multi-step query.

    Attributes:
        intent: Kapabilitas yang akan dieksekusi. Nilai valid:
            RAG, SQL, EVALUATOR, GENERATOR.
        processed_query: Query untuk RAG atau SQL. None untuk
            EVALUATOR dan GENERATOR.
        candidate_refs: Referensi kandidat untuk EVALUATOR dan
            GENERATOR. List kosong untuk RAG dan SQL.
    """

    intent: str
    processed_query: Optional[str]
    candidate_refs: list[str]


class RouterOutput(BaseModel):
    """Schema output terstruktur dari Router LLM call.

    Attributes:
        intent: Kapabilitas yang dipilih. Nilai valid:
            RAG, SQL, EVALUATOR, GENERATOR, CLARIFY, MULTI_STEP.
        reasoning: Penjelasan singkat kenapa intent ini dipilih.
            Dipakai untuk logging, tidak ditampilkan ke user.
        candidate_refs: Referensi kandidat dari pesan user, hanya
            angka eksplisit. Contoh: ["1", "3"]. List kosong jika
            tidak ada referensi atau referensi tidak numerik.
        clarification_type: Jenis klarifikasi jika intent CLARIFY.
            Nilai valid: missing_candidates, no_prior_search,
            ambiguous_comparison, search_overwrite_confirmation,
            unclear_intent. None jika bukan CLARIFY atau butuh
            LLM fallback di Clarification Node.
        clarification_context: Reasoning tentang mengapa query ambigu.
            Diisi hanya jika clarification_type adalah None dan
            Clarification Node perlu LLM fallback.
        processed_query: Query yang sudah diproses untuk embedding.
            Berisi query asli untuk input singkat, atau key requirements
            hasil ekstraksi untuk input yang terdeteksi sebagai JD.
        task_queue: Antrian task untuk multi-step query. Hanya diisi
            saat MULTI_STEP terdeteksi. Task pertama langsung
            dieksekusi via intent, processed_query, dan candidate_refs.
            Task sisanya disimpan di sini untuk dieksekusi oleh
            task_dispatcher setelah setiap execution node selesai.
    """

    intent: str
    reasoning: str
    candidate_refs: list[str]
    clarification_type: Optional[str]
    clarification_context: Optional[str]
    processed_query: Optional[str]
    task_queue: list[TaskItem]


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------

def router_node(state: AgentState) -> dict:
    """Klasifikasi intent dari query dan conversation context.

    Mengambil sliding window dari messages, membangun prompt dengan
    ringkasan active_candidates, lalu memanggil LLM untuk menghasilkan
    RouterOutput. Hasilnya ditulis ke state untuk dikonsumsi
    conditional edge dan node berikutnya.

    Args:
        state: State saat ini yang berisi messages dan active_candidates.

    Returns:
        Dict berisi field state yang diupdate: intent, clarification_type,
        clarification_context, processed_query, candidate_refs, task_queue.
    """
    # LLM diinisialisasi di dalam fungsi karena st.secrets hanya tersedia
    # saat runtime Streamlit, bukan saat module diimport
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        api_key=st.secrets["OPENAI_API_KEY"],
    )
    structured_llm = llm.with_structured_output(RouterOutput)

    messages = get_windowed_messages(state)
    system_prompt = build_router_prompt(state.get("active_candidates", []))

    # Gabungkan system prompt dengan sliding window conversation history
    llm_input = [{"role": "system", "content": system_prompt}] + [
        {
            "role": m.type if hasattr(m, "type") else m["role"],
            "content": m.content if hasattr(m, "content") else m["content"],
        }
        for m in messages
    ]

    result: RouterOutput = structured_llm.invoke(llm_input)

    logger.info(
        "Router: intent=%s | candidate_refs=%s | task_queue=%s | reasoning=%s",
        result.intent,
        result.candidate_refs,
        result.task_queue,
        result.reasoning,
    )

    return {
        "intent": result.intent,
        "clarification_type": result.clarification_type,
        "clarification_context": result.clarification_context,
        "processed_query": result.processed_query,
        "candidate_refs": result.candidate_refs,
        # Konversi TaskItem ke dict agar bisa disimpan ke state
        "task_queue": [t.model_dump() for t in result.task_queue],
    }