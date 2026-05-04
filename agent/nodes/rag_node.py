"""RAG Node untuk RecruitMind AI.

Menjalankan pipeline retrieval lengkap: embed query, search Qdrant,
fetch profil dari SQLite, rerank dengan FlashRank menggunakan
resume_summary sebagai input cross-encoder, lalu generate narasi
kandidat menggunakan LLM.
"""

import logging

import streamlit as st
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from agent.prompts.rag_prompt import SYSTEM_PROMPT, build_rag_prompt
from agent.state import AgentState
from config import LLM_MODEL, LLM_TEMPERATURE
from services.embedding_service import embed_text
from services.qdrant_service import search
from services.reranker_service import rerank
from services.sqlite_service import get_resumes_by_ids

logger = logging.getLogger(__name__)


def rag_node(state: AgentState) -> dict:
    """Jalankan pipeline retrieval dan generate narasi kandidat.

    Pipeline:
    1. Embed processed_query dari state
    2. Search Qdrant top-15
    3. Fetch profil dari SQLite untuk top-15 kandidat
    4. Rerank ke top-5 menggunakan resume_summary sebagai input cross-encoder
    5. Populate active_candidates di state
    6. Generate response Markdown dengan format [1][2][3]

    Resume_summary dipakai sebagai input reranker karena lebih informatif
    dan konsisten dibanding preview 200 karakter dari Qdrant payload.
    Fetch SQLite dilakukan sebelum reranking agar summary tersedia
    saat cross-encoder menilai relevansi.

    Args:
        state: State saat ini yang berisi processed_query.

    Returns:
        Dict berisi active_candidates yang diupdate dan messages
        dengan response LLM di-append.
    """
    query = state.get("processed_query") or ""

    # Stage 1: embed query
    query_vector = embed_text(query)
    logger.info("RAG: embedding selesai untuk query: %s", query[:50])

    # Stage 2: search Qdrant top-15
    qdrant_results = search(query_vector)
    logger.info("RAG: Qdrant mengembalikan %d kandidat", len(qdrant_results))

    # Stage 3: fetch profil dari SQLite untuk semua kandidat Qdrant
    # Dilakukan sebelum reranking agar resume_summary tersedia untuk cross-encoder
    resume_ids = [c["id"] for c in qdrant_results]
    sqlite_profiles = get_resumes_by_ids(resume_ids)
    sqlite_lookup = {p["id"]: p for p in sqlite_profiles}

    # Gabungkan resume_summary ke qdrant_results sebelum rerank
    for candidate in qdrant_results:
        profile = sqlite_lookup.get(candidate["id"], {})
        candidate["resume_summary"] = profile.get("resume_summary", "")
        candidate["top_skills"] = profile.get("top_skills", [])
        candidate["years_experience"] = profile.get("years_experience")
        candidate["current_position"] = profile.get("current_position")

    # Stage 4: rerank ke top-5 menggunakan resume_summary
    reranked = rerank(query, qdrant_results)
    logger.info(
        "RAG: top rerank score=%.4f | top qdrant score=%.4f",
        reranked[0]["rerank_score"] if reranked else 0,
        reranked[0]["qdrant_score"] if reranked else 0,
    )

    # Stage 5: susun active_candidates dari hasil reranking
    # Data sudah lengkap karena SQLite sudah di-fetch di Stage 3
    active_candidates = []
    for candidate in reranked:
        active_candidates.append(
            {
                "id": candidate["id"],
                "category": candidate["category"],
                "current_position": candidate.get("current_position"),
                "qdrant_score": candidate["qdrant_score"],
                "rerank_score": candidate["rerank_score"],
                "preview": candidate["preview"],
                "resume_summary": candidate.get("resume_summary", ""),
                "top_skills": candidate.get("top_skills", []),
                "years_experience": candidate.get("years_experience"),
            }
        )

    # Stage 6: generate narasi kandidat
    # LLM diinisialisasi di dalam fungsi karena st.secrets hanya tersedia
    # saat runtime Streamlit, bukan saat module diimport
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        api_key=st.secrets["OPENAI_API_KEY"],
    )

    user_prompt = build_rag_prompt(query, active_candidates)
    response = llm.invoke(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
    )

    logger.info(
        "RAG: response berhasil di-generate untuk %d kandidat",
        len(active_candidates),
    )

    return {
        "active_candidates": active_candidates,
        "messages": [AIMessage(content=response.content)],
    }