"""SQL Node untuk RecruitMind AI.

Mengubah pertanyaan analytics bahasa natural menjadi SQL query,
memvalidasi keamanannya, mengeksekusi ke SQLite, lalu memformat
hasil sebagai response untuk user.
"""

import logging

import streamlit as st
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from agent.prompts.sql_prompt import SYSTEM_PROMPT, build_sql_prompt
from agent.state import AgentState
from config import LLM_MODEL, LLM_TEMPERATURE
from services.sqlite_service import execute_analytics_query

logger = logging.getLogger(__name__)

# Operasi SQL yang diblokir untuk mencegah modifikasi atau penghapusan data
_BLOCKED_OPERATIONS = {
    "DROP", "DELETE", "TRUNCATE", "ALTER", "CREATE",
    "INSERT", "UPDATE", "EXECUTE", "ATTACH", "DETACH",
}


def _validate_sql(sql: str) -> tuple[bool, str]:
    """Validasi query SQL sebelum dieksekusi.

    Memeriksa dua hal: apakah query mengandung operasi yang diblokir,
    dan apakah query dimulai dengan SELECT.

    Args:
        sql: Query SQL yang akan divalidasi.

    Returns:
        Tuple (valid, pesan_error). Jika valid, pesan_error adalah
        string kosong. Jika tidak valid, pesan_error berisi alasan.
    """
    sql_upper = sql.upper().strip()

    for operation in _BLOCKED_OPERATIONS:
        if operation in sql_upper:
            return False, f"Operasi {operation} tidak diizinkan."

    if not sql_upper.startswith("SELECT"):
        return False, "Hanya SELECT query yang diizinkan."

    return True, ""


def _format_results(rows: list[dict]) -> str:
    """Format hasil query sebagai teks tabel sederhana.

    Args:
        rows: List dict hasil query dari SQLite.

    Returns:
        String tabel yang siap ditampilkan ke user, atau pesan
        jika tidak ada hasil.
    """
    if not rows:
        return "Query berhasil dieksekusi tetapi tidak ada data yang ditemukan."

    headers = list(rows[0].keys())
    header_line = " | ".join(headers)
    separator = " | ".join("-" * len(h) for h in headers)

    data_lines = []
    for row in rows:
        line = " | ".join(str(row.get(h, "")) for h in headers)
        data_lines.append(line)

    return "\n".join([header_line, separator] + data_lines)


def sql_node(state: AgentState) -> dict:
    """Generate SQL dari query natural language, validasi, eksekusi, format hasil.

    Args:
        state: State saat ini yang berisi messages dengan query user.

    Returns:
        Dict berisi sql_result yang diupdate dan messages dengan
        response di-append.
    """
    # Ambil query dari pesan terakhir user
    messages = state["messages"]
    last_message = messages[-1]
    query = (
        last_message.content
        if hasattr(last_message, "content")
        else last_message["content"]
    )

    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        api_key=st.secrets["OPENAI_API_KEY"],
    )

    # Generate SQL dari LLM
    response = llm.invoke(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_sql_prompt(query)},
        ]
    )
    generated_sql = response.content.strip()
    logger.info("SQL Node: query di-generate: %s", generated_sql)

    # Validasi keamanan sebelum eksekusi
    is_valid, error_message = _validate_sql(generated_sql)
    if not is_valid:
        logger.warning("SQL Node: query diblokir: %s | alasan: %s", generated_sql, error_message)
        return {
            "sql_result": {"status": "blocked", "sql": generated_sql, "reason": error_message},
            "messages": [AIMessage(content=f"Query tidak dapat dieksekusi: {error_message}")],
        }

    # Eksekusi query
    try:
        rows = execute_analytics_query(generated_sql)
        logger.info("SQL Node: query berhasil, %d baris dikembalikan", len(rows))
    except Exception as e:
        logger.error("SQL Node: eksekusi gagal: %s", str(e))
        return {
            "sql_result": {"status": "error", "sql": generated_sql, "error": str(e)},
            "messages": [AIMessage(content="Terjadi kesalahan saat mengeksekusi query. Coba ulangi pertanyaan dengan cara yang berbeda.")],
        }

    # Format hasil dan generate narasi singkat
    table_text = _format_results(rows)
    narasi_response = llm.invoke(
        [
            {"role": "user", "content": f"Pertanyaan: {query}\n\nHasil query:\n{table_text}\n\nBuat narasi singkat 1-2 kalimat yang menjelaskan temuan utama dari data ini."},
        ]
    )

    full_response = f"{table_text}\n\n{narasi_response.content}"

    return {
        "sql_result": {"status": "success", "sql": generated_sql, "row_count": len(rows)},
        "messages": [AIMessage(content=full_response)],
    }