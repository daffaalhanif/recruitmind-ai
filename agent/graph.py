"""Definisi graph LangGraph untuk RecruitMind AI.

File ini mendefinisikan struktur graph: node-node yang terlibat,
edges yang menghubungkan antar node, dan conditional edges yang
menentukan alur berdasarkan nilai di state.

Graph di-compile sekali dan di-reuse untuk semua invocation via
st.cache_resource di Streamlit. Checkpointer SqliteSaver memastikan
state percakapan persist antar invocation selama thread_id sama.
"""

import logging
import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from agent.nodes.clarification_node import clarification_node
from agent.nodes.evaluator_node import evaluator_node
from agent.nodes.generator_node import generator_node
from agent.nodes.input_node import input_node
from agent.nodes.rag_node import rag_node
from agent.nodes.response_node import response_node
from agent.nodes.router import router_node
from agent.nodes.sql_node import sql_node
from agent.state import AgentState
from config import CHECKPOINT_DB_PATH

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Task dispatcher node
# ---------------------------------------------------------------------------

def task_dispatcher_node(state: AgentState) -> dict:
    """Pop task berikutnya dari task_queue dan tulis parameternya ke state.

    Dipanggil oleh should_continue setelah execution node selesai jika
    task_queue tidak kosong. Mengambil task pertama, menghapusnya dari
    queue, lalu menulis intent dan parameter task tersebut ke state
    agar execution node berikutnya bisa membacanya seperti biasa.

    Dengan cara ini execution node tidak perlu tahu apakah mereka
    dipanggil dari single-step atau multi-step query.

    Args:
        state: State saat ini yang berisi task_queue.

    Returns:
        Dict berisi intent, processed_query, candidate_refs dari task
        yang di-pop, dan task_queue yang sudah diperbarui.
    """
    task_queue = list(state.get("task_queue", []))

    if not task_queue:
        return {}

    next_task = task_queue.pop(0)

    logger.info(
        "Task Dispatcher: eksekusi task intent=%s | sisa queue=%d",
        next_task.get("intent"),
        len(task_queue),
    )

    return {
        "intent": next_task.get("intent"),
        "processed_query": next_task.get("processed_query"),
        "candidate_refs": next_task.get("candidate_refs", []),
        "task_queue": task_queue,
    }


# ---------------------------------------------------------------------------
# Conditional edges
# ---------------------------------------------------------------------------

def route_after_router(state: AgentState) -> str:
    """Tentukan node berikutnya berdasarkan intent hasil Router.

    Args:
        state: State saat ini setelah router_node menulis intent.

    Returns:
        Nama node tujuan sebagai string.
    """
    intent = state.get("intent", "CLARIFY")

    routes = {
        "RAG": "rag",
        "SQL": "sql",
        "EVALUATOR": "evaluator",
        "GENERATOR": "generator",
        "CLARIFY": "clarification",
        "CHITCHAT": "clarification",
        "CONVERSATION": "clarification",
        "MULTI_STEP": "rag",
    }

    return routes.get(intent, "clarification")


def should_continue(state: AgentState) -> str:
    """Cek task_queue setelah execution node selesai.

    Jika masih ada task di queue, route ke task_dispatcher untuk pop
    task berikutnya dan tulis parameternya ke state. Jika kosong,
    lanjut ke response node.

    Args:
        state: State saat ini setelah execution node menulis output.

    Returns:
        Nama node tujuan: task_dispatcher atau response.
    """
    task_queue = state.get("task_queue", [])

    if task_queue:
        return "task_dispatcher"

    return "response"


def route_after_dispatcher(state: AgentState) -> str:
    """Tentukan execution node berikutnya setelah task_dispatcher.

    Membaca intent yang sudah ditulis task_dispatcher ke state
    dan route ke execution node yang sesuai.

    Args:
        state: State saat ini setelah task_dispatcher menulis intent.

    Returns:
        Nama node tujuan sebagai string.
    """
    intent = state.get("intent", "")

    routes = {
        "RAG": "rag",
        "SQL": "sql",
        "EVALUATOR": "evaluator",
        "GENERATOR": "generator",
    }

    return routes.get(intent, "response")


# ---------------------------------------------------------------------------
# Graph definition
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """Bangun dan compile graph LangGraph.

    Mendaftarkan semua node, menghubungkan edges, dan compile dengan
    SqliteSaver sebagai checkpointer untuk persist state percakapan.

    Returns:
        Graph yang sudah di-compile dan siap di-invoke.
    """
    graph = StateGraph(AgentState)

    # Daftarkan semua node
    graph.add_node("input", input_node)
    graph.add_node("router", router_node)
    graph.add_node("rag", rag_node)
    graph.add_node("sql", sql_node)
    graph.add_node("evaluator", evaluator_node)
    graph.add_node("generator", generator_node)
    graph.add_node("clarification", clarification_node)
    graph.add_node("task_dispatcher", task_dispatcher_node)
    graph.add_node("response", response_node)

    # Entry point
    graph.set_entry_point("input")

    # Edge dari input ke router
    graph.add_edge("input", "router")

    # Conditional edge dari router ke node yang sesuai
    graph.add_conditional_edges(
        "router",
        route_after_router,
        {
            "rag": "rag",
            "sql": "sql",
            "evaluator": "evaluator",
            "generator": "generator",
            "clarification": "clarification",
        },
    )

    # Setiap execution node cek task_queue setelah selesai
    for node in ["rag", "sql", "evaluator", "generator"]:
        graph.add_conditional_edges(
            node,
            should_continue,
            {
                "task_dispatcher": "task_dispatcher",
                "response": "response",
            },
        )

    # task_dispatcher route ke execution node berikutnya
    graph.add_conditional_edges(
        "task_dispatcher",
        route_after_dispatcher,
        {
            "rag": "rag",
            "sql": "sql",
            "evaluator": "evaluator",
            "generator": "generator",
            "response": "response",
        },
    )

    # Clarification selalu terminal
    graph.add_edge("clarification", "response")

    # Response node adalah titik akhir
    graph.add_edge("response", END)

    # check_same_thread=False diperlukan karena Streamlit menjalankan
    # callback di thread yang berbeda dari thread utama
    conn = sqlite3.connect(CHECKPOINT_DB_PATH, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    return graph.compile(checkpointer=checkpointer)