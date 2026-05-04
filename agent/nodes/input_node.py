"""Input Node untuk RecruitMind AI.

Node pertama yang dieksekusi di setiap invokasi graph. Tugasnya hanya
sebagai entry point eksplisit - tidak melakukan transformasi apapun
karena LangGraph sudah append HumanMessage ke state sebelum graph jalan.
"""

from agent.state import AgentState


def input_node(state: AgentState) -> dict:
    """Terima pesan user sebagai entry point graph.

    Tidak melakukan apapun karena pesan user sudah di-append ke messages
    oleh LangGraph sebelum graph di-invoke. Fungsinya hanya sebagai
    entry point eksplisit agar struktur graph konsisten dengan node lain.

    Args:
        state: State saat ini.

    Returns:
        Dict kosong karena tidak ada perubahan state di node ini.
    """
    return {}