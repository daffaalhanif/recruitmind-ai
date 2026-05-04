"""Fungsi utilitas yang dipakai bersama oleh node-node agent."""

from agent.state import AgentState
from config import SLIDING_WINDOW_SIZE


def get_windowed_messages(state: AgentState) -> list:
    """Ambil sliding window dari conversation history.

    Membatasi jumlah pesan yang dikirim ke LLM agar tidak melebihi
    SLIDING_WINDOW_SIZE. Full history tetap tersimpan di checkpointer
    dan tersedia untuk UI, tapi hanya window terakhir yang dikirim
    ke API untuk mengontrol penggunaan token.

    Args:
        state: State saat ini yang berisi full conversation history.

    Returns:
        List berisi maksimal SLIDING_WINDOW_SIZE pesan terakhir.
    """
    return state["messages"][-SLIDING_WINDOW_SIZE:]