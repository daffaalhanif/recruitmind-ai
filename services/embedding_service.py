"""Service layer untuk embedding di RecruitMind AI.

Membungkus OpenAI embedding API sehingga node-node agent tidak perlu
tahu nama model, dimensi vector, atau format response API.

Client diinisialisasi sekali saat modul diimport dan di-reuse untuk
semua call. Ini menghindari overhead inisialisasi ulang di setiap
pemanggilan embedding.
"""

import streamlit as st
from openai import OpenAI

from config import EMBEDDING_MODEL

# Client diinisialisasi sekali, dipakai untuk semua call di modul ini
_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


def embed_text(text: str) -> list[float]:
    """Ubah teks menjadi vector embedding.

    Dimensi vector ditentukan oleh model yang dikonfigurasi di config.py,
    saat ini text-embedding-3-small menghasilkan 1536 dimensi.

    Args:
        text: Teks yang akan di-embed. Untuk retrieval, ini adalah
            processed_query dari state. Untuk indexing, ini adalah
            resume_summary.

    Returns:
        List float yang merepresentasikan vector embedding.

    Raises:
        openai.OpenAIError: Jika terjadi error API (network, auth, rate limit).
    """
    response = _client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding