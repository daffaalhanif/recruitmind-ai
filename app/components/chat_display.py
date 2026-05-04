"""Komponen rendering pesan chat untuk RecruitMind AI.

Menangani tampilan riwayat percakapan dengan deteksi format output
berdasarkan konten pesan dari assistant.
"""

import logging
import re

import streamlit as st

logger = logging.getLogger("app.chat_display")


def render_messages(messages: list) -> None:
    """Render seluruh riwayat percakapan di area chat utama.

    Args:
        messages: List of {"role": str, "content": str}.
    """
    for message in messages:
        role = message.get("role", "assistant")
        content = message.get("content", "")

        with st.chat_message(role):
            if role == "user":
                st.markdown(content)
            else:
                _render_assistant_message(content)


def _render_assistant_message(content: str) -> None:
    """Render pesan assistant dengan format yang sesuai kontennya.

    Comparison table dirender via st.markdown agar konten tidak terpotong.
    SQL table dirender via st.dataframe agar bisa di-sort secara interaktif.
    Konten lain dirender via st.markdown biasa.

    Deteksi format berbasis heuristik konten karena pesan yang di-load
    dari checkpoint tidak membawa metadata intent.

    Args:
        content: String konten pesan dari assistant.
    """
    if _looks_like_comparison_table(content):
        _render_comparison_result(content)
    elif _looks_like_sql_table(content):
        _render_sql_result(content)
    else:
        st.markdown(content)


def _looks_like_sql_table(content: str) -> bool:
    """Deteksi apakah konten mengandung tabel Markdown hasil SQL query.

    Args:
        content: String konten pesan.

    Returns:
        True jika terdeteksi sebagai tabel Markdown.
    """
    lines = content.strip().split("\n")
    table_lines = [l for l in lines if l.strip().startswith("|")]
    if len(table_lines) < 3:
        return False
    separator_lines = [l for l in table_lines if re.match(r"\|\s*[-:]+\s*\|", l)]
    return len(separator_lines) >= 1


def _looks_like_comparison_table(content: str) -> bool:
    """Deteksi apakah konten adalah tabel perbandingan kandidat dari Evaluator.

    Evaluator selalu menghasilkan header dengan kata "Kandidat" diikuti angka.

    Args:
        content: String konten pesan.

    Returns:
        True jika terdeteksi sebagai tabel perbandingan.
    """
    has_kandidat_header = bool(
        re.search(r"\|\s*Kandidat\s*\d+\s*\|", content, re.IGNORECASE)
    )
    return has_kandidat_header and _looks_like_sql_table(content)


def _render_sql_result(content: str) -> None:
    """Render hasil SQL analytics sebagai st.dataframe dan narasi sebagai st.markdown.

    st.dataframe dipakai agar user bisa sort kolom secara interaktif,
    yang berguna untuk data tabular analytics.

    Args:
        content: String konten gabungan tabel dan narasi.
    """
    table_part, narrative_part = _split_table_and_narrative(content)

    if table_part:
        try:
            df = _markdown_table_to_dataframe(table_part)
            st.dataframe(df, use_container_width=True)
        except Exception as exc:
            logger.warning("Gagal parse tabel SQL ke dataframe: %s", exc)
            st.markdown(table_part)
    else:
        st.markdown(content)

    if narrative_part:
        st.markdown(narrative_part)


def _render_comparison_result(content: str) -> None:
    """Render tabel perbandingan kandidat via st.markdown agar konten tidak terpotong.

    st.markdown dipakai karena comparison table berisi teks panjang per sel
    yang akan terpotong jika menggunakan st.dataframe. Sorting tidak
    dibutuhkan untuk comparison table karena baris-barisnya adalah
    dimensi perbandingan, bukan data yang perlu diurutkan.

    Args:
        content: String konten tabel perbandingan dan rekomendasi.
    """
    st.markdown(content)


def _split_table_and_narrative(content: str) -> tuple[str, str]:
    """Pisahkan bagian tabel Markdown dari narasi di bawahnya.

    Args:
        content: String konten gabungan.

    Returns:
        Tuple (table_part, narrative_part). Salah satu bisa string kosong.
    """
    lines = content.strip().split("\n")
    table_lines = []
    narrative_lines = []
    in_table = True

    for line in lines:
        if in_table and line.strip().startswith("|"):
            table_lines.append(line)
        elif in_table and not line.strip():
            continue
        else:
            in_table = False
            narrative_lines.append(line)

    return "\n".join(table_lines).strip(), "\n".join(narrative_lines).strip()


def _markdown_table_to_dataframe(table_markdown: str):
    """Konversi Markdown table string menjadi pandas DataFrame.

    Args:
        table_markdown: String tabel dalam format Markdown.

    Returns:
        pandas DataFrame dengan header dari baris pertama tabel.

    Raises:
        ValueError: Jika format tabel tidak dapat di-parse.
    """
    import pandas as pd

    lines = [l.strip() for l in table_markdown.strip().split("\n") if l.strip()]
    data_lines = [l for l in lines if not re.match(r"\|\s*[-:]+\s*\|", l)]

    if len(data_lines) < 2:
        raise ValueError("Tabel tidak memiliki cukup baris data.")

    def parse_row(line: str) -> list:
        return [cell.strip() for cell in line.strip("|").split("|")]

    headers = parse_row(data_lines[0])
    rows = [parse_row(line) for line in data_lines[1:]]

    num_cols = len(headers)
    normalized_rows = []
    for row in rows:
        if len(row) < num_cols:
            row = row + [""] * (num_cols - len(row))
        normalized_rows.append(row[:num_cols])

    return pd.DataFrame(normalized_rows, columns=headers)