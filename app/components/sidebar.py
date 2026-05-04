"""Komponen sidebar untuk manajemen sesi RecruitMind AI.

Menampilkan daftar sesi aktif, tombol buat sesi baru,
dan opsi hapus sesi dengan konfirmasi.
"""

from typing import Callable

import streamlit as st

from config import MAX_ACTIVE_SESSIONS
from services.sqlite_service import get_all_sessions


def render_sidebar(
    on_new_session: Callable,
    on_load_session: Callable[[str], None],
    on_delete_session: Callable[[str], None],
) -> None:
    """Render seluruh konten sidebar aplikasi.

    Args:
        on_new_session: Callback saat user membuat sesi baru.
        on_load_session: Callback saat user memilih sesi, menerima thread_id.
        on_delete_session: Callback saat user mengonfirmasi hapus, menerima thread_id.
    """
    with st.sidebar:
        st.title("RecruitMind AI")
        st.divider()
        _render_new_session_button(on_new_session)
        st.subheader("Sesi Aktif")
        _render_session_list(on_load_session, on_delete_session)


def _render_new_session_button(on_new_session: Callable) -> None:
    """Render tombol buat sesi baru.

    Tombol di-disable jika sudah mencapai MAX_ACTIVE_SESSIONS
    agar user tahu alasannya tanpa tombol menghilang.

    Args:
        on_new_session: Callback yang dipanggil saat tombol ditekan.
    """
    sessions = get_all_sessions()
    at_limit = len(sessions) >= MAX_ACTIVE_SESSIONS

    if at_limit:
        st.button(
            "+ Sesi Baru",
            disabled=True,
            use_container_width=True,
            help=f"Batas maksimal {MAX_ACTIVE_SESSIONS} sesi tercapai. Hapus sesi untuk membuat yang baru.",
        )
        st.caption(f"Sesi penuh ({len(sessions)}/{MAX_ACTIVE_SESSIONS}).")
    else:
        if st.button("+ Sesi Baru", use_container_width=True):
            on_new_session()
            st.rerun()


def _render_session_list(
    on_load_session: Callable[[str], None],
    on_delete_session: Callable[[str], None],
) -> None:
    """Render daftar sesi aktif dari database.

    Sesi aktif diberi penanda visual. Setiap sesi punya tombol hapus
    yang memunculkan konfirmasi sebelum eksekusi.

    Args:
        on_load_session: Callback untuk memuat sesi.
        on_delete_session: Callback untuk menghapus sesi.
    """
    sessions = get_all_sessions()

    if not sessions:
        st.caption("Belum ada sesi. Buat sesi baru untuk mulai.")
        return

    active_thread_id = st.session_state.get("thread_id")

    for session in sessions:
        thread_id = session["thread_id"]
        title = session.get("title") or "Sesi Tanpa Judul"
        created_at = session.get("created_at", "")
        is_active = thread_id == active_thread_id

        col_title, col_delete = st.columns([4, 1])

        with col_title:
            label = f"* {title}" if is_active else title
            if st.button(
                label,
                key=f"load_{thread_id}",
                use_container_width=True,
                help=f"Dibuat: {_format_timestamp(created_at)}",
                type="primary" if is_active else "secondary",
            ):
                on_load_session(thread_id)
                st.rerun()

        with col_delete:
            if st.button("X", key=f"del_{thread_id}", help="Hapus sesi ini"):
                st.session_state.pending_delete_thread_id = thread_id
                st.rerun()

        st.caption(_format_timestamp(created_at))

    _render_delete_confirmation(on_delete_session, sessions)


def _render_delete_confirmation(
    on_delete_session: Callable[[str], None],
    sessions: list,
) -> None:
    """Render konfirmasi hapus sesi.

    Dialog hanya muncul jika pending_delete_thread_id sudah di-set.
    Mencegah penghapusan tidak sengaja karena butuh klik kedua.

    Args:
        on_delete_session: Callback jika user mengonfirmasi hapus.
        sessions: Daftar sesi untuk mencari judul sesi yang akan dihapus.
    """
    pending_id = st.session_state.get("pending_delete_thread_id")
    if not pending_id:
        return

    pending_title = "sesi ini"
    for session in sessions:
        if session["thread_id"] == pending_id:
            pending_title = session.get("title") or "sesi tanpa judul"
            break

    st.divider()
    st.warning(f"Hapus '{pending_title}'? Riwayat percakapan akan hilang permanen.")

    col_confirm, col_cancel = st.columns(2)
    with col_confirm:
        if st.button("Ya, Hapus", key="confirm_delete", type="primary"):
            on_delete_session(pending_id)
            st.session_state.pending_delete_thread_id = None
            st.rerun()
    with col_cancel:
        if st.button("Batal", key="cancel_delete"):
            st.session_state.pending_delete_thread_id = None
            st.rerun()


def _format_timestamp(timestamp: str) -> str:
    """Format timestamp ISO ke format ringkas untuk ditampilkan.

    Timestamp disimpan dalam UTC di database. Dikonversi ke WIB (UTC+7)
    sebelum ditampilkan karena aplikasi digunakan di Indonesia.

    Args:
        timestamp: String timestamp ISO 8601.

    Returns:
        String terformat dalam WIB, atau string kosong jika gagal.
    """
    if not timestamp:
        return ""
    try:
        from datetime import datetime, timezone, timedelta

        dt = datetime.fromisoformat(timestamp)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        wib = timezone(timedelta(hours=7))
        dt_wib = dt.astimezone(wib)
        return dt_wib.strftime("%d %b %Y, %H:%M")
    except Exception:
        return timestamp[:16] if len(timestamp) >= 16 else timestamp