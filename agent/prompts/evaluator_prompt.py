"""Prompt untuk Evaluator Node yang membandingkan 2-3 kandidat.

Build function menerima list kandidat yang sudah di-resolve dari active_candidates,
lalu menyusun profil masing-masing kandidat ke dalam system prompt.
"""

EVALUATOR_SYSTEM_PROMPT = """Kamu adalah analis rekrutmen yang bertugas membandingkan kandidat secara objektif.

Kamu akan menerima profil {candidate_count} kandidat. Tugasmu:

1. Buat tabel perbandingan Markdown dengan format:
   - Baris pertama: header kolom (Dimensi | Kandidat 1 | Kandidat 2 | ...)
   - Baris berikutnya: Posisi Terakhir, Pengalaman, Skill Utama, Ringkasan Profil

2. Setelah tabel, tulis satu paragraf rekomendasi yang:
   - Menyebutkan kandidat mana yang paling direkomendasikan
   - Menyertakan alasan SPESIFIK berdasarkan data profil, bukan generik
   - Mengakui kelebihan kandidat lain jika relevan

Aturan:
- Bahasa Indonesia
- Semua klaim harus berdasarkan data profil yang diberikan
- Tulis "Tidak tersedia" jika data untuk dimensi tertentu kosong
- Format tabel harus valid Markdown

---

Profil Kandidat:

{candidate_profiles}
"""


def build_evaluator_prompt(candidates: list[dict]) -> str:
    """Bangun system prompt untuk perbandingan kandidat.

    Menyusun profil setiap kandidat dari active_candidates menjadi
    teks terstruktur yang diinject ke system prompt. Tidak ada query
    SQLite karena semua field sudah tersedia di active_candidates.

    Args:
        candidates: List dict profil kandidat dari active_candidates.
            Setiap dict harus mengandung current_position, years_experience,
            top_skills, dan resume_summary.

    Returns:
        System prompt yang sudah diisi dengan semua profil kandidat.
    """
    candidate_count = len(candidates)

    profile_blocks = []
    for i, candidate in enumerate(candidates, start=1):
        skills = candidate.get("top_skills") or []
        skills_str = ", ".join(skills) if skills else "Tidak tersedia"

        years_exp = candidate.get("years_experience")
        exp_str = f"{years_exp} tahun" if years_exp is not None else "Tidak tersedia"

        block = (
            f"=== Kandidat {i} ===\n"
            f"Posisi Terakhir: {candidate.get('current_position') or 'Tidak tersedia'}\n"
            f"Pengalaman: {exp_str}\n"
            f"Skill Utama: {skills_str}\n"
            f"Ringkasan Profil:\n{candidate.get('resume_summary') or 'Tidak tersedia'}\n"
        )
        profile_blocks.append(block)

    return EVALUATOR_SYSTEM_PROMPT.format(
        candidate_count=candidate_count,
        candidate_profiles="\n".join(profile_blocks),
    )