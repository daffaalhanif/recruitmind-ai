"""Prompt untuk Generator Node yang membuat pertanyaan wawancara per kandidat.

Build function menerima satu profil kandidat dari active_candidates dan
menyusunnya ke dalam system prompt.
"""

GENERATOR_SYSTEM_PROMPT = """Kamu adalah interviewer rekrutmen berpengalaman.

Kamu akan menerima profil satu kandidat. Tugasmu: buat 7 pertanyaan wawancara \
yang SPESIFIK terhadap profil kandidat ini.

Format output:
1. [Pertanyaan pertama]
2. [Pertanyaan kedua]
... dan seterusnya

Aturan WAJIB:
- Setiap pertanyaan harus merujuk ke detail spesifik dari profil: posisi yang \
pernah dijabat, skill tertentu, atau konteks dari ringkasan profil
- DILARANG membuat pertanyaan generik yang bisa berlaku untuk kandidat manapun. \
Contoh yang dilarang: "Apa kelebihan dan kekurangan kamu?", \
"Di mana kamu melihat diri kamu 5 tahun ke depan?"
- Bahasa Indonesia

---

Profil Kandidat:

Posisi Terakhir: {current_position}
Skill Utama: {top_skills}
Ringkasan Profil:
{resume_summary}
"""


def build_generator_prompt(candidate: dict) -> str:
    """Bangun system prompt untuk generasi pertanyaan wawancara.

    Mengisi template dengan data profil satu kandidat dari active_candidates.
    Tidak ada query SQLite karena semua field sudah tersedia di active_candidates.

    Args:
        candidate: Dict profil kandidat dari active_candidates. Membutuhkan
            field current_position, top_skills, dan resume_summary.

    Returns:
        System prompt yang sudah diisi dengan profil kandidat.
    """
    skills = candidate.get("top_skills") or []
    skills_str = ", ".join(skills) if skills else "Tidak tersedia"

    return GENERATOR_SYSTEM_PROMPT.format(
        current_position=candidate.get("current_position") or "Tidak tersedia",
        top_skills=skills_str,
        resume_summary=candidate.get("resume_summary") or "Tidak tersedia",
    )