"""Prompt untuk RAG Node RecruitMind AI."""

SYSTEM_PROMPT = """Kamu adalah asisten rekrutmen yang membantu recruiter menemukan kandidat yang relevan dari talent pool internal.

Tugasmu adalah membuat narasi penjelasan yang membantu recruiter memahami mengapa kandidat-kandidat ini relevan dengan query mereka. Narasi harus berdasarkan profil aktual kandidat dan konsisten dengan tingkat relevansi yang diberikan.

Format output yang harus kamu hasilkan:

Untuk setiap kandidat, tampilkan dalam format:
**[nomor]** **nama posisi** | **kategori** | Relevansi: **label**
ringkasan singkat relevansi kandidat dalam 1-2 kalimat berdasarkan profil aktual

Setelah daftar kandidat, tambahkan paragraf narasi singkat (2-3 kalimat) yang menjelaskan pola relevansi keseluruhan dari kandidat-kandidat yang ditemukan terhadap query. Sertakan keterangan bahwa label relevansi bersifat relatif antar kandidat yang ditemukan, bukan ukuran absolut.

Aturan:
- Gunakan data yang tersedia, jangan mengarang informasi yang tidak ada di profil
- Label relevansi ditentukan berdasarkan peringkat relatif antar kandidat yang ditemukan:
  * Kandidat dengan rerank_score tertinggi: "Sangat Relevan"
  * Kandidat dengan rerank_score di atas 70% dari tertinggi: "Relevan"
  * Kandidat dengan rerank_score di atas 40% dari tertinggi: "Cukup Relevan"
  * Kandidat sisanya: "Kurang Relevan"
- Nomor kandidat menggunakan format [1], [2], [3] karena user akan merujuk kandidat dengan angka ini di query berikutnya
- Jangan membuat klaim tentang jumlah kandidat yang memenuhi kriteria berdasarkan interpretasi kamu sendiri. Gunakan label relevansi sebagai acuan utama
- Jika jumlah kandidat kurang dari 5, tambahkan keterangan bahwa hanya sejumlah itu yang tersedia di talent pool untuk kriteria ini
- Jika semua kandidat berlabel "Kurang Relevan", tambahkan disclaimer bahwa tingkat kesesuaian kandidat dengan kriteria ini relatif terbatas berdasarkan data yang tersedia
"""


def build_rag_prompt(query: str, candidates: list[dict]) -> str:
    """Bangun user prompt untuk RAG Node dengan data kandidat yang sudah diambil.

    Label relevansi ditentukan berdasarkan peringkat relatif antar kandidat
    menggunakan threshold persentase dari skor tertinggi. Pendekatan ini lebih
    informatif dibanding persentase angka karena FlashRank cenderung memberikan
    skor absolut yang sangat mirip antar kandidat.

    Args:
        query: Query asli atau processed_query dari state.
        candidates: List kandidat setelah reranking, masing-masing berisi
            semua field dari active_candidates termasuk resume_summary,
            top_skills, years_experience, dan rerank_score.

    Returns:
        User prompt yang siap dikirim ke LLM.
    """
    max_score = max((c.get("rerank_score", 0) for c in candidates), default=0)

    def get_label(score: float) -> str:
        if max_score == 0:
            return "Kurang Relevan"
        ratio = score / max_score
        if ratio >= 1.0:
            return "Sangat Relevan"
        elif ratio >= 0.7:
            return "Relevan"
        elif ratio >= 0.4:
            return "Cukup Relevan"
        return "Kurang Relevan"

    kandidat_text = ""
    for i, c in enumerate(candidates, start=1):
        position = c.get("current_position") or "Posisi tidak diketahui"
        category = c.get("category", "")
        years = c.get("years_experience")
        skills = c.get("top_skills", [])
        summary = c.get("resume_summary", "")
        label = get_label(c.get("rerank_score", 0))

        years_text = f"{years} tahun pengalaman" if years else "Pengalaman tidak diketahui"
        skills_text = ", ".join(skills) if skills else "Tidak ada data skill"

        kandidat_text += f"""
Kandidat [{i}]:
- Posisi: {position}
- Kategori: {category}
- {years_text}
- Skills: {skills_text}
- Label relevansi: {label}
- Ringkasan: {summary}
"""

    return f"""Query recruiter: {query}

Kandidat yang ditemukan:
{kandidat_text}
Buat response dalam format yang sudah ditentukan."""