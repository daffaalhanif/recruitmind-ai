# LLM
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0

# Embedding
EMBEDDING_MODEL = "text-embedding-3-small"
VECTOR_SIZE = 1536  # Dimensi vector sesuai model text-embedding-3-small

# Qdrant
COLLECTION_NAME = "recruitmind_resumes"  # Nama collection di Qdrant, harus konsisten dengan yang dipakai di qdrant_service
PAYLOAD_PREVIEW_LENGTH = 200

# Path file database lokal
DB_PATH = "data/recruitmind.db"
CHECKPOINT_DB_PATH = "data/checkpoints.db"

# Retrieval
TOP_K_RETRIEVAL = 15
TOP_K_RERANK = 5

# Pipeline ETL
EXTRACTION_BATCH_SIZE = 10  # Jumlah resume per batch sebelum jeda
EXTRACTION_BATCH_DELAY = 2  # Jeda antar batch dalam detik untuk menghindari rate limit OpenAI API
INGEST_BATCH_SIZE = 100     # Jumlah resume per batch saat upload ke Qdrant

# Session management
MAX_ACTIVE_SESSIONS = 5
SLIDING_WINDOW_SIZE = 10
SESSION_TITLE_MAX_WORDS = 5

# Input validation
MAX_QUERY_LENGTH = 8000
JD_DETECTION_MIN_LENGTH = 500