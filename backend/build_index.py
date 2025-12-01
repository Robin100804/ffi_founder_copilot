import os
import uuid
import requests
import chromadb
import pdfplumber
from docx import Document

# Ordner mit deinen FFI-Texten (Playbooks, Leitfäden, Event-Docs, etc.)
DATA_DIR = "data"

# Ordner für die Vektordatenbank (muss zu main.py passen)
DB_DIR = "chroma_db"

# Embedding-Setup (muss zu main.py passen)
EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"

# Chroma-Client + Collection (Name muss zu main.py passen!)
client = chromadb.PersistentClient(path=DB_DIR)
collection = client.get_or_create_collection("ffi_founder_docs")


# ─────────────────────────────
# TEXT-EXTRAKTION
# ─────────────────────────────

def extract_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def extract_pdf(path: str) -> str:
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
    return text


def extract_docx(path: str) -> str:
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs)


def extract_file(path: str) -> str:
    path_lower = path.lower()

    if path_lower.endswith(".txt") or path_lower.endswith(".md"):
        return extract_txt(path)

    if path_lower.endswith(".pdf"):
        return extract_pdf(path)

    if path_lower.endswith(".docx"):
        return extract_docx(path)

    raise ValueError(f"Unsupported file type: {path}")


# ─────────────────────────────
# CHUNKING
# ─────────────────────────────

def chunk_text(text: str, max_chars: int = 800, overlap: int = 150):
    """
    Zerlegt längere Texte in überlappende Chunks,
    damit die Embeddings nicht zu lang werden.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = min(len(text), start + max_chars)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = end - overlap

    return chunks


# ─────────────────────────────
# EMBEDDING VIA OLLAMA
# ─────────────────────────────

def get_embedding(text: str) -> list[float]:
    resp = requests.post(
        OLLAMA_EMBED_URL,
        json={"model": EMBEDDING_MODEL, "prompt": text},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


# ─────────────────────────────
# INDEXIERUNG
# ─────────────────────────────

def index_documents():
    os.makedirs(DATA_DIR, exist_ok=True)

    files = [
        f for f in os.listdir(DATA_DIR)
        if f.lower().endswith((".txt", ".md", ".pdf", ".docx"))
    ]

    if not files:
        print("Keine unterstützten Dateien in ./data gefunden.")
        return

    for filename in files:
        full_path = os.path.join(DATA_DIR, filename)

        print(f"\n📄 Lese Datei: {filename} ...")
        text = extract_file(full_path)

        chunks = chunk_text(text)
        print(f" → {len(chunks)} Chunks erzeugt.")

        ids = []
        docs = []
        metas = []
        embeds = []

        for i, chunk in enumerate(chunks):
            emb = get_embedding(chunk)
            chunk_id = str(uuid.uuid4())

            ids.append(chunk_id)
            docs.append(chunk)
            metas.append({"source": filename, "chunk": i})
            embeds.append(emb)

        collection.add(
            ids=ids,
            embeddings=embeds,
            documents=docs,
            metadatas=metas,
        )

        print(f" ✔ Indexiert: {filename}")

    print("\n🎉 Fertig! Alle Dokumente wurden indexiert.\n")


if __name__ == "__main__":
    index_documents()
