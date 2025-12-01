from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import requests
import chromadb

# Adresse deines lokal laufenden Ollama-Servers
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
EMBEDDING_MODEL = "nomic-embed-text"

# System-Prompt für den FFI-Gründungsassistenten
SYSTEM_PROMPT = (
    "Du bist der offizielle Gründungsassistent der Future Founders Initiative (FFI). "
    "Deine Aufgabe ist es, Nutzer*innen als kritischer, analytischer Sparringspartner "
    "bei der Entwicklung und Validierung von Startup-Ideen zu unterstützen. "
    "Du bewertest und verbesserst nicht die Person, sondern die Idee, die Logik und die Umsetzbarkeit.\n\n"

    "💬 STIL:\n"
    "- Direkt, analytisch, präzise.\n"
    "- Keine Motivationsfloskeln, kein Marketing-Sprech, kein Bullshit.\n"
    "- Kein unkritisches Bestätigen – Wahrheit vor Zustimmung.\n"
    "- Stelle Rückfragen, wenn Informationen fehlen.\n"
    "- Schreibe wie ein erfahrener Gründer, der radikal ehrlich unterstützt.\n"
    "- Keine langen Belehrungen, kein generischer Startup-Ratgeber.\n\n"

    "🎯 FOKUS:\n"
    "- Identifiziere immer zuerst das PROBLEM, nicht die Lösung.\n"
    "- Analysiere Annahmen, Risiken, logische Lücken, Inkonsistenzen.\n"
    "- Zeige mindestens 2–3 alternative Perspektiven auf.\n"
    "- Reduziere jede Idee auf: Problem → Zielgruppe → Value Proposition → Hypothesen → Tests.\n"
    "- Arbeite konsequent in Experimenten (Interview, Landing Page, Pre-Sale, Shadow-Test etc.).\n"
    "- Entwickle konkrete nächste Schritte, keine Theorie.\n\n"

    "📐 STRUKTUR DER ANTWORT (IMMER GENAU SO):\n"
    "1. Kurzfazit (2–4 Sätze): ehrliche Bewertung der Idee und der logischen Struktur.\n"
    "2. Kritische Analyse (Bulletpoints): Annahmen, Risiken, Schwächen, fehlende Infos.\n"
    "3. Alternativen (Bulletpoints): 2–3 andere Problem- oder Zielgruppenperspektiven.\n"
    "4. Nächste Schritte (präzise To-Dos): kleine, sofort ausführbare Validierungsschritte.\n"
    "Formatiere deine Antwort IMMER mit:\n"
    "- klaren Absätzen zwischen den Blöcken,\n"
    "- Bulletpoints in Analyse- und Alternativen-Teil,\n"
    "- maximal 4–7 Sätzen pro Abschnitt,\n"
    "- keinerlei Motivationstext oder unnötige Füllwörter.\n"
   "Wenn der Nutzer nach der Quelle fragt, dann nenne ausschließlich die Quellen aus den RAG-" "Snippets (Dateiname in eckigen Klammern). "
"Wenn KEINE Snippets vorhanden sind, antworte: „Ich habe keinen Kontext aus der Wissensbasis erhalten. "
"Erfinde niemals eine Datei oder Quelle und behaupte niemals, du hättest keinen Zugriff, wenn Snippets vorhanden waren."
)

app = FastAPI()

# ───────────────── CORS, damit dein Frontend zugreifen darf ────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # später gerne einschränken (z.B. auf deine Domain)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ───────────────── Static-Files (Logo, CSS, JS, …) ────────────────────────
# WICHTIG: Wir mounten das Verzeichnis "static" relativ zu diesem backend-Ordner.
# Dein Logo liegt unter: backend/static/ffi-logo.png
app.mount("/static", StaticFiles(directory="static"), name="static")

# ──────────────── Vektordatenbank initialisieren ────────────────
# Hier speicherst du FFI-Wissen: Leitfäden, Event-Formate, Playbooks, Beispiele etc.
chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection("ffi_founder_docs")


def get_embedding(text: str) -> list[float]:
    resp = requests.post(
        OLLAMA_EMBED_URL,
        json={"model": EMBEDDING_MODEL, "prompt": text},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["embedding"]


def retrieve_context(user_text: str, k: int = 4) -> str:
    """
    Holt die k ähnlichsten Textstücke aus deiner FFI-Wissensbasis
    (z.B. Leitfäden, Event-Playbooks, Beispiele, Checklisten).
    """
    try:
        query_emb = get_embedding(user_text)
    except Exception:
        # Wenn Embedding fehlschlägt, lieber ohne Kontext weitermachen
        return ""

    results = collection.query(
        query_embeddings=[query_emb],
        n_results=k,
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    if not docs:
        return ""

    snippets = []
    for doc, meta in zip(docs, metas):
        source = meta.get("source", "unbekannt")
        snippets.append(f"[{source}] {doc}")

    context = "\n\n---\n\n".join(snippets)
    return context


# ───────────────── Modelle für die Request-Validierung ─────────────────────

class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]


# ───────────────── Routen ───────────────────────────────────────────────────

@app.get("/")
def read_root():
    """
    Liefert deine index.html aus (FFI Founder Chat UI).
    Die Datei liegt im gleichen Ordner wie main.py.
    """
    return FileResponse("index.html")


@app.post("/chat")
def chat(req: ChatRequest):
    """
    Erwartet vom Frontend ein JSON der Form:
    {
        "messages": [
            { "role": "user", "content": "Meine Startup-Idee / Situation ..." }
        ]
    }

    Der Assistent antwortet als FFI-Gründungscoach mit kritischem Sparring,
    basierend auf System-Prompt + optionalem FFI-Kontext aus der Vektordatenbank.
    """
    if not req.messages:
        raise HTTPException(status_code=400, detail="Keine Nachrichten erhalten.")

    user_text = req.messages[-1].content

    # 1) Relevanten Kontext aus deiner FFI-Vektordatenbank holen
    context = retrieve_context(user_text)

    # 2) Systemnachrichten bauen
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if context:
        messages.append({
            "role": "system",
            "content": (
                "Du erhältst nun Auszüge aus internen Materialien der Future Founders Initiative (FFI), "
                "z.B. Event-Playbooks, Leitfäden, Beispiele und Checklisten. "
                "Nutze diese Inhalte, um deine Antworten als FFI-Gründungsassistent mit konkreten Methoden, "
                "Begriffen und Beispielen zu unterfüttern. "
                "Zitiere daraus nur, wenn es den Nutzer wirklich weiterbringt.\n\n"
                f"{context}"
            ),
        })

    # 3) User-Nachricht anhängen
    messages.append({"role": "user", "content": user_text})

    payload = {
        "model": "llama3.2",
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "top_p": 0.8,
        },
    }

    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()

        answer = data.get("message", {}).get("content")
        if answer is None:
            raise ValueError("Antwort von Ollama enthielt kein 'message.content'.")

        return {"reply": answer}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ollama-Fehler: {e}")
