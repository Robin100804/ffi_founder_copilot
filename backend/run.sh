#!/usr/bin/env bash
set -e

# In den Ordner wechseln, in dem dieses Skript liegt
cd "$(dirname "$0")"

echo "🔧 Prüfe virtuelle Umgebung..."

# venv anlegen, falls noch nicht vorhanden
if [ ! -d "venv" ]; then
    echo "⚙️  Erstelle virtuelle Umgebung..."
    python3 -m venv venv
fi

# venv aktivieren
echo "✅ Aktiviere virtuelle Umgebung..."
source venv/bin/activate

# Dependencies installieren (falls requirements.txt vorhanden)
if [ -f "requirements.txt" ]; then
    echo "📦 Installiere Abhängigkeiten aus requirements.txt..."
    pip install -r requirements.txt
fi

# Server starten
echo "🚀 Starte Uvicorn unter http://127.0.0.1:8000 ..."
uvicorn main:app --reload
