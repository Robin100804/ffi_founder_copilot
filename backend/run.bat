@echo off
setlocal

REM In den Ordner wechseln, in dem dieses Skript liegt
cd /d %~dp0

echo 🔧 Prüfe virtuelle Umgebung...

IF NOT EXIST venv (
    echo ⚙️  Erstelle virtuelle Umgebung...
    python -m venv venv
)

echo ✅ Aktiviere virtuelle Umgebung...
call venv\Scripts\activate

IF EXIST requirements.txt (
    echo 📦 Installiere Abhängigkeiten aus requirements.txt...
    pip install -r requirements.txt
)

echo 🚀 Starte Uvicorn unter http://127.0.0.1:8000 ...
uvicorn main:app --reload

pause
