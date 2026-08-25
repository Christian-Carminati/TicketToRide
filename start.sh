#!/usr/bin/env bash

# ==============================================================================
# Ticket to Ride RL Lab - Startup Script
# Avvia contemporaneamente il backend FastAPI e il frontend Vite (React).
# ==============================================================================

set -e

# Colori per il terminale
GOLD='\033[0;33m'
CYAN='\033[0;36m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color
BOLD='\033[1m'

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

echo -e "${GOLD}${BOLD}"
echo "  ╔══════════════════════════════════════════════════════════════╗"
echo "  ║             🚂 TICKET TO RIDE - RL LAB STUDIO                ║"
echo "  ║          Victorian Analytical & Multi-Agent Workbench        ║"
echo "  ╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# 1. Verifica ambiente virtuale Python
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
if [ ! -f "$PYTHON_BIN" ]; then
    echo -e "${RED}[!] Ambiente virtuale .venv non trovato in $ROOT_DIR/.venv${NC}"
    echo -e "${CYAN}[*] Tento di usare 'python3'...${NC}"
    PYTHON_BIN="python3"
fi

# 2. Verifica dipendenze frontend
if [ ! -d "$ROOT_DIR/frontend/node_modules" ]; then
    echo -e "${CYAN}[*] Installazione dipendenze frontend (npm install)...${NC}"
    (cd "$ROOT_DIR/frontend" && npm install)
fi

# 3. Gestione terminazione pulita su Ctrl+C
cleanup() {
    echo ""
    echo -e "${GOLD}[*] Chiusura dei servizi in corso...${NC}"
    if [ -n "$BACKEND_PID" ]; then
        kill "$BACKEND_PID" 2>/dev/null || true
    fi
    if [ -n "$FRONTEND_PID" ]; then
        kill "$FRONTEND_PID" 2>/dev/null || true
    fi
    wait 2>/dev/null || true
    echo -e "${GREEN}[✓] Servizi arrestati correttamente.${NC}"
    exit 0
}

trap cleanup SIGINT SIGTERM EXIT

# 4. Avvio Backend FastAPI (uvicorn)
echo -e "${CYAN}[1/2] Avvio Backend FastAPI su http://localhost:8000 ...${NC}"
export PYTHONPATH="$ROOT_DIR/src:$PYTHONPATH"
"$PYTHON_BIN" -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

# Attesa breve per l'avvio del backend
sleep 1.5

# 5. Avvio Frontend Vite
echo -e "${CYAN}[2/2] Avvio Frontend Vite su http://localhost:5173 ...${NC}"
(cd "$ROOT_DIR/frontend" && npm run dev) &
FRONTEND_PID=$!

echo ""
echo -e "${GREEN}${BOLD}══════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}${BOLD}  ✓ Servizi attivi con successo!${NC}"
echo -e "  • ${BOLD}Schermata di Gioco (Frontend):${NC}  ${CYAN}http://localhost:5173${NC}"
echo -e "  • ${BOLD}Backend API (FastAPI):${NC}          ${CYAN}http://localhost:8000${NC}"
echo -e "  • ${BOLD}Documentazione Swagger / OpenAPI:${NC} ${CYAN}http://localhost:8000/docs${NC}"
echo -e "${GREEN}${BOLD}══════════════════════════════════════════════════════════════════${NC}"
echo -e "${GOLD}Premi [CTRL+C] per arrestare entrambi i servizi.${NC}"
echo ""

# Rimani in ascolto dei processi
wait
