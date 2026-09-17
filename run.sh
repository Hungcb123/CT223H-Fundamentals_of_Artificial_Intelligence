#!/usr/bin/env bash
# ==========================================================
# PneumoScan AI — One-click Runner
# ==========================================================

cd "$(dirname "$0")"

# Tìm python trong virtualenv có sẵn
if [ -f "/home/hung/.venvs/ocr_service/bin/python" ]; then
    PYTHON="/home/hung/.venvs/ocr_service/bin/python"
elif [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON="python3"
else
    PYTHON="python"
fi

echo "=========================================================="
echo "Khởi chạy PneumoScan AI Web Server"
echo "Python: $PYTHON"
echo "Địa chỉ: http://localhost:8000"
echo "Nhấn Ctrl + C để dừng server"
echo "=========================================================="

exec "$PYTHON" -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
