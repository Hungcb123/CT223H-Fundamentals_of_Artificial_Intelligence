#!/usr/bin/env bash
# ==========================================================
# Script tự động tải và giải nén Dataset BreakHis từ Kaggle
# Sử dụng: ./download_breakhis.sh
# ==========================================================

set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_KAGGLE="/home/hung/.venvs/ocr_service/bin/kaggle"

if [ ! -f "$VENV_KAGGLE" ]; then
    echo "❌ Không tìm thấy kaggle trong virtualenv: $VENV_KAGGLE"
    echo "Đang cài đặt kaggle vào virtualenv..."
    /home/hung/.venvs/ocr_service/bin/pip install kaggle
fi

echo "=========================================================="
echo "⚡ Đang kiểm tra cấu hình Kaggle API..."
echo "=========================================================="

# Đảm bảo credentials từ .env được nạp
if [ -f "$DIR/.env" ]; then
    export $(grep -v '^#' "$DIR/.env" | xargs)
fi

mkdir -p "$DIR/breakhis/data"

echo "📥 Đang tải dataset BreakHis từ Kaggle và tự động giải nén..."
echo "Thư mục đích: $DIR/breakhis/data"
echo "=========================================================="

"$VENV_KAGGLE" datasets download -d ambarish/breakhis -p "$DIR/breakhis/data" --unzip

echo "=========================================================="
echo "🎉 Hoàn tất! Dữ liệu BreakHis đã sẵn sàng trong: $DIR/breakhis/data"
echo "=========================================================="
