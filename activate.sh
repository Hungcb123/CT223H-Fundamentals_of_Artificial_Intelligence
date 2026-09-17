#!/usr/bin/env bash
# ==========================================================
# Script kích hoạt Virtual Environment nhanh
# Sử dụng: source activate.sh  (hoặc chạy ./activate.sh)
# ==========================================================

VENV_PATH="/home/hung/.venvs/ocr_service"

if [ ! -f "$VENV_PATH/bin/activate" ]; then
    echo "❌ Không tìm thấy virtualenv tại: $VENV_PATH"
    return 1 2>/dev/null || exit 1
fi

# Nếu script được chạy trực tiếp (./activate.sh hoặc bash activate.sh)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "⚡ Đang mở shell mới với môi trường ảo kích hoạt sẵn..."
    echo "💡 Gõ 'exit' khi muốn thoát môi trường ảo này."
    bash --init-file <(echo "source $VENV_PATH/bin/activate; echo '✅ Đã kích hoạt virtualenv: $VENV_PATH'; echo 'Python: \$(which python)'; echo 'Kaggle: \$(which kaggle)'")
else
    # Nếu script được source (source activate.sh hoặc . activate.sh)
    source "$VENV_PATH/bin/activate"
    echo "✅ Đã kích hoạt virtualenv: $VENV_PATH"
    echo "Python: $(which python)"
    echo "Kaggle: $(which kaggle)"
fi
