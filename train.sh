#!/usr/bin/env bash
set -e  # first error → exit

# — project folders —
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DATA_DIR="$PROJECT_DIR/Bone_Fracture_Binary_Classification"
RUNS_DIR="$PROJECT_DIR/runs"          # log / checkpoint çıktılarını buraya taşıyacağız

# — virtual-env —
if [[ ! -d "$PROJECT_DIR/venv_tf" ]]; then
  echo "[+] Creating virtual environment (venv_tf)…"
  python3 -m venv "$PROJECT_DIR/venv_tf"
fi
source "$PROJECT_DIR/venv_tf/bin/activate"

# — deps —
echo "[+] Installing Python packages (only first run takes time)…"
python -m pip install --upgrade pip wheel setuptools
pip install -r "$PROJECT_DIR/requirements.txt"

# — sanity checks —
[[ -d "$DATA_DIR" ]] || { echo "❌  Dataset not found: $DATA_DIR"; exit 1; }

# — training —
echo "[+] Starting training…"
python "$PROJECT_DIR/train_eval_tf.py" \
  --data-dir "$DATA_DIR" \
  --epochs 20 \
  --batch-size 32

# — organise outputs —
mkdir -p "$RUNS_DIR"
mv checkpoints "$RUNS_DIR"/  2>/dev/null || true
mv logs         "$RUNS_DIR"/  2>/dev/null || true

echo "
✅  Training finished.  See results inside:  $RUNS_DIR
To leave the virtual-env:  deactivate
"