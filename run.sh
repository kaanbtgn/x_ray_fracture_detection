#!/usr/bin/env bash
set -e

# 1. Sanal ortam
if [[ -d "venv_tf" ]]; then
  source venv_tf/bin/activate
elif [[ -d "venv" ]]; then
  source venv/bin/activate
else
  python3 -m venv venv
  source venv/bin/activate
fi

# 2. Bağımlılıklar
pip install --upgrade pip
pip install -r requirements.txt

# 3. FastAPI sunucusu
python -m uvicorn app:app --host 0.0.0.0 --port 8501 --reload