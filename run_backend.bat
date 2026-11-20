@echo off
cd /d %~dp0backend
if not exist .env copy .env.example .env
python -m venv .venv
call .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
