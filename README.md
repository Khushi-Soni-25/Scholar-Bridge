# ScholarBridge — Scholarship Discovery Platform

A student-first scholarship discovery website built around the supplied Gujarat scholarship knowledge base, recommendation engine, Hybrid RAG, LangChain and LangGraph work.

## Stack
- Frontend: React + Vite + Framer Motion + Lucide
- Backend: FastAPI
- Database: MongoDB Atlas
- Recommendation: supplied deterministic ScholarBridge engine
- RAG: supplied Qdrant/BM25 hybrid retrieval
- AI orchestration: supplied LangChain + LangGraph workflow

## Architecture
`React UI → FastAPI → MongoDB Atlas`

`FastAPI → ScholarBridge recommendation engine`

`FastAPI → Hybrid RAG → evidence`

`FastAPI → LangChain → grounded explanation`

`FastAPI → LangGraph → workflow orchestration`

Eligibility is determined by the deterministic rule matcher. RAG/LLM components explain and retrieve evidence; they are not used to override eligibility.

## 1. MongoDB Atlas
1. Create a MongoDB Atlas cluster and database.
2. Copy `backend/.env.example` to `backend/.env`.
3. Set `MONGODB_URI`, `MONGODB_DB_NAME` and a strong `JWT_SECRET`.
4. Install backend requirements.
5. Run `python backend/seed_mongodb.py` from the project root (or run it with the backend virtual environment active).

The seed imports the supplied master scholarship, rules, benefits, documents and portal CSVs.

## 2. Backend
Windows:

`run_backend.bat`

Manual:

```bash
cd backend
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

## 3. Frontend

```bash
cd frontend
npm install
npm run dev
```

Vite will normally open the frontend at `https://scholar-bridge-aicw.vercel.app/`.

Optional `.env` in frontend:

`VITE_API_URL=[http://localhost:8000/api](https://scholar-bridge-backend-cllv.onrender.com/)`

## Main flows
- Register/login with MongoDB-backed account
- Student profile persistence
- Recommendation engine matching
- Saved scholarships
- Scholarship directory + details
- Hybrid RAG evidence search
- LangChain grounded explanation endpoint
- LangGraph workflow endpoint
- Personal dashboard and search history

## Design direction
The UI intentionally uses a restrained editorial/e-commerce feel: large typography, quiet surfaces, thin borders, subtle green/lime accent, rounded cards, motion on entry/hover, and minimal decorative elements. It is inspired by modern fintech/product websites without copying any specific site.

## Important
The supplied datasets and engine files are preserved under `data/` and `backend/scholarbridge_engine/`. If your existing RAG environment requires a local Qdrant index or model cache, configure its existing environment variables before using the RAG endpoints.


---

## 👩‍💻 Developer

**Khushi Soni**
**Nandini Shah**
**Hemangi Patel**
**Dipali Patel**

---
