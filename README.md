# Backend (FastAPI + PostgreSQL) — Quick Steps

## 0) Requirements
- Python 3.11+ (works on 3.9+ too)
- Docker Desktop (optional but recommended for PostgreSQL)
- Git (optional)

## 1) Clone/Copy this folder
```bash
cd backend
cp .env.example .env
```

Edit `.env` if needed (passwords, DB name).

## 2) Start PostgreSQL (Docker) — Recommended
```bash
docker compose up -d
# postgres on port 5432, pgAdmin on http://localhost:5050
```

> If you already have PostgreSQL installed locally, you can skip Docker. Just ensure `DATABASE_URL` is correct.

## 3) Create virtual environment & install dependencies
```bash
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## 4) Run the API
```bash
uvicorn main:app --reload --port 8000
```

- Health check: `http://localhost:8000/health`
- List users: `GET http://localhost:8000/users`
- Create user: `POST http://localhost:8000/users` with JSON:
```json
{ "name": "Ali", "email": "ali@example.com" }
```

## Notes
- Tables are auto-created with `Base.metadata.create_all`. In production, use Alembic migrations.
- During development CORS allows all origins. Restrict in production.
