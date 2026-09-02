# AI chatbot demo

Minimal production-oriented demo with a FastAPI/LangChain backend and a Next.js App Router frontend. The API keeps in-memory history by `session_id` and streams Claude output as SSE.

## Project tree

```text
backend/
  .env.example
  __init__.py
  chain.py
  main.py
  requirements.txt
frontend/
  .env.example
  app/
    globals.css
    layout.tsx
    page.tsx
  components/
    ChatInput.tsx
    ChatMessage.tsx
  next.config.mjs
  next-env.d.ts
  package.json
  postcss.config.mjs
  tailwind.config.ts
  tsconfig.json
```

## Run locally

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
Copy-Item backend\.env.example backend\.env
# Edit backend\.env and set ANTHROPIC_API_KEY
uvicorn backend.main:app --reload --port 8000
```

In a second terminal:

```powershell
cd frontend
npm install
Copy-Item .env.example .env.local
npm run dev
```

Open http://localhost:3000. The API health check is available at http://localhost:8000/health.

For deployment, replace the in-memory history store with Redis or a database and set `FRONTEND_ORIGIN` to the deployed frontend origin.
