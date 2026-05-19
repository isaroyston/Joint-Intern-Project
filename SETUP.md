# Project Setup

This guide shows how to get the project running from a fresh clone.

## 1) Install Python and create a virtual environment

1. Open a terminal in the project root.
2. Create and activate a virtual environment.

```bash
python -m venv venv
```

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

3. Install the dependencies.

```bash
pip install -r requirements.txt
```

## 2) Create your `.env` file

Create a `.env` file in the project root and add these values:

```env
OPENAI_API_KEY=
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
EMBEDDING_DIMENSIONS=384

SENTINEL_API_KEY=
SENTINEL_API_URL=https://sentinel.stg.aiguardian.gov.sg/api/v1/validate

SUPABASE_URL=
SUPABASE_SERVICE_KEY=
SUPABASE_ANON_KEY=

FLASK_SECRET_KEY=dev-secret-key-change-in-production
RED_TEAM_EVAL_MODEL=gpt-4o-mini
```

Important:

- The code uses `SUPABASE_SERVICE_KEY`, not `SUPABASE_KEY`.
- `SUPABASE_ANON_KEY` is only needed if you use Supabase login or signup helpers.

## 3) Set up Supabase

### Part A: Set up Supabase project & install pgvector
If you are new to Supabase, follow these steps.

1. Go to [supabase.com](https://supabase.com) and sign in or create a free account.
2. Click **New project**.
3. Choose your organization, give the project a name, and set a database password.
4. Wait for the project to finish creating.
5. Open your project dashboard.
6. In the left menu, go to **SQL Editor**.
7. Click **New query**.
8. Copy and paste the SQL below.
9. Click **Run**.

Run this first to create the tables and search function the app needs:

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    balance NUMERIC DEFAULT 0,
    daily_limit NUMERIC DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(255) DEFAULT 'Untitled Conversation',
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS messages (
    id UUID PRIMARY KEY,
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role VARCHAR(50) NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY,
    content TEXT NOT NULL,
    embedding vector(384),
    source VARCHAR(255) NOT NULL,
    doc_type VARCHAR(50) DEFAULT 'policy' CHECK (doc_type IN ('policy', 'rag', 'system')),
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source);
CREATE INDEX IF NOT EXISTS idx_documents_doc_type ON documents(doc_type);
CREATE INDEX IF NOT EXISTS idx_documents_embedding ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE OR REPLACE FUNCTION search_documents(
    query_embedding vector,
    match_limit INT DEFAULT 5,
    match_threshold FLOAT DEFAULT 0.7
)
RETURNS TABLE (
    id UUID,
    content TEXT,
    source VARCHAR,
    doc_type VARCHAR,
    metadata JSONB,
    similarity FLOAT
) LANGUAGE sql STABLE AS $$
SELECT
    documents.id,
    documents.content,
    documents.source,
    documents.doc_type,
    documents.metadata,
    1 - (documents.embedding <=> query_embedding) AS similarity
FROM
    documents
WHERE
    1 - (documents.embedding <=> query_embedding) > match_threshold
ORDER BY
    documents.embedding <=> query_embedding
LIMIT
    match_limit;
$$;

-- Sessions table
CREATE TABLE IF NOT EXISTS sessions (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    conversation_id UUID REFERENCES conversations(id) ON DELETE SET NULL,
    ip_address VARCHAR(45),
    user_agent TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NOT NULL,
    is_active BOOLEAN DEFAULT true
);

-- Audit logs table
CREATE TABLE IF NOT EXISTS audit_logs (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    action VARCHAR(100) NOT NULL,
    resource VARCHAR(100) NOT NULL,
    details JSONB DEFAULT '{}' :: jsonb,
    status VARCHAR(50) DEFAULT 'success' CHECK (status IN ('success', 'failed', 'alert')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Message flags table
CREATE TABLE IF NOT EXISTS message_flags (
    id UUID PRIMARY KEY,
    message_id UUID NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    reason VARCHAR(100) NOT NULL,
    details JSONB DEFAULT '{}' :: jsonb,
    flagged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    reviewed BOOLEAN DEFAULT false,
    reviewer_id UUID REFERENCES users(id) ON DELETE SET NULL,
    review_notes TEXT
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id);
CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_messages_user_id ON messages(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id ON audit_logs(user_id);
```

Why these tables matter (short):

- `users` — application profile state (stores `balance` and `daily_limit`) and links to conversations and sessions.
- `conversations` — groups of messages representing a chat session.
- `messages` — individual chat records used for history, auditing and red-team review.
- `documents` — policy/source text chunks with vector embeddings used for retrieval.
- `search_documents` — server-side function that performs vector similarity search for retrieval-augmented generation (RAG).

### Part B: Create required test accounts (Auth + profile)
After you run the SQL, check that the **Table Editor** shows `users`, `conversations`, `messages`, and `documents`.

The chatbot reads and writes to these tables, and the ingestion script stores the policy documents in `documents`.

If you are starting from scratch, make sure pgvector is enabled in Supabase before running ingestion.

Important:

- If you are only testing the database, you can create the row directly in the `users` table with `balance` and `daily_limit`.
- If you want the person to actually log in to the app, create the user in **Authentication** first, then make the matching `users` row.

Optional: create test accounts manually in Supabase.

If you want to make a test account by hand, do it in this order:

1. Go to **Authentication** in the Supabase dashboard.
2. Open **Users**.
3. Click **Add user**.
4. Enter the email and password.
5. Save the Auth user.
6. Open **Table Editor**.
7. Go to the `users` table.
8. Create the matching profile row for that same user id.

The `users` table row should contain these fields:

- `id`
- `email`
- `metadata`
- `created_at`
- `updated_at`
- `balance`
- `daily_limit`

If you want to backfill a batch of test users and randomize their balances, paste this in the SQL Editor:

```sql
insert into public.users (id, email, balance, daily_limit, metadata, created_at, updated_at)
select
    id,
    email,
    floor(random() * 100000)::int,
    floor(500 + random() * 9500)::int,
    jsonb_build_object('created_via', 'auth_backfill'),
    now(),
    now()
from auth.users
where email ~ '^user([1-9]|[12][0-9]|30)@test\.com$'
on conflict (id) do update
set
    email = excluded.email,
    balance = excluded.balance,
    daily_limit = excluded.daily_limit,
    updated_at = now();
```

If you already created these tables in Supabase, you only need to confirm that the `search_documents` function exists.

## 4) Load the policy documents into Supabase

The chatbot uses the PDF files in `src/documents/` as its knowledge base. Run the ingestion script once after Supabase is ready:

```bash
python ingest.py
```

What this does:

- Reads every PDF in `src/documents/`
- Splits the documents into chunks
- Creates embeddings with OpenAI
- Stores the chunks in Supabase `documents`
- Removes old policy chunks first, so rerunning the script refreshes the data

If the script finishes without errors, the retrieval layer is ready.

## 5) Start the chatbot app

Run the Flask chatbot UI:

```bash
python app.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:5000
```

If you prefer the CLI version, run:

```bash
python main.py
```

## 6) Start the red-team tools

The red-team dashboard needs the chatbot API running first.

1. Start the API server:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

2. In a second terminal, start the dashboard:

```bash
streamlit run attacks/streamlit_app.py
```

3. In the dashboard, set:

- Base URL: `http://127.0.0.1:8000`
- Chat path: `/chat`
- Reset path: `/reset`
- Request prompt field: `message`
- Response field: `response`

## 7) Quick run order Summary

Use this order when setting up the project for the first time:

1. Create and activate `venv`
2. Install requirements
3. Add `.env`
4. Set up Supabase tables and pgvector
5. Run `python ingest.py`
6. Start `python app.py`
7. Start `uvicorn api:app --port 8000`
8. Start `streamlit run attacks/streamlit_app.py`

## 8) Common issues

- If `ingest.py` fails, check `OPENAI_API_KEY`, `SUPABASE_URL`, and `SUPABASE_SERVICE_KEY` first.
- If the chatbot cannot connect to Supabase, confirm the service key is correct and the tables exist.
- If the red-team dashboard cannot reach the bot, make sure the API server is running before opening Streamlit.
