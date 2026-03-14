-- Enable pgvector (if not already enabled)
create extension if not exists vector;

-- Sentence-level embeddings per video (MiniLM: 384 dims)
create table if not exists public.video_sentence_embeddings (
  id uuid default gen_random_uuid() primary key,
  video_id text not null,
  sentence_index int not null,
  text text,
  start double precision,
  "end" double precision,
  embedding vector(384),
  created_at timestamptz default now() not null
);

-- Helpful indexes
create index if not exists video_sentence_embeddings_video_id_idx on public.video_sentence_embeddings (video_id);
-- Optional approximate index for vector similarity
-- create index if not exists video_sentence_embeddings_embedding_idx on public.video_sentence_embeddings using ivfflat (embedding) with (lists=100);

-- Note: RLS is not required for backend service-role inserts; if you enable RLS later,
-- add policies that allow SELECT for owners via a join to user_videos and INSERT via service role.