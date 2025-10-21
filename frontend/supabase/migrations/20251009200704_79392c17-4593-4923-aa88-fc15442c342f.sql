-- Create video metadata table for better video management
CREATE TABLE IF NOT EXISTS public.user_videos (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  title TEXT NOT NULL,
  description TEXT,
  original_filename TEXT NOT NULL,
  file_size BIGINT NOT NULL,
  duration REAL,
  format TEXT,
  resolution TEXT,
  thumbnail_url TEXT,
  video_url TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'uploaded' CHECK (status IN ('uploaded', 'processing', 'completed', 'failed')),
  is_public BOOLEAN NOT NULL DEFAULT false,
  tags TEXT[],
  metadata JSONB DEFAULT '{}'::jsonb,
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- Enable RLS
ALTER TABLE public.user_videos ENABLE ROW LEVEL SECURITY;

-- RLS Policies for user_videos
CREATE POLICY "Users can view own videos"
  ON public.user_videos FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own videos"
  ON public.user_videos FOR INSERT
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own videos"
  ON public.user_videos FOR UPDATE
  USING (auth.uid() = user_id);

CREATE POLICY "Users can delete own videos"
  ON public.user_videos FOR DELETE
  USING (auth.uid() = user_id);

-- Public videos viewable by everyone
CREATE POLICY "Public videos viewable by all"
  ON public.user_videos FOR SELECT
  USING (is_public = true);

-- Add trigger for updated_at
CREATE TRIGGER update_user_videos_updated_at
  BEFORE UPDATE ON public.user_videos
  FOR EACH ROW
  EXECUTE FUNCTION public.handle_updated_at();

-- Add index for better query performance
CREATE INDEX idx_user_videos_user_id ON public.user_videos(user_id);
CREATE INDEX idx_user_videos_status ON public.user_videos(status);
CREATE INDEX idx_user_videos_created_at ON public.user_videos(created_at DESC);
CREATE INDEX idx_user_videos_tags ON public.user_videos USING GIN(tags);

-- Enhance processing_history with foreign key to user_videos
ALTER TABLE public.processing_history 
  ADD COLUMN IF NOT EXISTS video_id UUID REFERENCES public.user_videos(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_processing_history_video_id ON public.processing_history(video_id);