-- Fix RLS policies on profiles table
DROP POLICY IF EXISTS "Authenticated users can view own profile" ON public.profiles;
DROP POLICY IF EXISTS "Authenticated users can update own profile" ON public.profiles;
DROP POLICY IF EXISTS "Deny anonymous access to profiles" ON public.profiles;

CREATE POLICY "Users can view own profile"
  ON public.profiles FOR SELECT
  TO authenticated
  USING (auth.uid() = id);

CREATE POLICY "Users can update own profile"
  ON public.profiles FOR UPDATE
  TO authenticated
  USING (auth.uid() = id)
  WITH CHECK (auth.uid() = id);

-- Fix RLS policies on processing_history table
DROP POLICY IF EXISTS "Authenticated users can view own history" ON public.processing_history;
DROP POLICY IF EXISTS "Authenticated users can insert own history" ON public.processing_history;
DROP POLICY IF EXISTS "Authenticated users can update own history" ON public.processing_history;

CREATE POLICY "Users can view own history"
  ON public.processing_history FOR SELECT
  TO authenticated
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own history"
  ON public.processing_history FOR INSERT
  TO authenticated
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own history"
  ON public.processing_history FOR UPDATE
  TO authenticated
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

-- Fix RLS policies on user_videos table
DROP POLICY IF EXISTS "Authenticated users full access to own videos" ON public.user_videos;
DROP POLICY IF EXISTS "Deny anonymous access to private videos" ON public.user_videos;

CREATE POLICY "Users can manage own videos"
  ON public.user_videos FOR ALL
  TO authenticated
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Public videos viewable by everyone"
  ON public.user_videos FOR SELECT
  USING (is_public = true);