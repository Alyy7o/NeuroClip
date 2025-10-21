-- Drop existing policies to recreate them with explicit anonymous protection
DROP POLICY IF EXISTS "Users can view own profile" ON public.profiles;
DROP POLICY IF EXISTS "Users can update own profile" ON public.profiles;
DROP POLICY IF EXISTS "Users can view own history" ON public.processing_history;
DROP POLICY IF EXISTS "Users can insert own history" ON public.processing_history;
DROP POLICY IF EXISTS "Users can update own history" ON public.processing_history;

-- Profiles table: Explicit protection against anonymous access
CREATE POLICY "Authenticated users can view own profile"
  ON public.profiles FOR SELECT
  TO authenticated
  USING (auth.uid() = id AND auth.uid() IS NOT NULL);

CREATE POLICY "Authenticated users can update own profile"
  ON public.profiles FOR UPDATE
  TO authenticated
  USING (auth.uid() = id AND auth.uid() IS NOT NULL);

-- Deny all access to anonymous users on profiles
CREATE POLICY "Deny anonymous access to profiles"
  ON public.profiles FOR ALL
  TO anon
  USING (false);

-- Processing history: Explicit protection against anonymous access
CREATE POLICY "Authenticated users can view own history"
  ON public.processing_history FOR SELECT
  TO authenticated
  USING (auth.uid() = user_id AND auth.uid() IS NOT NULL);

CREATE POLICY "Authenticated users can insert own history"
  ON public.processing_history FOR INSERT
  TO authenticated
  WITH CHECK (auth.uid() = user_id AND auth.uid() IS NOT NULL);

CREATE POLICY "Authenticated users can update own history"
  ON public.processing_history FOR UPDATE
  TO authenticated
  USING (auth.uid() = user_id AND auth.uid() IS NOT NULL);

-- Deny all access to anonymous users on processing_history
CREATE POLICY "Deny anonymous access to processing_history"
  ON public.processing_history FOR ALL
  TO anon
  USING (false);

-- User videos: Explicit protection (except for public videos)
CREATE POLICY "Deny anonymous access to private videos"
  ON public.user_videos FOR ALL
  TO anon
  USING (is_public = true);

-- Ensure authenticated users can manage their own videos
CREATE POLICY "Authenticated users full access to own videos"
  ON public.user_videos FOR ALL
  TO authenticated
  USING (auth.uid() = user_id AND auth.uid() IS NOT NULL);