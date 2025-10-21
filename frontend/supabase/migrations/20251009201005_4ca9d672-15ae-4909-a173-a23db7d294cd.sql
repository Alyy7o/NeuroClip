-- Clean up duplicate policies on user_videos table
-- The new comprehensive policies are already in place, remove the old ones
DROP POLICY IF EXISTS "Users can view own videos" ON public.user_videos;
DROP POLICY IF EXISTS "Users can insert own videos" ON public.user_videos;
DROP POLICY IF EXISTS "Users can update own videos" ON public.user_videos;
DROP POLICY IF EXISTS "Users can delete own videos" ON public.user_videos;
DROP POLICY IF EXISTS "Public videos viewable by all" ON public.user_videos;

-- The following policies are already in place and provide comprehensive protection:
-- 1. "Authenticated users full access to own videos" - allows authenticated users full CRUD on their videos
-- 2. "Deny anonymous access to private videos" - allows anonymous users to only see public videos