-- Remove the conflicting restrictive policy on processing_history
DROP POLICY IF EXISTS "Deny anonymous access to processing_history" ON public.processing_history;

-- The existing PERMISSIVE policies already properly restrict access:
-- 1. "Authenticated users can view own history" - SELECT only where auth.uid() = user_id
-- 2. "Authenticated users can insert own history" - INSERT only where auth.uid() = user_id  
-- 3. "Authenticated users can update own history" - UPDATE only where auth.uid() = user_id
-- These policies inherently block anonymous users (auth.uid() IS NULL won't match user_id)
-- and prevent authenticated users from accessing other users' data