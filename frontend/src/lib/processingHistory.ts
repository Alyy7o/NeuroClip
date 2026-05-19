import { supabase } from '@/integrations/supabase/client';

type HistoryModule = 'blurring' | 'compression' | 'summarization';

async function ensureProfile(userId: string): Promise<void> {
  const { data: prof } = await supabase.from('profiles').select('id').eq('id', userId).maybeSingle();
  if (prof?.id) return;
  const { data: { user } } = await supabase.auth.getUser();
  if (!user || user.id !== userId) return;
  await supabase.from('profiles').upsert({
    id: user.id,
    email: user.email || `user_${userId.slice(0, 8)}@neuroclip.local`,
    full_name: (user.user_metadata?.full_name as string) || 'NeuroClip User',
  });
}

export async function startProcessingHistory(params: {
  userId: string;
  module: HistoryModule;
  inputType: 'file' | 'url';
  inputUrl?: string;
  query?: string;
}): Promise<{ historyId: string | null; error: string | null }> {
  await ensureProfile(params.userId);
  const { data, error } = await supabase
    .from('processing_history')
    .insert({
      user_id: params.userId,
      module: params.module,
      input_type: params.inputType,
      input_url: params.inputUrl ?? null,
      query: params.query ?? null,
      status: 'processing',
    })
    .select('id')
    .single();

  return {
    historyId: data?.id ?? null,
    error: error ? `${error.code ?? ''}: ${error.message}`.trim() : null,
  };
}

/** user_videos must exist before processing_history.video_id (FK). */
export async function completeProcessingHistory(
  historyId: string,
  params: {
    userId: string;
    resultUrl: string;
    videoId?: string;
    fileName?: string;
    duration?: number;
    fileSize?: number;
    module?: HistoryModule;
  }
): Promise<{ error: string | null }> {
  if (params.videoId) {
    const { error: uvErr } = await supabase.from('user_videos').upsert({
      id: params.videoId,
      user_id: params.userId,
      title: params.fileName ? `Processed: ${params.fileName}` : 'Processed video',
      original_filename: params.fileName || 'video.mp4',
      video_url: params.resultUrl,
      file_size: params.fileSize ?? 0,
      duration: params.duration ?? null,
      status: 'completed',
      metadata: params.module ? { module: params.module } : null,
    });
    if (uvErr) {
      return { error: `user_videos: ${uvErr.message}` };
    }
  }

  const { error } = await supabase
    .from('processing_history')
    .update({
      status: 'completed',
      result_url: params.resultUrl,
      ...(params.videoId ? { video_id: params.videoId } : {}),
    })
    .eq('id', historyId);

  return { error: error ? error.message : null };
}

export async function failProcessingHistory(
  historyId: string,
  message: string
): Promise<void> {
  await supabase
    .from('processing_history')
    .update({ status: 'failed', error_message: message })
    .eq('id', historyId);
}
