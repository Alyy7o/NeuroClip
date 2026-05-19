import { useCallback, useEffect, useMemo, useState } from 'react';
import { motion } from 'framer-motion';
import { History as HistoryIcon, ScissorsLineDashed, EyeOff, Minimize2, Calendar, FileVideo, Play, Download } from 'lucide-react';
import { DashboardLayout } from '@/components/DashboardLayout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { supabase } from '@/integrations/supabase/client';
import { useAuth } from '@/contexts/AuthContext';
import { useNavigate } from 'react-router-dom';
import { Badge } from '@/components/ui/badge';
import {
  downloadHistoryMedia,
  effectiveHistoryStatus,
  mediaFilename,
  resolveMediaPath,
} from '@/lib/historyMedia';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8040';

const moduleLabels: Record<string, string> = {
  summarization: 'Summarization',
  blurring: 'Blurring',
  compression: 'Compression',
};

const moduleIcons = {
  summarization: ScissorsLineDashed,
  blurring: EyeOff,
  compression: Minimize2,
};

const moduleColors = {
  summarization: 'from-violet-500 to-purple-600',
  blurring: 'from-blue-500 to-cyan-600',
  compression: 'from-emerald-500 to-teal-600',
};

export default function History() {
  type VideoMeta = {
    id: string;
    title?: string;
    original_filename?: string;
    video_url?: string;
    duration?: number;
    created_at?: string;
    metadata?: { json_path?: string; srt_path?: string } | null;
    thumbnail_url?: string | null;
  };
  type HistoryItem = {
    id: string;
    created_at: string;
    query?: string | null;
    status: string;
    module?: string | null;
    video_id?: string | null;
    result_url?: string | null;
    input_url?: string | null;
    video?: VideoMeta | null;
  };
  const [items, setItems] = useState<HistoryItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);
  const { user } = useAuth();
  const navigate = useNavigate();

  const fetchHistory = useCallback(async () => {
    if (!user?.id) {
      setItems([]);
      setLoading(false);
      return;
    }
    setLoading(true);
    try {
      const params = new URLSearchParams({
        user_id: user.id,
        page: String(page),
        page_size: String(pageSize),
      });
      if (search.trim()) params.set('q', search.trim());

      const apiUrl = `${API_BASE}/history?${params}`;
      const resp = await fetch(apiUrl, {
        headers: { 'ngrok-skip-browser-warning': 'true' },
      });

      let items: HistoryItem[] = [];
      let usedApi = false;

      const contentType = resp.headers.get('content-type') || '';
      let body: { items?: unknown[]; sources?: Record<string, number> } | null = null;
      if (resp.ok && contentType.includes('application/json')) {
        try {
          body = await resp.json();
        } catch {
          body = null;
        }
      }

      if (body && Array.isArray(body.items)) {
        usedApi = true;
        items = (body.items || []).map((r: {
          id: string;
          created_at: string;
          query?: string | null;
          status: string;
          module?: string | null;
          video_id?: string | null;
          result_url?: string | null;
          input_url?: string | null;
          video?: VideoMeta | null;
        }) => {
          const resultUrl = r.result_url ?? null;
          const video = r.video ?? (resultUrl
            ? { id: r.video_id || r.id, title: r.input_url || undefined, video_url: resultUrl }
            : null);
          return {
            id: r.id,
            created_at: r.created_at,
            query: r.query,
            status: effectiveHistoryStatus(
              r.status,
              video && 'status' in video ? (video as { status?: string }).status : null,
              !!(resultUrl || video?.video_url)
            ),
            module: r.module,
            video_id: r.video_id ?? null,
            result_url: resultUrl,
            input_url: r.input_url ?? null,
            video,
          };
        });
      }

      if (!usedApi) {
        // Direct DB: user_videos + processing_history (Supabase client + RLS)
        let sel = supabase
          .from('processing_history')
          .select('id,created_at,query,status,video_id,module,result_url,input_url,input_type')
          .eq('user_id', user.id)
          .order('created_at', { ascending: false })
          .range((page - 1) * pageSize, page * pageSize - 1);
        if (search.trim()) {
          sel = sel.or(`query.ilike.%${search.trim()}%,input_url.ilike.%${search.trim()}%`);
        }
        const { data: uvRows, error: uvErr } = await supabase
          .from('user_videos')
          .select('id,title,original_filename,video_url,duration,created_at,status,metadata')
          .eq('user_id', user.id)
          .order('created_at', { ascending: false })
          .limit(100);
        if (uvErr) throw uvErr;

        const { data: rows, error } = await sel;
        if (error) throw error;

        const vmeta: Record<string, VideoMeta> = {};
        const seen = new Set<string>();

        for (const r of uvRows || []) {
          vmeta[r.id] = {
            id: r.id,
            title: r.title,
            original_filename: r.original_filename,
            video_url: r.video_url,
            duration: r.duration ?? undefined,
          };
          const meta = (r.metadata as { module?: string } | null) ?? {};
          const mod =
            meta.module ||
            (r.title?.toLowerCase().includes('anonym') ? 'blurring' : 'compression');
          items.push({
            id: `uv-${r.id}`,
            created_at: r.created_at,
            query: null,
            status: effectiveHistoryStatus(null, r.status, !!r.video_url),
            module: mod,
            video_id: r.id,
            result_url: r.video_url,
            input_url: r.original_filename,
            video: vmeta[r.id],
          });
          seen.add(r.id);
        }

        for (const r of rows || []) {
          const vid = r.video_id || null;
          if (vid && seen.has(vid)) {
            const existing = items.find((i) => i.video_id === vid);
            if (existing) {
              existing.id = r.id;
              existing.query = r.query ?? existing.query;
              existing.module = r.module || existing.module;
              existing.result_url = r.result_url || existing.result_url;
              existing.input_url = r.input_url || existing.input_url;
              if (vid && vmeta[vid]) existing.video = vmeta[vid];
              const hasMedia = !!(
                existing.result_url ||
                existing.video?.video_url
              );
              existing.status = effectiveHistoryStatus(
                r.status,
                existing.status,
                hasMedia
              );
            }
            continue;
          }
          if (vid) seen.add(vid);
          const resultUrl = r.result_url;
          const videoMeta = vid
            ? vmeta[vid]
            : resultUrl
              ? { id: vid || r.id, title: r.input_url || undefined, video_url: resultUrl }
              : null;
          const uvRow = vid ? (uvRows || []).find((u) => u.id === vid) : undefined;
          items.push({
            id: r.id,
            created_at: r.created_at,
            query: r.query,
            status: effectiveHistoryStatus(
              r.status,
              uvRow?.status,
              !!(resultUrl || videoMeta?.video_url)
            ),
            module: r.module,
            video_id: vid,
            result_url: resultUrl,
            input_url: r.input_url,
            video: videoMeta,
          });
        }
        items.sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime());
        const start = (page - 1) * pageSize;
        items = items.slice(start, start + pageSize);
      }

      setItems(items);
    } catch (e) {
      console.error('Error fetching history:', e);
      setItems([]);
    } finally {
      setLoading(false);
    }
  }, [user?.id, page, pageSize, search]);

  useEffect(() => {
    fetchHistory();
  }, [fetchHistory]);



  const hasMore = useMemo(() => items.length === pageSize, [items, pageSize]);

  const onSearchChange = (v: string) => {
    setSearch(v);
    setPage(1);
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  if (loading) {
    return (
      <DashboardLayout>
        <div className="flex items-center justify-center min-h-[400px]">
          <div className="h-12 w-12 animate-spin rounded-full border-4 border-primary border-t-transparent" />
        </div>
      </DashboardLayout>
    );
  }

  return (
    <DashboardLayout>
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="max-w-6xl mx-auto space-y-4 sm:space-y-6 px-4 sm:px-6"
      >
        <div className="flex flex-col sm:flex-row sm:items-center gap-3 sm:gap-4">
          <div className="flex h-10 w-10 sm:h-12 sm:w-12 items-center justify-center rounded-xl bg-gradient-to-br from-slate-500 to-gray-600 text-white shadow-lg flex-shrink-0">
            <HistoryIcon className="h-5 w-5 sm:h-6 sm:w-6" />
          </div>
          <div className="min-w-0">
            <h1 className="text-2xl sm:text-3xl font-bold truncate">Processing History</h1>
            <p className="text-sm sm:text-base text-muted-foreground">
              View all your past video processing jobs
            </p>
          </div>
        </div>

        <div className="flex flex-col sm:flex-row sm:items-center gap-3 sm:gap-4">
          <input
            className="flex-1 px-3 py-2 rounded-md border border-border bg-background"
            placeholder="Search history..."
            value={search}
            onChange={(e) => onSearchChange(e.target.value)}
          />
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm" onClick={() => setPage(Math.max(1, page - 1))} disabled={page === 1}>Prev</Button>
            <Button variant="outline" size="sm" onClick={() => setPage(page + 1)} disabled={!hasMore}>Next</Button>
          </div>
        </div>

        {items.length === 0 ? (
          <Card className="gradient-card border-border/50">
            <CardContent className="py-12 text-center">
              <FileVideo className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
              <h3 className="text-lg font-semibold mb-2">No history yet</h3>
              <p className="text-muted-foreground">
                Your processed videos will appear here
              </p>
            </CardContent>
          </Card>
        ) : (
          <div className="grid gap-4">
            {items.map((item, index) => {
              const mod = (item.module || 'blurring') as keyof typeof moduleIcons;
              const Icon = moduleIcons[mod] || FileVideo;
              const gradient = moduleColors[mod] || moduleColors.blurring;
              const v = item.video as VideoMeta | undefined;
              const fileLabel =
                v?.original_filename || item.input_url || v?.title;
              const title = fileLabel
                ? `${moduleLabels[item.module || ''] || item.module || 'Job'} · ${fileLabel}`
                : moduleLabels[item.module || ''] || item.module || 'Processed video';
              const thumb = (v?.thumbnail_url as string | undefined) || undefined;
              const dur =
                v?.duration && v.duration > 0 ? Math.round(v.duration) : null;
              const mediaPath = resolveMediaPath(item.result_url, v?.video_url);
              const canDownload = !!mediaPath && item.status === 'completed';

              return (
                <motion.div
                  key={item.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.05 }}
                >
                  <Card className="gradient-card border-border/50 hover-lift transition-all hover:scale-[1.01]">
                    <CardHeader className="p-4 sm:p-6">
                      <div className="flex flex-col gap-3 sm:gap-4">
                        <div className="flex items-start justify-between gap-3">
                          <div className="flex items-center gap-2 sm:gap-3 min-w-0 flex-1">
                            <div className={`flex h-8 w-8 sm:h-10 sm:w-10 items-center justify-center rounded-lg bg-gradient-to-br ${gradient} text-white shadow-md flex-shrink-0`}>
                              <Icon className="h-4 w-4 sm:h-5 sm:w-5" />
                            </div>
                            <div className="min-w-0 flex-1">
                              <CardTitle className="text-base sm:text-lg">{title}</CardTitle>
                              <CardDescription className="flex items-center gap-1.5 sm:gap-2 mt-1 text-xs sm:text-sm">
                                <Calendar className="h-3 w-3 flex-shrink-0" />
                                <span className="truncate">
                                  {formatDate(item.created_at)}
                                  {dur != null ? ` · ${dur}s` : ''}
                                </span>
                              </CardDescription>
                            </div>
                          </div>
                          <div className="flex items-center gap-2 flex-shrink-0">
                            <Badge
                              variant={item.status === 'completed' ? 'default' : 'secondary'}
                              className="capitalize text-xs"
                            >
                              {item.status}
                            </Badge>
                          </div>
                        </div>                        {thumb && (
                          <img src={thumb} alt="thumbnail" className="w-full max-w-sm rounded-md border border-border" />
                        )}
                        
                        <div className="flex flex-col sm:flex-row flex-wrap gap-2">
                          {item.module === 'summarization' && (
                            <Button
                              size="sm"
                              onClick={() => {
                                const vid = v?.id || item.video_id;
                                if (vid) navigate(`/video/${vid}?q=${encodeURIComponent(item.query || '')}`);
                              }}
                              className="gradient-primary w-full sm:w-auto"
                            >
                              <Play className="h-4 w-4 mr-1" />
                              <span>Open Video</span>
                            </Button>
                          )}
                          {canDownload && mediaPath && (
                            <Button
                              size="sm"
                              variant={item.module === 'summarization' ? 'outline' : 'default'}
                              className={
                                item.module === 'summarization'
                                  ? 'w-full sm:w-auto'
                                  : 'gradient-primary w-full sm:w-auto'
                              }
                              onClick={() =>
                                downloadHistoryMedia(
                                  mediaPath,
                                  mediaFilename(mediaPath, fileLabel || 'video.mp4')
                                )
                              }
                            >
                              <Download className="h-4 w-4 mr-1" />
                              <span>Download Video</span>
                            </Button>
                          )}
                        </div>
                      </div>
                    </CardHeader>
                    <CardContent className="p-4 sm:p-6 pt-0">
                      <div className="space-y-2 text-xs sm:text-sm">
                        {item.query && (
                          <p className="text-muted-foreground line-clamp-2">
                            <span className="font-medium">Query:</span> {item.query}
                          </p>
                        )}
                        {canDownload && mediaPath && (
                          <button
                            type="button"
                            className="text-sm text-primary hover:underline text-left"
                            onClick={() =>
                              downloadHistoryMedia(
                                mediaPath,
                                mediaFilename(mediaPath, fileLabel || 'video.mp4')
                              )
                            }
                          >
                            Download processed file
                          </button>
                        )}
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>
              );
            })}
          </div>
        )}

      </motion.div>
    </DashboardLayout>
  );
}
