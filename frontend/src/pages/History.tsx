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
    job_id?: string | null;
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
    if (!user?.id) return;
    setLoading(true);
    try {
      let sel = supabase
        .from('processing_history')
        .select('id,created_at,query,status,job_id,module')
        .eq('user_id', user.id)
        .order('created_at', { ascending: false })
        .range((page - 1) * pageSize, (page * pageSize) - 1);
      if (search.trim()) {
        sel = sel.or(`query.ilike.%${search.trim()}%,job_id.eq.${search.trim()}`);
      }
      const { data: rows, error } = await sel;
      if (error) throw error;
      const vids: string[] = Array.from(new Set((rows || []).map((r: any) => r.job_id).filter(Boolean))).map(String);
      const vmeta: Record<string, VideoMeta> = {};
      if (vids.length > 0) {
        const { data: vm, error: vErr } = await supabase
          .from('video_embeddings' as any)
          .select('job_id,title,video_url,duration')
          .in('job_id', vids as string[])
          .returns<{ job_id: string; title?: string; video_url?: string; duration?: number }[]>();
        if (vErr) throw vErr;
        for (const r of vm || []) vmeta[r.job_id] = {
          id: r.job_id,
          title: r.title,
          video_url: r.video_url,
          duration: r.duration,
          original_filename: undefined,
          created_at: undefined,
          metadata: null,
          thumbnail_url: null,
        };
      }
      const items: HistoryItem[] = (rows || []).map((r: any) => ({
        id: r.id,
        created_at: r.created_at,
        query: r.query,
        status: r.status,
        module: r.module,
        job_id: r.job_id,
        video: vmeta[r.job_id] || null,
      }));

      console.log("items", items)
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
              const Icon = moduleIcons[item.module as keyof typeof moduleIcons];
              const gradient = moduleColors[item.module as keyof typeof moduleColors];
              const v = item.video as VideoMeta | undefined;
              const title = v?.original_filename || 'Video';
              const thumb = (v?.thumbnail_url as string | undefined) || undefined;
              const dur = v?.duration || 0;

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
                                <span className="truncate">{formatDate(item.created_at)} • {dur}s</span>
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
                        
                        <div className="flex flex-col sm:flex-row gap-2">
                          {item.module !== 'compression' ? (
                            <Button
                              size="sm"
                              onClick={() => {
                                const vid = v?.id || item.job_id;
                                if (vid) navigate(`/video/${vid}?q=${encodeURIComponent(item.query || '')}`);
                              }}
                              className="gradient-primary w-full sm:w-auto"
                            >
                              <Play className="h-4 w-4 mr-1" />
                              <span>Open Video</span>
                            </Button>
                          ) : (
                            <Button
                              size="sm"
                              onClick={() => {
                                if (!v?.video_url) return;
                                const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8040';
                                // If URL starts with /static, it's a relative path on the server
                                const downloadUrl = v.video_url.startsWith('http') 
                                  ? v.video_url 
                                  : `${API_BASE}${v.video_url}`;
                                
                                const link = document.createElement('a');
                                link.href = downloadUrl;
                                link.download = v.video_url.split('/').pop() || 'video.mp4';
                                link.target = '_blank';
                                document.body.appendChild(link);
                                link.click();
                                document.body.removeChild(link);
                              }}
                              className="gradient-primary w-full sm:w-auto"
                            >
                              <Download className="h-4 w-4 mr-1" />
                              <span>Download Video</span>
                            </Button>
                          )}

                          {v?.video_url && item.module !== 'compression' && (
                            <Button
                              size="sm"
                              variant="outline"
                              onClick={() => {
                                const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8040';
                                const filename = v.video_url?.split(/[/\\]/).pop();
                                if (!filename) return;
                                
                                // Guess path based on typical structure if not absolute
                                const path = v.video_url?.includes('uploads') ? `uploads/${filename}` : `clips/${filename}`;
                                const downloadUrl = `${API_BASE}/download?path=${path}`;
                                window.open(downloadUrl, '_blank');
                              }}
                              className="w-full sm:w-auto"
                            >
                              <Download className="h-4 w-4 mr-1" />
                              <span>Download</span>
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
