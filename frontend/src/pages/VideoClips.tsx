import { Download } from 'lucide-react'
import { formatTimestamp } from '@/lib/formatTime'
import { useEffect, useState } from 'react'
import { useParams, useSearchParams } from 'react-router-dom'
import { DashboardLayout } from '@/components/DashboardLayout'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Label } from '@/components/ui/label'
import { VideoPlayer } from '@/components/VideoPlayer'
import { useToast } from '@/hooks/use-toast'
import { supabase } from '@/integrations/supabase/client'

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8040'

export default function VideoClips() {
  const { id } = useParams()
  const [sp] = useSearchParams()
  type VideoMeta = { id?: string; job_id?: string; title?: string; video_url?: string; created_at?: string; duration?: number; original_filename?: string };
  type HistoryRow = { id: string; created_at: string; query?: string | null; status: string };
  type ClipRow = { rank: number; score: number; text: string; start: number; end: number; clip_url: string | null };
  type VideoEmbRow = { job_id?: string; video_embedding?: number[] | null; transcript_embedding?: number[] | null };
  const [video, setVideo] = useState<VideoMeta | null>(null)
  const [query, setQuery] = useState('')
  const [clips, setClips] = useState<ClipRow[]>([])
  const [history, setHistory] = useState<HistoryRow[]>([])
  const [histPage, setHistPage] = useState(1)
  const [histPageSize, setHistPageSize] = useState(10)
  const [ve, setVe] = useState<VideoEmbRow | null>(null)
  const [loading, setLoading] = useState(true)
  const [searching, setSearching] = useState(false)
  const [useWindows, setUseWindows] = useState(true)
  const [windowSize, setWindowSize] = useState(6)
  const [windowStride, setWindowStride] = useState(2)
  const [expandNeighbors, setExpandNeighbors] = useState(true)
  const [minClipSecs, setMinClipSecs] = useState(20)
  const [maxClipSecs, setMaxClipSecs] = useState(120)
  const [useRerank, setUseRerank] = useState(true)
  const { toast } = useToast()

  useEffect(() => {
    const load = async () => {
      try {
        const { data: vids, error: vErr } = await supabase
          .from('video_embeddings' as any)
          .select('job_id,title,video_url,duration')
          .eq('job_id', String(id))
          .limit(1)
          .returns<{ job_id: string; title?: string; video_url?: string; duration?: number }[]>()
        if (vErr) throw vErr
        const row = (vids || [])[0]
        setVideo(row ? { id: row.job_id, job_id: row.job_id, title: row.title, video_url: row.video_url, duration: row.duration } : null)
        const start = (histPage - 1) * histPageSize
        const end = start + histPageSize - 1
        const { data: hist, error: hErr } = await supabase
          .from('processing_history')
          .select('id,created_at,query,status')
          .eq('video_id', id)
          .order('created_at', { ascending: false })
          .range(start, end)
        if (hErr) throw hErr
        setHistory(hist || [])
        const eresp = await fetch(`${API_BASE}/video-embeddings?job_id=${id}`, {
          headers: { 'ngrok-skip-browser-warning': 'true' }
        })
        if (eresp.ok) {
          const ej = await eresp.json()
          setVe(ej)
        } else {
          setVe(null)
        }
      } catch (e: unknown) {
        const msg = e instanceof Error ? e.message : String(e)
        toast({ title: 'Load failed', description: msg, variant: 'destructive' })
      } finally {
        setLoading(false)
      }
    }
    load()
    const q = sp.get('q') || ''
    if (q.trim()) {
      setQuery(q)
      runSearch(q)
    }
  }, [id, histPage, histPageSize])

  const runSearch = async (q: string) => {
    if (!id || !q.trim()) {
      toast({ title: 'Enter a query', description: 'Please type what to find in the video.' })
      return
    }
    setSearching(true)
    try {
      setClips([])
      let resp = await fetch(`${API_BASE}/clips/search-db`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'ngrok-skip-browser-warning': 'true'
        },
        body: JSON.stringify({ job_id: id, query: q, top_k: 5, margin_secs: 0.5, use_windows: useWindows, window_size: windowSize, window_stride: windowStride, expand_neighbors: expandNeighbors, min_clip_secs: minClipSecs, max_clip_secs: maxClipSecs, rerank: useRerank }),
      })
      if (!resp.ok) {
        try { const err = await resp.json(); console.warn('search-db failed:', err) } catch { }
        resp = await fetch(`${API_BASE}/clips/search`, {
          method: 'POST',
          headers: { 
            'Content-Type': 'application/json',
            'ngrok-skip-browser-warning': 'true'
          },
          body: JSON.stringify({ job_id: id, query: q, top_k: 5, margin_secs: 0.5 }),
        })
      }
      const json = await resp.json()
      if (!resp.ok) {
        toast({ title: 'Search failed', description: json.detail || 'Error' })
        setClips([])
        return
      }
      const mapped = (json.results || []).map((c: { clip_url?: string; start: number; end: number; score: number; text: string }) => {
        const base = c.clip_url ? `${API_BASE}${c.clip_url}` : null
        return { ...c, clip_url: base ? `${base}?ts=${Date.now()}` : null }
      })
      console.log('clips mapped', mapped.length, mapped)
      setClips(mapped)
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e)
      toast({ title: 'Search failed', description: msg })
      setClips([])
    } finally {
      setSearching(false)
    }
  }

  if (loading) {
    return (
      <DashboardLayout>
        <div className="flex items-center justify-center min-h-[300px]">
          <div className="h-12 w-12 animate-spin rounded-full border-4 border-primary border-t-transparent" />
        </div>
      </DashboardLayout>
    )
  }

  return (
    <DashboardLayout>
      <div className="max-w-6xl mx-auto space-y-6 px-4">
        <Card className="gradient-card border-border/50">
          <CardHeader>
            <CardTitle>{video?.title || video?.original_filename || 'Video'}</CardTitle>
            <CardDescription>Search clips using saved embeddings</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="clip-query">Query</Label>
              <div className="flex gap-2">
                <input id="clip-query" className="flex-1 px-3 py-2 rounded-md border border-border bg-background" placeholder="e.g., password strength advice" value={query} onChange={(e) => setQuery(e.target.value)} onKeyDown={(e) => { if (e.key === 'Enter') runSearch(query) }} />
                <Button onClick={() => runSearch(query)} disabled={searching || !query.trim()}>{searching ? 'Searching…' : 'Search Clips'}</Button>
              </div>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
              <div className="space-y-1">
                <Label>Context</Label>
                <div className="flex items-center gap-2">
                  <label className="text-xs"><input type="checkbox" checked={useWindows} onChange={(e) => setUseWindows(e.target.checked)} /> Windowed</label>
                  <label className="text-xs"><input type="checkbox" checked={expandNeighbors} onChange={(e) => setExpandNeighbors(e.target.checked)} /> Merge neighbors</label>
                  <label className="text-xs"><input type="checkbox" checked={useRerank} onChange={(e) => setUseRerank(e.target.checked)} /> Rerank</label>
                </div>
              </div>
              <div className="space-y-1">
                <Label>Window size/stride</Label>
                <div className="flex items-center gap-2">
                  <input type="number" min={1} className="w-16 px-2 py-1 rounded-md border border-border bg-background" value={windowSize} onChange={(e) => setWindowSize(Number(e.target.value) || 6)} />
                  <input type="number" min={1} className="w-16 px-2 py-1 rounded-md border border-border bg-background" value={windowStride} onChange={(e) => setWindowStride(Number(e.target.value) || 2)} />
                </div>
              </div>
              <div className="space-y-1">
                <Label>Min/Max seconds</Label>
                <div className="flex items-center gap-2">
                  <input type="number" min={1} className="w-16 px-2 py-1 rounded-md border border-border bg-background" value={minClipSecs} onChange={(e) => setMinClipSecs(Number(e.target.value) || 20)} />
                  <input type="number" min={1} className="w-16 px-2 py-1 rounded-md border border-border bg-background" value={maxClipSecs} onChange={(e) => setMaxClipSecs(Number(e.target.value) || 120)} />
                </div>
              </div>
            </div>
            {clips.length > 0 && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {clips.map((c, i) => (
                  <div key={c.clip_url || `${c.rank}-${i}`} className="space-y-2">
                    {c.clip_url ? <VideoPlayer key={c.clip_url} videoUrl={c.clip_url} autoPlay={false} /> : <div className="text-sm text-muted-foreground">Clip not available</div>}
                    <div className="text-sm text-muted-foreground">{formatTimestamp(c.start)} - {formatTimestamp(c.end)} • score {c.score.toFixed(3)}</div>
                    <div className="text-sm">{c.text}</div>
                    {c.clip_url && (
                      <Button variant="outline" size="sm" className="w-full mt-1" onClick={() => {
                        const clipUrlStr = c.clip_url || '';
                        let path = '';
                        try {
                          const urlObj = new URL(clipUrlStr, window.location.origin);
                          const servePath = urlObj.searchParams.get('path');
                          if (servePath) {
                            path = servePath;
                          } else if (urlObj.pathname.includes('/static/')) {
                            path = urlObj.pathname.split('/static/')[1];
                          }
                        } catch {
                          const match = clipUrlStr.match(/[?&]path=([^&]+)/);
                          if (match) path = decodeURIComponent(match[1]);
                        }
                        if (!path) return;

                        fetch(`${API_BASE}/serve-clip?path=${encodeURIComponent(path)}`, {
                          headers: { 'ngrok-skip-browser-warning': 'true' },
                        })
                        .then(r => r.blob())
                        .then(blob => {
                          const url = URL.createObjectURL(blob);
                          const link = document.createElement('a');
                          link.href = url;
                          link.download = path.split('/').pop() || 'clip.mp4';
                          document.body.appendChild(link);
                          link.click();
                          document.body.removeChild(link);
                          URL.revokeObjectURL(url);
                        })
                        .catch(() => {
                          window.open(`${API_BASE}/serve-clip?path=${encodeURIComponent(path)}`, '_blank');
                        });
                      }}>
                        <Download className="h-4 w-4 mr-2" />
                        Download
                      </Button>
                    )}
                  </div>
                ))}
              </div>
            )}
            {clips.length === 0 && !searching && (
              <div className="text-sm text-muted-foreground">No clips found for the current query.</div>
            )}
          </CardContent>
        </Card>

        <Card className="gradient-card border-border/50">
          <CardHeader>
            <CardTitle>Past Queries</CardTitle>
            <CardDescription>Click to re-run</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2">
            <div className="flex items-center gap-2 justify-end pb-2">
              <Button variant="outline" size="sm" onClick={() => setHistPage(Math.max(1, histPage - 1))} disabled={histPage === 1}>Prev</Button>
              <Button variant="outline" size="sm" onClick={() => setHistPage(histPage + 1)} disabled={(history || []).length < histPageSize}>Next</Button>
            </div>
            {(history || []).map((h: HistoryRow) => (
              <div key={h.id} className="flex items-center justify-between p-2 rounded-md border border-border">
                <div className="text-sm">{new Date(h.created_at).toLocaleString()} • {h.query}</div>
                <Button size="sm" onClick={() => runSearch(h.query || '')}>Search</Button>
              </div>
            ))}
          </CardContent>
        </Card>

        {/* {ve && (
          <Card className="gradient-card border-border/50">
            <CardHeader>
              <CardTitle>Embeddings</CardTitle>
              <CardDescription>Loaded from database via video_id</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="text-sm text-muted-foreground">video_id: { ve.job_id}</div>
              <div className="text-sm">video_embedding: {Array.isArray(ve.video_embedding) ? ve.video_embedding.length : 0} dims</div>
              <div className="text-sm">transcript_embedding: {Array.isArray(ve.transcript_embedding) ? ve.transcript_embedding.length : 0} dims</div>
            </CardContent>
          </Card>
        )} */}
      </div>
    </DashboardLayout>
  )
}
