import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ScissorsLineDashed, Play, Download, ArrowLeft } from 'lucide-react';
import { DashboardLayout } from '@/components/DashboardLayout';
import { VideoInput } from '@/components/VideoInput';
import { VideoPreview } from '@/components/VideoPreview';
import { UrlVideoPreview } from '@/components/UrlVideoPreview';
import { VideoPlayer } from '@/components/VideoPlayer';
import { ProcessingLoader } from '@/components/ProcessingLoader';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { useToast } from '@/hooks/use-toast';
import { supabase } from '@/integrations/supabase/client';
import { useAuth } from '@/contexts/AuthContext';
import { z } from 'zod';

// Backend API base (set VITE_API_BASE_URL in your .env/.env.local)
const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8040';

type UploadResponse = {
  message?: string;
  job_id: string;
  video_path: string;
  srt_path: string;
  json_path?: string;
};

type ClipItem = {
  rank: number;
  score: number;
  text: string;
  start: number;
  end: number;
  clip_url: string;
  llm_summary?: string;
};

type UploadAndSearchResponse = {
  results: ClipItem[];
};

type ResultData = {
  videoUrl: string | null;
  summary: string;
  duration: number;
  originalDuration: number;
  clips?: ClipItem[];
};

const processingSchema = z.object({
  query: z.string()
    .trim()
    .min(1, { message: "Query cannot be empty" })
    .max(1000, { message: "Query must be less than 1000 characters" }),
  url: z.string()
    .trim()
    .max(2048, { message: "URL too long" })
    .optional()
    .refine((val) => !val || val === '' || z.string().url().safeParse(val).success, {
      message: "Invalid URL format"
    }),
  fileName: z.string()
    .max(255, { message: "File name too long" })
    .optional()
});

export default function Summarization() {
  const [file, setFile] = useState<File | null>(null);
  const [url, setUrl] = useState('');
  const [query, setQuery] = useState('');
  const [processing, setProcessing] = useState(false);
  const [showResults, setShowResults] = useState(false);
  const [result, setResult] = useState<ResultData | null>(null);
  const [startTime, setStartTime] = useState(0);
  const [endTime, setEndTime] = useState(0);
  const { toast } = useToast();
  const { user } = useAuth();

  const handleTimeRangeChange = (start: number, end: number) => {
    setStartTime(start);
    setEndTime(end);
  };

  const handleProcess = async () => {
    if (!file && !url) {
      toast({
        title: 'No video provided',
        description: 'Please upload a video or provide a URL',
        variant: 'destructive',
      });
      return;
    }

    if (!query.trim()) {
      toast({
        title: 'No query provided',
        description: 'Please describe what you want to extract from the video',
        variant: 'destructive',
      });
      return;
    }

    setProcessing(true);

    try {
      const queryWithTime = `${query} [${startTime.toFixed(1)}s - ${endTime.toFixed(1)}s]`;

      const result = processingSchema.safeParse({
        query: queryWithTime,
        url: url || '',
        fileName: file?.name
      });

      if (!result.success) {
        toast({
          title: 'Validation Error',
          description: result.error.errors[0].message,
          variant: 'destructive'
        });
        setProcessing(false);
        return;
      }

      const validated = result.data;


      // History is recorded server-side in /upload-video using the service role.

      if (file) {
        // Upload selected file to backend for processing/transcription
        const form = new FormData();
        form.append('file', file);
        form.append('query', validated.query);
        if (user?.id) form.append('user_id', user.id);

        // Advanced params for upload-and-search
        form.append('top_k', '1');
        form.append('use_windows', 'true');
        form.append('window_size', '6');
        form.append('window_stride', '2');
        form.append('expand_neighbors', 'true');
        form.append('min_clip_secs', '20');
        form.append('max_clip_secs', '300');
        form.append('rerank', 'true');

        // We use upload-and-search now to get results directly
        const resp = await fetch(`${API_BASE}/upload-and-search`, {
          method: 'POST',
          body: form,
        });

        const data: UploadAndSearchResponse | { detail?: string } = await resp.json();
        if (!resp.ok) {
          const detail = (data as { detail?: string }).detail;
          throw new Error(detail || 'Upload failed');
        }
        const uploadData = data as UploadAndSearchResponse;

        // Skip manual clip search since upload-and-search returns results
        const clips = (uploadData.results || []).map((c: any) => ({
          ...c,
          clip_url: c.clip_url.startsWith('http') ? c.clip_url : `${API_BASE}${c.clip_url}`,
        }));

        setResult({
          videoUrl: URL.createObjectURL(file), // Local preview
          summary: `Found ${clips.length} relevant segments`,
          duration: endTime - startTime,
          originalDuration: endTime,
          clips,
        });

        setShowResults(true);
        setProcessing(false);
        return; // Exit early as we handled everything

      } else if (url) {
        // Process YouTube URL
        const resp = await fetch(`${API_BASE}/upload-via-url`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            url,
            query,
            user_id: user?.id
          }),
        });

        const data: UploadResponse | { detail?: string } = await resp.json();
        if (!resp.ok) {
          const detail = (data as { detail?: string }).detail;
          throw new Error(detail || 'Processing failed');
        }
        const uploadData = data as UploadResponse;

        // Note: Backend handles user_videos upsert if user_id is provided.

        const searchResp = await fetch(`${API_BASE}/clips/search`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            json_path: uploadData.json_path,
            query,
            use_windows: true,
            window_size: 6,
            window_stride: 2,
            expand_neighbors: true,
            min_clip_secs: 20,
            max_clip_secs: 120,
            rerank: true
          }),
        });

        const searchJson: { results: ClipItem[] } = await searchResp.json();
        if (!searchResp.ok) {
          throw new Error((searchJson as any)?.detail || 'Clip search failed');
        }

        const clips = (searchJson.results || []).map(c => ({
          ...c,
          clip_url: `${API_BASE}${c.clip_url}`,
        }));

        setResult({
          videoUrl: uploadData.video_path ? `${API_BASE}/uploads/${uploadData.video_path.split(/[\\/]/).pop()}` : url,
          summary: `Found ${clips.length} relevant segments`,
          duration: endTime - startTime,
          originalDuration: endTime,
          clips,
        });
      }

      setShowResults(true);

      toast({
        title: 'Processing complete!',
        description: 'Your video has been summarized successfully.',
      });
    } catch (error: unknown) {
      const msg = error instanceof Error ? error.message : String(error);
      toast({
        title: 'Processing failed',
        description: msg,
        variant: 'destructive',
      });
    } finally {
      setProcessing(false);
    }
  };

  const handleReset = () => {
    setShowResults(false);
    setResult(null);
    setFile(null);
    setUrl('');
    setQuery('');
    setStartTime(0);
    setEndTime(0);
  };

  return (
    <DashboardLayout>
      <AnimatePresence mode="wait">
        {processing && <ProcessingLoader />}
      </AnimatePresence>
      {!showResults ? (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -20 }}
          className="max-w-4xl mx-auto space-y-6"
        >
          <div className="flex items-center gap-3">
            <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-gradient-to-br from-violet-500 to-purple-600 text-white shadow-lg">
              <ScissorsLineDashed className="h-6 w-6" />
            </div>
            <div>
              <h1 className="text-2xl md:text-3xl font-bold">Video Summarization</h1>
              <p className="text-muted-foreground">
                Extract key highlights with AI-powered analysis
              </p>
            </div>
          </div>

          <Card className="gradient-card border-border/50">
            <CardHeader>
              <CardTitle>Input Video</CardTitle>
              <CardDescription>Upload a video file or provide a URL</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {file ? (
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <p className="text-sm text-muted-foreground">Selected: {file.name}</p>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setFile(null)}
                    >
                      Change Video
                    </Button>
                  </div>
                  <VideoPreview file={file} onTimeRangeChange={handleTimeRangeChange} />
                </div>
              ) : url ? (
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <p className="text-sm text-muted-foreground">Video URL loaded</p>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setUrl('')}
                    >
                      Change Video
                    </Button>
                  </div>
                  <UrlVideoPreview url={url} onTimeRangeChange={handleTimeRangeChange} />
                </div>
              ) : (
                <VideoInput
                  onFileSelect={setFile}
                  onUrlSubmit={setUrl}
                  file={file}
                  url={url}
                  disabled={processing}
                />
              )}

              <div className="space-y-2">
                <Label htmlFor="query">What would you like to extract?</Label>
                <Textarea
                  id="query"
                  placeholder="e.g., Extract the main highlights of the speech"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  disabled={processing}
                  rows={4}
                  className="resize-none"
                />
              </div>

              <Button
                onClick={handleProcess}
                disabled={processing || (!file && !url)}
                className="w-full gradient-primary hover:opacity-90 transition-opacity"
                size="lg"
              >
                {processing ? (
                  <>
                    <div className="mr-2 h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
                    Processing...
                  </>
                ) : (
                  <>
                    <Play className="mr-2 h-4 w-4" />
                    Process Video
                  </>
                )}
              </Button>
            </CardContent>
          </Card>
        </motion.div>
      ) : (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -20 }}
          className="max-w-4xl mx-auto space-y-6"
        >
          <Button
            variant="ghost"
            onClick={handleReset}
            className="mb-4"
          >
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Input
          </Button>

          <div className="flex items-center gap-3">
            <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-gradient-to-br from-violet-500 to-purple-600 text-white shadow-lg">
              <ScissorsLineDashed className="h-6 w-6" />
            </div>
            <div>
              <h1 className="text-3xl font-bold">Processing Results</h1>
              <p className="text-muted-foreground">
                Your extracted video is ready
              </p>
            </div>
          </div>

          <Card className="gradient-card border-border/50">
            <CardHeader>
              <CardTitle>Top Segments</CardTitle>
              <CardDescription>
                Showing top 5 clips for your query
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className={`grid gap-4 ${(result.clips || []).length === 1 ? 'grid-cols-1' : 'grid-cols-1 md:grid-cols-2'}`}>
                {(result.clips || []).slice(0, 5).map((c: ClipItem) => (
                  <div key={c.rank} className="space-y-2 border border-border p-3 rounded-lg bg-card/50">
                    <VideoPlayer videoUrl={c.clip_url} autoPlay={false} />
                    <div className="flex flex-col gap-2">
                      <div className="flex justify-between items-center text-sm text-muted-foreground">
                        <span>{c.start.toFixed(2)}s - {c.end.toFixed(2)}s</span>
                        <span className="text-xs bg-primary/10 px-2 py-0.5 rounded text-primary">Score: {c.score.toFixed(3)}</span>
                      </div>
                      {c.llm_summary && (
                        <div className="mt-1 p-3 rounded-md bg-gradient-to-br from-violet-500/5 to-purple-600/10 border border-violet-500/20">
                          <p className="text-xs font-semibold text-violet-400 mb-1 flex items-center gap-1">
                            <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20"><path d="M10 2a1 1 0 011 1v1.323l3.954 1.582 1.599-.547a1 1 0 01.894 1.79l-1.233.782 1.233.782a1 1 0 01-.894 1.79l-1.599-.547L10 15.677V17a1 1 0 11-2 0v-1.323l-3.954-1.582-1.599.547a1 1 0 01-.894-1.79l1.233-.782-1.233-.782a1 1 0 01.894-1.79l1.599.547L9 5.323V4a1 1 0 011-1z" /></svg>
                            AI Summary
                          </p>
                          <p className="text-sm leading-relaxed text-foreground/80">
                            {c.llm_summary}
                          </p>
                        </div>
                      )}
                      <Button
                        size="sm"
                        variant="outline"
                        className="w-full text-xs h-8"
                        onClick={() => {
                          const params = new URLSearchParams({
                            url: c.clip_url,
                            score: c.score.toString(),
                            start: c.start.toFixed(2),
                            end: c.end.toFixed(2)
                          });
                          window.location.href = `/download-clip?${params.toString()}`;
                        }}
                      >
                        <Download className="mr-2 h-3 w-3" />
                        Download
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
              <div className="flex gap-3">
                <Button variant="outline" onClick={handleReset} className="w-full">
                  Process Another Video
                </Button>
              </div>
            </CardContent>
          </Card>

          <Card className="gradient-card border-border/50">
            <CardHeader>
              <CardTitle>Matched Text</CardTitle>
              <CardDescription>Relevant transcript lines with context</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {(result.clips || []).slice(0, 5).map((c: ClipItem) => (
                  <div key={c.rank} className="p-4 rounded-md bg-muted/30 border border-border/50">
                    <div className="flex items-center gap-2 mb-2">
                      <div className="h-6 w-6 rounded-full bg-primary/10 text-primary flex items-center justify-center text-xs font-bold">
                        {c.rank}
                      </div>
                      <span className="text-sm font-medium text-muted-foreground">
                        {c.start.toFixed(1)}s - {c.end.toFixed(1)}s
                      </span>
                    </div>
                    <p className="text-sm leading-relaxed text-foreground/90">
                      {c.text}
                    </p>
                    {c.llm_summary && (
                      <div className="mt-3 pt-3 border-t border-border/50">
                        <p className="text-xs font-semibold text-violet-400 mb-1">AI Analysis</p>
                        <p className="text-sm leading-relaxed text-foreground/80">
                          {c.llm_summary}
                        </p>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </motion.div>
      )
      }
    </DashboardLayout >
  );
}
