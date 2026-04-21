import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ScissorsLineDashed, Play, Download, ArrowLeft, RotateCcw } from 'lucide-react';
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
import { useAuth } from '@/contexts/AuthContext';
import { z } from 'zod';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8040';

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
  topicExplanation?: string;
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
  const [currentStep, setCurrentStep] = useState(0);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [showResults, setShowResults] = useState(false);
  const [result, setResult] = useState<ResultData | null>(null);
  const [startTime, setStartTime] = useState(0);
  const [endTime, setEndTime] = useState(0);
  const { toast } = useToast();
  const { user } = useAuth();

  const summarizationSteps = [
    'Uploading video to server',
    'Extracting high-value frames (OCR)',
    'Generating audio transcript',
    'Indexing for semantic search',
    'Finalizing AI summary results'
  ];

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
    setCurrentStep(0);
    setUploadProgress(0);

    try {
      // Only include time range if the user explicitly trimmed the video (start > 0)
      const cleanQuery = startTime > 0
        ? `${query} [${startTime.toFixed(1)}s - ${endTime.toFixed(1)}s]`
        : query;
      const validated = processingSchema.parse({
        query: cleanQuery,
        url: url || '',
        fileName: file?.name
      });

      if (file) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('query', validated.query);
        if (user?.id) formData.append('user_id', user.id);
        formData.append('top_k', '10');
        formData.append('use_windows', 'true');
        formData.append('window_size', '6');
        formData.append('window_stride', '2');
        formData.append('expand_neighbors', 'true');
        formData.append('min_clip_secs', '10');
        formData.append('max_clip_secs', '300');
        formData.append('rerank', 'true');

        const xhr = new XMLHttpRequest();
        const uploadPromise = new Promise<any>((resolve, reject) => {
          xhr.upload.addEventListener('progress', (e) => {
            if (e.lengthComputable) {
              const progress = Math.round((e.loaded / e.total) * 100);
              setUploadProgress(progress);
              if (progress === 100) setCurrentStep(1);
            }
          });

          xhr.onload = () => {
            if (xhr.status >= 200 && xhr.status < 300) {
              resolve(JSON.parse(xhr.responseText));
            } else {
              reject(new Error('Job failed on server'));
            }
          };
          xhr.onerror = () => reject(new Error('Network error'));
          xhr.open('POST', `${API_BASE}/upload-and-search`);
          xhr.setRequestHeader('ngrok-skip-browser-warning', 'true');
          xhr.send(formData);
        });

        const stepInterval = setInterval(() => {
          setCurrentStep(prev => {
            if (prev >= 1 && prev < summarizationSteps.length - 1) return prev + 1;
            return prev;
          });
        }, 8000);

        const uploadData = await uploadPromise;
        clearInterval(stepInterval);
        setCurrentStep(summarizationSteps.length - 1);

        const clips = (uploadData.results || []).map((c: any) => ({
          ...c,
          clip_url: c.clip_url.startsWith('http') ? c.clip_url : `${API_BASE}${c.clip_url}`,
        }));

        setResult({
          videoUrl: URL.createObjectURL(file),
          summary: `Found ${clips.length} relevant segments`,
          duration: endTime - startTime,
          originalDuration: endTime,
          clips,
          topicExplanation: uploadData.topic_explanation,
        });
      } else if (url) {
        setCurrentStep(0);
        setUploadProgress(50); // URL processing doesn't have a "file upload" progress, so we mock it
        
        const resp = await fetch(`${API_BASE}/upload-via-url`, {
          method: 'POST',
          headers: { 
            'Content-Type': 'application/json',
            'ngrok-skip-browser-warning': 'true'
          },
          body: JSON.stringify({ url, query, user_id: user?.id }),
        });

        if (!resp.ok) throw new Error('URL processing failed');
        const uploadData = await resp.json();
        
        setCurrentStep(2); // Jump to transcription after download

        const searchResp = await fetch(`${API_BASE}/clips/search`, {
          method: 'POST',
          headers: { 
            'Content-Type': 'application/json',
            'ngrok-skip-browser-warning': 'true'
          },
          body: JSON.stringify({
            json_path: uploadData.json_path,
            query,
            top_k: 10,
            use_windows: true,
            window_size: 6,
            window_stride: 2,
            expand_neighbors: true,
            min_clip_secs: 10,
            max_clip_secs: 120,
            rerank: true
          }),
        });

        if (!searchResp.ok) throw new Error('Search failed');
        const searchJson = await searchResp.json();
        setCurrentStep(summarizationSteps.length - 1);

        const clips = (searchJson.results || []).map((c: any) => ({
          ...c,
          clip_url: `${API_BASE}${c.clip_url}`,
        }));

        setResult({
          videoUrl: uploadData.video_path ? `${API_BASE}/uploads/${uploadData.video_path.split(/[\\/]/).pop()}` : url,
          summary: `Found ${clips.length} relevant segments`,
          duration: endTime - startTime,
          originalDuration: endTime,
          clips,
          topicExplanation: searchJson.topic_explanation,
        });
      }

      setShowResults(true);
      toast({
        title: 'Processing complete!',
        description: 'Your video has been summarized successfully.',
      });
    } catch (error: any) {
      toast({
        title: 'Processing failed',
        description: error.message,
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
  };

  return (
    <DashboardLayout>
      <AnimatePresence mode="wait">
        {processing && (
          <ProcessingLoader 
            steps={summarizationSteps} 
            currentStep={currentStep} 
            uploadProgress={uploadProgress} 
          />
        )}
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
              <p className="text-muted-foreground">Extract key highlights with AI-powered analysis</p>
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
                    <Button variant="outline" size="sm" onClick={() => setFile(null)}>Change Video</Button>
                  </div>
                  <VideoPreview file={file} onTimeRangeChange={handleTimeRangeChange} />
                </div>
              ) : url ? (
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <p className="text-sm text-muted-foreground">Video URL loaded</p>
                    <Button variant="outline" size="sm" onClick={() => setUrl('')}>Change Video</Button>
                  </div>
                  <UrlVideoPreview url={url} onTimeRangeChange={handleTimeRangeChange} />
                </div>
              ) : (
                <VideoInput onFileSelect={setFile} onUrlSubmit={setUrl} file={file} url={url} disabled={processing} />
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

              <Button onClick={handleProcess} disabled={processing || (!file && !url)} className="w-full gradient-primary" size="lg">
                <Play className="mr-2 h-4 w-4" />
                Process Video
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
          <Button variant="ghost" onClick={handleReset} className="mb-4">
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Input
          </Button>

          <div className="flex items-center gap-3">
            <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-gradient-to-br from-violet-500 to-purple-600 text-white shadow-lg">
              <ScissorsLineDashed className="h-6 w-6" />
            </div>
            <div>
              <h1 className="text-3xl font-bold">Processing Results</h1>
              <p className="text-muted-foreground">Your extracted video is ready</p>
            </div>
          </div>

          {/* User Query Display */}
          <div className="p-4 rounded-xl bg-white/5 border border-white/10">
            <p className="text-xs font-medium text-muted-foreground mb-1">Your Query</p>
            <p className="text-sm text-foreground/90 font-medium italic">"{query}"</p>
          </div>

          {/* Topic Summary — shown BEFORE clips */}
          {result?.topicExplanation && (
            <Card className="border-violet-500/30 bg-gradient-to-br from-violet-500/5 to-purple-500/5">
              <CardContent className="p-6">
                <div className="flex items-start gap-3">
                  <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-violet-500/15 text-violet-400 mt-0.5">
                    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 20h9"/><path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4Z"/></svg>
                  </div>
                  <div>
                    <h3 className="text-base font-bold text-violet-300 mb-2">AI Summary</h3>
                    <p className="text-foreground/80 leading-relaxed text-sm">
                      {result.topicExplanation}
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}

          <Card className="gradient-card border-border/50">
            <CardHeader>
              <CardTitle>Relevant Segments ({result?.clips?.length || 0})</CardTitle>
              <CardDescription>All segments matching your query, ranked by relevance</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {result?.clips && result.clips.length > 0 ? (
                <div className={`grid gap-4 ${result.clips.length === 1 ? 'grid-cols-1' : 'grid-cols-1 md:grid-cols-2'}`}>
                  {result.clips.map((c) => (
                    <div key={c.rank} className="space-y-2 border border-border p-3 rounded-lg bg-card/50">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="text-xs font-bold bg-violet-500/20 text-violet-300 px-2 py-0.5 rounded">#{c.rank}</span>
                        <span className="text-xs text-muted-foreground">{c.start.toFixed(1)}s — {c.end.toFixed(1)}s</span>
                        <span className={`ml-auto text-xs px-2 py-0.5 rounded ${
                          c.score >= 0.9 ? 'bg-green-500/15 text-green-400' :
                          c.score >= 0.6 ? 'bg-yellow-500/15 text-yellow-400' :
                          'bg-orange-500/15 text-orange-400'
                        }`}>
                          {c.score >= 0.9 ? 'High' : c.score >= 0.6 ? 'Medium' : 'Low'} relevance
                        </span>
                      </div>
                      <VideoPlayer videoUrl={c.clip_url} autoPlay={false} />
                      <div className="flex flex-col gap-2">
                        {c.llm_summary && (
                          <div className="mt-1 p-3 rounded-md bg-white/5 border border-white/10">
                            <p className="text-xs font-semibold text-violet-400 mb-1">AI Analysis</p>
                            <p className="text-sm leading-relaxed text-foreground/80">{c.llm_summary}</p>
                          </div>
                        )}
                        <Button
                          size="sm"
                          variant="outline"
                          className="w-full text-xs"
                          onClick={() => {
                            const params = new URLSearchParams({ url: c.clip_url, score: c.score.toString(), start: c.start.toString(), end: c.end.toString() });
                            window.location.href = `/download-clip?${params.toString()}`;
                          }}
                        >
                          <Download className="mr-2 h-3 w-3" /> Download
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-12">
                  <div className="flex h-16 w-16 items-center justify-center rounded-full bg-yellow-500/10 mx-auto mb-4">
                    <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-yellow-400"><circle cx="12" cy="12" r="10"/><path d="M12 8v4"/><path d="M12 16h.01"/></svg>
                  </div>
                  <h3 className="text-lg font-semibold text-foreground/80 mb-2">No Relevant Segments Found</h3>
                  <p className="text-sm text-muted-foreground max-w-md mx-auto">
                    The AI couldn't find segments in this video that substantively discuss your query. 
                    Try rephrasing your query or using different keywords.
                  </p>
                </div>
              )}

              <Button variant="outline" onClick={handleReset} className="w-full mt-6">
                <RotateCcw className="mr-2 h-4 w-4" /> Process Another Video
              </Button>
            </CardContent>
          </Card>
        </motion.div>
      )}
    </DashboardLayout>
  );
}
