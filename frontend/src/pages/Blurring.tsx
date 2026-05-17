import { useState } from 'react';
import { formatTimestamp } from '@/lib/formatTime';
import { motion, AnimatePresence } from 'framer-motion';
import { EyeOff, Play, Download, RotateCcw } from 'lucide-react';
import { DashboardLayout } from '@/components/DashboardLayout';
import { VideoInput } from '@/components/VideoInput';
import { VideoPreview } from '@/components/VideoPreview';
import { UrlVideoPreview } from '@/components/UrlVideoPreview';
import { VideoPlayer } from '@/components/VideoPlayer';
import { ProcessingLoader } from '@/components/ProcessingLoader';
import { ReferenceImageUpload } from '@/components/ReferenceImageUpload';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { useToast } from '@/hooks/use-toast';
import { supabase } from '@/integrations/supabase/client';
import { useAuth } from '@/contexts/AuthContext';
import { z } from 'zod';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8040';

const processingSchema = z.object({
  query: z.string().trim().max(1000, { message: 'Query must be less than 1000 characters' }).optional(),
  url: z
    .string()
    .trim()
    .max(2048, { message: 'URL too long' })
    .optional()
    .refine((val) => !val || val === '' || z.string().url().safeParse(val).success, {
      message: 'Invalid URL format',
    }),
  fileName: z.string().max(255, { message: 'File name too long' }).optional(),
});

const blurringSteps = [
  'Uploading video & reference images',
  'Building master signature',
  'Detecting & tracking persons',
  'Applying blur & writing output',
];

export default function Blurring() {
  const [file, setFile] = useState<File | null>(null);
  const [url, setUrl] = useState('');
  const [referenceImages, setReferenceImages] = useState<File[]>([]);
  const [query, setQuery] = useState('');
  const [timeRange, setTimeRange] = useState({ start: 0, end: 0 });
  const [processing, setProcessing] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [showResults, setShowResults] = useState(false);
  const [result, setResult] = useState<{
    processedVideoUrl: string;
    explanation: string;
    duration: number;
    targetIdsBlurred: number;
    processingTimeSec: number;
    downloadPath: string;
  } | null>(null);
  const { toast } = useToast();
  const { user } = useAuth();

  const handleProcess = async () => {
    if (!file) {
      toast({
        title: 'File upload required',
        description: 'Anonymization requires uploading a video file (URL-only is not supported yet).',
        variant: 'destructive',
      });
      return;
    }

    if (referenceImages.length === 0) {
      toast({
        title: 'Reference images required',
        description: 'Add at least one photo of the person(s) to blur in the video.',
        variant: 'destructive',
      });
      return;
    }

    setProcessing(true);
    setCurrentStep(0);
    setUploadProgress(0);

    let historyId: string | null = null;

    try {
      const healthResp = await fetch(`${API_BASE}/health`, {
        headers: { 'ngrok-skip-browser-warning': 'true' },
      });
      const healthData = healthResp.ok ? await healthResp.json() : {};
      if (healthResp.ok && healthData.has_anonymize_endpoint !== true) {
        throw new Error(
          'Blur API is not available on this server (outdated backend). ' +
            'Restart your Kaggle notebook with the latest NeuroClip code, then update VITE_API_BASE_URL to the new ngrok URL.'
        );
      }
      const queryWithTime = query.trim()
        ? `${query.trim()} (Time range: ${formatTimestamp(timeRange.start)} - ${formatTimestamp(timeRange.end)})`
        : `Time range: ${formatTimestamp(timeRange.start)} - ${formatTimestamp(timeRange.end)}`;

      const validationResult = processingSchema.safeParse({
        query: queryWithTime,
        url: url || '',
        fileName: file.name,
      });

      if (!validationResult.success) {
        toast({
          title: 'Validation Error',
          description: validationResult.error.errors[0].message,
          variant: 'destructive',
        });
        setProcessing(false);
        return;
      }

      if (user?.id) {
        const { data: hist, error: histErr } = await supabase
          .from('processing_history')
          .insert({
            user_id: user.id,
            module: 'blurring',
            input_type: 'file',
            input_url: file.name,
            query: queryWithTime,
            status: 'processing',
          })
          .select('id')
          .single();

        if (histErr) console.warn('History insert failed:', histErr);
        else historyId = hist?.id ?? null;
      }

      const formData = new FormData();
      formData.append('file', file);
      referenceImages.forEach((img) => formData.append('reference_images', img));
      if (user?.id) formData.append('user_id', user.id);
      if (query.trim()) formData.append('query', queryWithTime);
      if (timeRange.start > 0) formData.append('start_sec', String(timeRange.start));
      if (timeRange.end > 0) formData.append('end_sec', String(timeRange.end));
      formData.append('match_threshold', '0.78');
      formData.append('throttle', '3');
      formData.append('grace', '30');
      formData.append('min_match_streak', '2');

      const xhr = new XMLHttpRequest();
      const uploadPromise = new Promise<Record<string, unknown>>((resolve, reject) => {
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
          } else if (xhr.status === 404) {
            reject(
              new Error(
                'POST /anonymize-video returned 404. Your Kaggle backend is missing the blur route. ' +
                  'Re-run the notebook with the latest NeuroClip repo (pip install -r requirements-blur.txt, restart uvicorn).'
              )
            );
          } else {
            try {
              const err = JSON.parse(xhr.responseText || '{}');
              reject(new Error(err.detail || 'Anonymization failed'));
            } catch {
              reject(new Error(`Anonymization failed (HTTP ${xhr.status})`));
            }
          }
        };

        xhr.onerror = () => reject(new Error('Network error during upload'));
        xhr.open('POST', `${API_BASE}/anonymize-video`);
        xhr.setRequestHeader('ngrok-skip-browser-warning', 'true');
        xhr.send(formData);
      });

      const stepInterval = setInterval(() => {
        setCurrentStep((prev) => {
          if (prev >= 1 && prev < blurringSteps.length - 1) return prev + 1;
          return prev;
        });
      }, 12000);

      const data = await uploadPromise;
      clearInterval(stepInterval);
      setCurrentStep(blurringSteps.length - 1);

      const relUrl = data.url as string;
      const downloadPath = relUrl.replace(/^\/static\//, '');
      const playbackUrl = `${API_BASE}/serve-clip?path=${encodeURIComponent(downloadPath)}`;
      const targetIds = Number(data.target_ids_blurred ?? 0);
      const procTime = Number(data.processing_time_sec ?? 0);
      const videoDurationSec = Number(data.video_duration_sec ?? 0);

      if (historyId && user?.id) {
        await supabase
          .from('processing_history')
          .update({
            status: 'completed',
            result_url: relUrl,
          })
          .eq('id', historyId);
      }

      setResult({
        processedVideoUrl: playbackUrl,
        targetIdsBlurred: targetIds,
        processingTimeSec: procTime,
        downloadPath,
        duration: videoDurationSec || (timeRange.end > timeRange.start ? timeRange.end - timeRange.start : timeRange.end),
        explanation: `Anonymized ${referenceImages.length} reference photo(s) · ${targetIds} person(s) blurred · ${procTime.toFixed(0)}s processing.${
          query.trim() ? ` Notes: ${query.trim()}` : ''
        }`,
      });

      setShowResults(true);
      toast({
        title: 'Processing complete!',
        description: 'Your anonymized video is ready.',
      });
    } catch (error: unknown) {
      const message = error instanceof Error ? error.message : 'Processing failed';
      if (historyId && user?.id) {
        await supabase
          .from('processing_history')
          .update({ status: 'failed', error_message: message })
          .eq('id', historyId);
      }
      toast({
        title: 'Processing failed',
        description: message,
        variant: 'destructive',
      });
    } finally {
      setProcessing(false);
    }
  };

  const handleReprocess = () => {
    setShowResults(false);
    setResult(null);
    setFile(null);
    setUrl('');
    setQuery('');
    setReferenceImages([]);
  };

  const handleDownload = () => {
    if (!result?.downloadPath) return;
    const a = document.createElement('a');
    a.href = `${API_BASE}/download?path=${encodeURIComponent(result.downloadPath)}`;
    a.download = 'anonymized_video.mp4';
    a.target = '_blank';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  return (
    <DashboardLayout>
      <AnimatePresence mode="wait">
        {processing ? (
          <ProcessingLoader
            key="loader"
            steps={blurringSteps}
            currentStep={currentStep}
            uploadProgress={uploadProgress}
          />
        ) : showResults ? (
          <motion.div
            key="results"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="max-w-5xl mx-auto space-y-6"
          >
            <motion.div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <motion.div className="flex h-12 w-12 items-center justify-center rounded-xl bg-gradient-to-br from-blue-500 to-cyan-600 text-white shadow-lg">
                  <EyeOff className="h-6 w-6" />
                </motion.div>
                <motion.div>
                  <h1 className="text-2xl md:text-3xl font-bold">Blurred Video</h1>
                  <p className="text-sm md:text-base text-muted-foreground">
                    {result?.targetIdsBlurred ?? 0} target(s) · {result?.processingTimeSec?.toFixed(1)}s
                  </p>
                </motion.div>
              </div>
              <Button variant="outline" onClick={handleReprocess}>
                <RotateCcw className="mr-2 h-4 w-4" />
                <span className="hidden sm:inline">New Video</span>
              </Button>
            </motion.div>

            <Card className="gradient-card border-border/50">
              <CardHeader>
                <CardTitle>Processed Video</CardTitle>
                <CardDescription>
                  Duration: {formatTimestamp(result?.duration || 0)}
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <VideoPlayer videoUrl={result?.processedVideoUrl} />

                <div className="flex flex-col sm:flex-row gap-3">
                  <Button className="flex-1 gradient-primary" size="lg" onClick={handleDownload}>
                    <Download className="mr-2 h-4 w-4" />
                    Download Blurred Video
                  </Button>
                  <Button variant="outline" size="lg" onClick={handleReprocess}>
                    <RotateCcw className="mr-2 h-4 w-4" />
                    Process Another
                  </Button>
                </div>
              </CardContent>
            </Card>

            <Card className="gradient-card border-border/50">
              <CardHeader>
                <CardTitle>Processing Summary</CardTitle>
                <CardDescription>Details about your blurred video</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="prose prose-sm max-w-none dark:prose-invert">
                  <p className="whitespace-pre-line text-sm md:text-base">{result?.explanation}</p>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        ) : (
          <motion.div
            key="input"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="max-w-4xl mx-auto space-y-6"
          >
            <div className="flex items-center gap-3">
              <motion.div className="flex h-12 w-12 items-center justify-center rounded-xl bg-gradient-to-br from-blue-500 to-cyan-600 text-white shadow-lg">
                <EyeOff className="h-6 w-6" />
              </motion.div>
              <motion.div>
                <h1 className="text-2xl md:text-3xl font-bold">Video Blurring</h1>
                <p className="text-sm md:text-base text-muted-foreground">
                  Blur faces that match your reference photos
                </p>
              </motion.div>
            </div>

            <Card className="gradient-card border-border/50">
              <CardHeader>
                <CardTitle>Input Video</CardTitle>
                <CardDescription>Upload a video file (required for anonymization)</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                {file ? (
                  <div className="space-y-4">
                    <motion.div className="flex items-center justify-between">
                      <p className="text-sm text-muted-foreground">Selected: {file.name}</p>
                      <Button variant="outline" size="sm" onClick={() => setFile(null)}>
                        Change Video
                      </Button>
                    </motion.div>
                    <VideoPreview
                      file={file}
                      onTimeRangeChange={(start, end) => setTimeRange({ start, end })}
                    />
                  </div>
                ) : url ? (
                  <div className="space-y-4">
                    <p className="text-sm text-amber-600 dark:text-amber-400">
                      URL preview only — please upload a file to run anonymization.
                    </p>
                    <UrlVideoPreview
                      url={url}
                      onTimeRangeChange={(start, end) => setTimeRange({ start, end })}
                    />
                    <Button variant="outline" size="sm" onClick={() => setUrl('')}>
                      Clear URL
                    </Button>
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

                <ReferenceImageUpload
                  images={referenceImages}
                  onChange={setReferenceImages}
                  disabled={processing}
                />

                <div className="space-y-2">
                  <Label htmlFor="query">Notes (optional)</Label>
                  <Textarea
                    id="query"
                    placeholder="e.g., Blur only the presenter on stage (saved for history; matching uses reference photos)"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    disabled={processing}
                    rows={3}
                    className="resize-none"
                  />
                </div>

                <Button
                  onClick={handleProcess}
                  disabled={processing || !file || referenceImages.length === 0}
                  className="w-full gradient-primary hover:opacity-90 transition-opacity"
                  size="lg"
                >
                  <Play className="mr-2 h-4 w-4" />
                  Process Video
                </Button>
              </CardContent>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>
    </DashboardLayout>
  );
}
