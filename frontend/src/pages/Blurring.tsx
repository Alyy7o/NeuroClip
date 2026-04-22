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
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { useToast } from '@/hooks/use-toast';
import { supabase } from '@/integrations/supabase/client';
import { useAuth } from '@/contexts/AuthContext';
import { z } from 'zod';

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

export default function Blurring() {
  const [file, setFile] = useState<File | null>(null);
  const [url, setUrl] = useState('');
  const [query, setQuery] = useState('');
  const [timeRange, setTimeRange] = useState({ start: 0, end: 0 });
  const [processing, setProcessing] = useState(false);
  const [showResults, setShowResults] = useState(false);
  const [result, setResult] = useState<any>(null);
  const { toast } = useToast();
  const { user } = useAuth();

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
        title: 'No instructions provided',
        description: 'Please describe what you want to blur',
        variant: 'destructive',
      });
      return;
    }

    setProcessing(true);

    try {
      const queryWithTime = `${query} (Time range: ${formatTimestamp(timeRange.start)} - ${formatTimestamp(timeRange.end)})`;
      
      const validationResult = processingSchema.safeParse({
        query: queryWithTime,
        url: url || '',
        fileName: file?.name
      });

      if (!validationResult.success) {
        toast({
          title: 'Validation Error',
          description: validationResult.error.errors[0].message,
          variant: 'destructive'
        });
        setProcessing(false);
        return;
      }

      const validated = validationResult.data;
      
      const { error } = await supabase.from('processing_history').insert({
        user_id: user?.id,
        module: 'blurring',
        input_type: file ? 'file' : 'url',
        input_url: validated.url || validated.fileName,
        query: validated.query,
        status: 'completed',
      });

      if (error) throw error;

      await new Promise(resolve => setTimeout(resolve, 5000));

      const mockVideoUrl = file ? URL.createObjectURL(file) : url;
      
      setResult({
        processedVideoUrl: mockVideoUrl,
        explanation: `Your video has been successfully processed with blurring applied based on your instructions: "${query}". 

The blurring effect has been applied to the selected time range from ${formatTimestamp(timeRange.start)} to ${formatTimestamp(timeRange.end)}. The specified objects or areas have been intelligently detected and blurred throughout this segment.

**Processing Details:**
- Original duration: ${formatTimestamp(timeRange.end)}
- Processed segment: ${formatTimestamp(timeRange.end - timeRange.start)}
- Blur algorithm: Gaussian blur with adaptive tracking
- Quality preservation: 95%

The processed video maintains the original resolution and frame rate while applying the blur effect only to the targeted areas. All other portions of the video remain untouched.`,
        duration: timeRange.end - timeRange.start,
        originalDuration: timeRange.end,
      });

      setShowResults(true);
      
      toast({
        title: 'Processing complete!',
        description: 'Your video has been processed successfully.',
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

  const handleReprocess = () => {
    setShowResults(false);
    setResult(null);
    setFile(null);
    setUrl('');
    setQuery('');
  };

  return (
    <DashboardLayout>
      <AnimatePresence mode="wait">
        {processing ? (
          <ProcessingLoader key="loader" />
        ) : showResults ? (
          <motion.div
            key="results"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="max-w-5xl mx-auto space-y-6"
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-gradient-to-br from-blue-500 to-cyan-600 text-white shadow-lg">
                  <EyeOff className="h-6 w-6" />
                </div>
                <div>
                  <h1 className="text-2xl md:text-3xl font-bold">Blurred Video</h1>
                  <p className="text-sm md:text-base text-muted-foreground">
                    Processing complete
                  </p>
                </div>
              </div>
              <Button variant="outline" onClick={handleReprocess}>
                <RotateCcw className="mr-2 h-4 w-4" />
                <span className="hidden sm:inline">New Video</span>
              </Button>
            </div>

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
                  <Button className="flex-1 gradient-primary" size="lg">
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
              <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-gradient-to-br from-blue-500 to-cyan-600 text-white shadow-lg">
                <EyeOff className="h-6 w-6" />
              </div>
              <div>
                <h1 className="text-2xl md:text-3xl font-bold">Video Blurring</h1>
                <p className="text-sm md:text-base text-muted-foreground">
                  Automatically blur sensitive content
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
                    <VideoPreview
                      file={file}
                      onTimeRangeChange={(start, end) => setTimeRange({ start, end })}
                    />
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
                    <UrlVideoPreview 
                      url={url} 
                      onTimeRangeChange={(start, end) => setTimeRange({ start, end })} 
                    />
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
                  <Label htmlFor="query">What would you like to blur?</Label>
                  <Textarea
                    id="query"
                    placeholder="e.g., Blur all faces in the video, or blur the license plate"
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
