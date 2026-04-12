import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Minimize2, Play, Download, RotateCcw } from 'lucide-react';
import { DashboardLayout } from '@/components/DashboardLayout';
import { VideoInput } from '@/components/VideoInput';
import { VideoPreview } from '@/components/VideoPreview';
import { ProcessingLoader } from '@/components/ProcessingLoader';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { useToast } from '@/hooks/use-toast';
import { useAuth } from '@/contexts/AuthContext';
import { z } from 'zod';

const processingSchema = z.object({
  fileName: z.string()
    .max(255, { message: "File name too long" })
    .optional()
});

export default function Compression() {
  const [file, setFile] = useState<File | null>(null);
  const [timeRange, setTimeRange] = useState({ start: 0, end: 0 });
  const [processing, setProcessing] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [showResults, setShowResults] = useState(false);
  const [result, setResult] = useState<any>(null);
  const { toast } = useToast();
  const { user } = useAuth();

  const compressionSteps = [
    'Uploading video to server',
    'Scanning for acceleration',
    'Compressing video',
    'Finalizing and saving results'
  ];

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(2) + ' KB';
    return (bytes / 1024 / 1024).toFixed(2) + ' MB';
  };

  const handleProcess = async () => {
    if (!file) {
      toast({
        title: 'No video provided',
        description: 'Please upload a video to compress',
        variant: 'destructive',
      });
      return;
    }

    setProcessing(true);
    setCurrentStep(0);
    setUploadProgress(0);

    try {
      // Prepare form data
      const formData = new FormData();
      if (file) {
        formData.append('file', file);
      }
      if (user?.id) {
        formData.append('user_id', user.id);
      }

      const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8040';
      
      // Use XMLHttpRequest for upload progress
      const xhr = new XMLHttpRequest();
      
      const uploadPromise = new Promise((resolve, reject) => {
        xhr.upload.addEventListener('progress', (event) => {
          if (event.lengthComputable) {
            const progress = Math.round((event.loaded / event.total) * 100);
            setUploadProgress(progress);
            if (progress === 100) {
              setCurrentStep(1); // Move to "Scanning for GPU"
            }
          }
        });

        xhr.onload = () => {
          if (xhr.status >= 200 && xhr.status < 300) {
            resolve(JSON.parse(xhr.responseText));
          } else {
            try {
              const err = JSON.parse(xhr.responseText || '{}');
              reject(new Error(err.detail || 'Compression failed'));
            } catch {
              reject(new Error('Compression failed'));
            }
          }
        };

        xhr.onerror = () => reject(new Error('Network error during upload'));
        
        xhr.open('POST', `${API_BASE}/compress-video`);
        xhr.send(formData);
      });

      // Simulation of step transitions while backend works
      const stepInterval = setInterval(() => {
        setCurrentStep(prev => {
          if (prev >= 1 && prev < compressionSteps.length - 1) {
            return prev + 1;
          }
          return prev;
        });
      }, 5000);

      const data: any = await uploadPromise;
      clearInterval(stepInterval);
      setCurrentStep(compressionSteps.length - 1); // Finalizing

      const originalSize = data.original_size;
      const compressedSize = data.compressed_size;
      const reduction = data.reduction;
      const durationSeconds = data.duration_seconds;
      const processedVideoUrl = `${API_BASE}${data.url}`;
      const usedOriginal = data.used_original || false;

      let explanation = '';
      if (usedOriginal) {
        explanation = `Your video is already highly optimized and cannot be compressed further without significant quality loss.

The original file has been preserved at its current quality.
${data.duration_seconds ? `\nAnalysis took: ${data.duration_seconds} seconds` : ''}

- File size: ${formatFileSize(originalSize)}
- Encoder: ${data.encoder}
- Result: Original preserved (already optimal)`;
      } else {
        explanation = `Your video has been successfully compressed using advanced adaptive encoding (${data.encoder}).
${data.duration_seconds ? `\nCompression took: ${data.duration_seconds} seconds` : ''}

Compression Statistics:

- Original size: ${formatFileSize(originalSize)}
- Compressed size: ${formatFileSize(compressedSize)}
- Size reduction: ${reduction.toFixed(2)}%`;
      }

      setResult({
        processedVideoUrl: processedVideoUrl,
        originalSize,
        compressedSize,
        reduction: reduction,
        durationSeconds: durationSeconds,
        usedOriginal,
        explanation,
        duration: timeRange.end - timeRange.start,
        originalDuration: timeRange.end,
      });

      setShowResults(true);

      toast({
        title: 'Compression complete!',
        description: 'Your video has been compressed successfully.',
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
  };

  return (
    <DashboardLayout>
      <AnimatePresence mode="wait">
        {processing ? (
          <ProcessingLoader 
            key="loader" 
            steps={compressionSteps} 
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
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-gradient-to-br from-emerald-500 to-teal-600 text-white shadow-lg">
                  <Minimize2 className="h-6 w-6" />
                </div>
                <div>
                  <h1 className="text-2xl md:text-3xl font-bold">
                    {result?.usedOriginal ? 'Video Already Optimized' : 'Compressed Video'}
                  </h1>
                  <p className="text-sm md:text-base text-muted-foreground">
                    {result?.usedOriginal
                      ? 'Original preserved — already at optimal compression'
                      : `${result?.reduction?.toFixed(2)}% size reduction`}
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
                <CardTitle>Compression Statistics</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                  <Card className="bg-muted/30">
                    <CardContent className="pt-6 text-center">
                      <p className="text-xs sm:text-sm text-muted-foreground mb-1">Original Size</p>
                      <p className="text-xl sm:text-2xl font-bold">{formatFileSize(result.originalSize)}</p>
                    </CardContent>
                  </Card>
                  <Card className="bg-primary/10 border-primary/20">
                    <CardContent className="pt-6 text-center">
                      <p className="text-xs sm:text-sm text-muted-foreground mb-1">Compressed Size</p>
                      <p className="text-xl sm:text-2xl font-bold text-primary">{formatFileSize(result.compressedSize)}</p>
                    </CardContent>
                  </Card>
                  <Card className={result?.usedOriginal ? 'bg-yellow-500/10 border-yellow-500/20' : result?.reduction > 0 ? 'bg-accent/10 border-accent/20' : 'bg-red-500/10 border-red-500/20'}>
                    <CardContent className="pt-6 text-center">
                      <p className="text-xs sm:text-sm text-muted-foreground mb-1">Size Reduction</p>
                      <p className={`text-xl sm:text-2xl font-bold ${result?.usedOriginal ? 'text-yellow-400' : result?.reduction > 0 ? 'text-accent' : 'text-red-400'}`}>
                        {result?.usedOriginal ? 'N/A' : `${result.reduction?.toFixed(2)}%`}
                      </p>
                    </CardContent>
                  </Card>
                </div>
              </CardContent>
            </Card>

            <Card className="gradient-card border-border/50">
              <CardHeader>
                <CardTitle>Compressed Video Result</CardTitle>
                <CardDescription className="flex flex-col gap-1">
                  <span>Video length: {result?.duration?.toFixed(1)}s</span>
                  {result?.durationSeconds && (
                    <span className="text-primary font-medium">
                      Compression time: {result.durationSeconds.toFixed(1)}s
                    </span>
                  )}
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="flex flex-col sm:flex-row gap-3">
                  <Button className="flex-1 gradient-primary" size="lg" onClick={() => {
                    const a = document.createElement('a');
                    a.href = result?.processedVideoUrl;
                    a.download = 'compressed_video.mp4';
                    a.target = '_blank';
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                  }}>
                    <Download className="mr-2 h-4 w-4" />
                    Download Compressed Video
                  </Button>
                  <Button variant="outline" size="lg" onClick={handleReprocess}>
                    <RotateCcw className="mr-2 h-4 w-4" />
                    Compress Another
                  </Button>
                </div>
              </CardContent>
            </Card>

            <Card className="gradient-card border-border/50">
              <CardHeader>
                <CardTitle>Compression Details</CardTitle>
                <CardDescription>Technical information about the compression</CardDescription>
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
              <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-gradient-to-br from-emerald-500 to-teal-600 text-white shadow-lg">
                <Minimize2 className="h-6 w-6" />
              </div>
              <div>
                <h1 className="text-2xl md:text-3xl font-bold">Video Compression</h1>
                <p className="text-sm md:text-base text-muted-foreground">
                  Reduce file size while maintaining quality
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
                ) : (
                  <VideoInput
                    onFileSelect={setFile}
                    onUrlSubmit={() => {}}
                    file={file}
                    hideUrl={true}
                    disabled={processing}
                  />
                )}

                <Button
                  onClick={handleProcess}
                  disabled={processing || !file}
                  className="w-full gradient-primary hover:opacity-90 transition-opacity"
                  size="lg"
                >
                  <Play className="mr-2 h-4 w-4" />
                  Compress Video
                </Button>
              </CardContent>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>
    </DashboardLayout>
  );
}
