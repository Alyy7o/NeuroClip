// import { useState, useEffect } from 'react';
// import { motion, AnimatePresence } from 'framer-motion';
// import { ScissorsLineDashed, Play, Download, ArrowLeft, Image as ImageIcon } from 'lucide-react';
// import { DashboardLayout } from '@/components/DashboardLayout';
// import { VideoInput } from '@/components/VideoInput';
// import { VideoPreview } from '@/components/VideoPreview';
// import { UrlVideoPreview } from '@/components/UrlVideoPreview';
// import { VideoPlayer } from '@/components/VideoPlayer';
// import { ProcessingLoader } from '@/components/ProcessingLoader';
// import { Button } from '@/components/ui/button';
// import { Textarea } from '@/components/ui/textarea';
// import { Label } from '@/components/ui/label';
// import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
// import { useToast } from '@/hooks/use-toast';
// import { supabase } from '@/integrations/supabase/client';
// import { useAuth } from '@/contexts/AuthContext';
// import { z } from 'zod';

// const server_url = process.env.SERVER_URL;

// const processingSchema = z.object({
//   query: z.string()
//     .trim()
//     .min(1, { message: "Query cannot be empty" })
//     .max(1000, { message: "Query must be less than 1000 characters" }),
//   url: z.string()
//     .trim()
//     .max(2048, { message: "URL too long" })
//     .optional()
//     .refine((val) => !val || val === '' || z.string().url().safeParse(val).success, {
//       message: "Invalid URL format"
//     }),
//   fileName: z.string()
//     .max(255, { message: "File name too long" })
//     .optional()
// });

// // Component to display extracted frames
// function FramesDisplay({ framesDir }: { framesDir: string }) {
//   const [frames, setFrames] = useState<any[]>([]);
//   const [loading, setLoading] = useState(true);

//   useEffect(() => {
//     const fetchFrames = async () => {
//       try {
//         const response = await fetch(`${server_url}/${framesDir}`);
//         const data = await response.json();
//         setFrames(data.frames || []);
//       } catch (error) {
//         console.error('Error fetching frames:', error);
//       } finally {
//         setLoading(false);
//       }
//     };

//     fetchFrames();
//   }, [framesDir]);

//   if (loading) return <div className="text-center py-4">Loading frames...</div>;
  
//   if (frames.length === 0) return <div className="text-center py-4 text-muted-foreground">No frames available</div>;

//   return (
//     <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mt-4">
//       {frames.map((frame, index) => (
//         <div key={index} className="relative group">
//           <img
//             src={`${server_url}/${frame.url}`}
//             alt={`Frame ${index + 1}`}
//             className="w-full h-auto rounded-lg border border-border"
//           />
//           <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity rounded-lg flex items-center justify-center">
//             <Button variant="secondary" size="sm">
//               <Download className="h-4 w-4 mr-1" />
//               Download
//             </Button>
//           </div>
//         </div>
//       ))}
//     </div>
//   );
// }

// export default function Summarization() {
//   const [file, setFile] = useState<File | null>(null);
//   const [url, setUrl] = useState('');
//   const [query, setQuery] = useState('');
//   const [processing, setProcessing] = useState(false);
//   const [showResults, setShowResults] = useState(false);
//   const [result, setResult] = useState<any>(null);
//   const [startTime, setStartTime] = useState(0);
//   const [endTime, setEndTime] = useState(0);
//   const { toast } = useToast();
//   const { user } = useAuth();

//   const handleTimeRangeChange = (start: number, end: number) => {
//     setStartTime(start);
//     setEndTime(end);
//   };

//   const handleProcess = async () => {
//     if (!file && !url) {
//       toast({
//         title: 'No video provided',
//         description: 'Please upload a video or provide a URL',
//         variant: 'destructive',
//       });
//       return;
//     }

//     if (!query.trim()) {
//       toast({
//         title: 'No query provided',
//         description: 'Please describe what you want to extract from the video',
//         variant: 'destructive',
//       });
//       return;
//     }

//     setProcessing(true);

//     try {
//       const queryWithTime = `${query} [${startTime.toFixed(1)}s - ${endTime.toFixed(1)}s]`;
      
//       const result = processingSchema.safeParse({
//         query: queryWithTime,
//         url: url || '',
//         fileName: file?.name
//       });

//       if (!result.success) {
//         toast({
//           title: 'Validation Error',
//           description: result.error.errors[0].message,
//           variant: 'destructive'
//         });
//         setProcessing(false);
//         return;
//       }

//       const validated = result.data;

//       // Save to history with timestamp info
//       const { error } = await supabase.from('processing_history').insert({
//         user_id: user?.id,
//         module: 'summarization',
//         input_type: file ? 'file' : 'url',
//         input_url: validated.url || validated.fileName,
//         query: validated.query,
//         status: 'completed',
//       });

//       if (error) throw error;

//       // Process based on input type
//       if (url) {
//         // Process YouTube video via backend
//         const response = await fetch(`${server_url}/api/process-youtube`, {
//           method: 'POST',
//           headers: { 'Content-Type': 'application/json' },
//           body: JSON.stringify({ 
//             url: validated.url,
//             startTime,
//             endTime 
//           })
//         });
        
//         const data = await response.json();
        
//         if (!response.ok) throw new Error(data.error);
        
//         setResult({
//           jobId: data.jobId,
//           videoInfo: data.videoInfo,
//           transcript: data.transcript,
//           transcriptId: data.transcriptId,
//           framesDir: data.framesDir,
//           audioUrl: data.audioUrl,
//           summary: `Processed "${data.videoInfo.title}" from ${startTime.toFixed(1)}s to ${endTime.toFixed(1)}s`,
//           duration: endTime - startTime,
//           originalDuration: data.videoInfo.duration
//         });
//       } else {
//         // Handle regular file uploads (mock for now)
//         await new Promise(resolve => setTimeout(resolve, 5000));
        
//         setResult({
//           videoUrl: URL.createObjectURL(file as File),
//           summary: `Based on your query "${query}", here's the extracted summary:\n\nThe video segment from ${startTime.toFixed(1)}s to ${endTime.toFixed(1)}s contains the key highlights you requested. This portion captures the most relevant content, focusing on the main points that match your criteria.\n\nKey Insights:\n• The extracted segment emphasizes the core message\n• Important visual elements have been preserved\n• Audio clarity has been maintained throughout\n• The content flows naturally within the selected timeframe\n\nThe processing has successfully isolated the requested portion while maintaining video quality and coherence.`,
//           duration: endTime - startTime,
//           originalDuration: endTime,
//         });
//       }

//       setShowResults(true);

//       toast({
//         title: 'Processing complete!',
//         description: 'Your video has been summarized successfully.',
//       });
//     } catch (error: any) {
//       toast({
//         title: 'Processing failed',
//         description: error.message,
//         variant: 'destructive',
//       });
//     } finally {
//       setProcessing(false);
//     }
//   };

//   const handleReset = () => {
//     setShowResults(false);
//     setResult(null);
//     setFile(null);
//     setUrl('');
//     setQuery('');
//     setStartTime(0);
//     setEndTime(0);
//   };

//   return (
//     <DashboardLayout>
//       <AnimatePresence mode="wait">
//         {processing && <ProcessingLoader />}
//       </AnimatePresence>
//       {!showResults ? (
//         <motion.div
//           initial={{ opacity: 0, y: 20 }}
//           animate={{ opacity: 1, y: 0 }}
//           exit={{ opacity: 0, y: -20 }}
//           className="max-w-4xl mx-auto space-y-6"
//         >
//         <div className="flex items-center gap-3">
//           <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-gradient-to-br from-violet-500 to-purple-600 text-white shadow-lg">
//             <ScissorsLineDashed className="h-6 w-6" />
//           </div>
//           <div>
//             <h1 className="text-3xl font-bold">Video Summarization</h1>
//             <p className="text-muted-foreground">
//               Extract key highlights with AI-powered analysis
//             </p>
//           </div>
//         </div>

//         <Card className="gradient-card border-border/50">
//           <CardHeader>
//             <CardTitle>Input Video</CardTitle>
//             <CardDescription>Upload a video file or provide a URL</CardDescription>
//           </CardHeader>
//           <CardContent className="space-y-6">
//             {file ? (
//               <div className="space-y-4">
//                 <div className="flex items-center justify-between">
//                   <p className="text-sm text-muted-foreground">Selected: {file.name}</p>
//                   <Button 
//                     variant="outline" 
//                     size="sm"
//                     onClick={() => setFile(null)}
//                   >
//                     Change Video
//                   </Button>
//                 </div>
//                 <VideoPreview file={file} onTimeRangeChange={handleTimeRangeChange} />
//               </div>
//             ) : url ? (
//               <div className="space-y-4">
//                 <div className="flex items-center justify-between">
//                   <p className="text-sm text-muted-foreground">Video URL loaded</p>
//                   <Button 
//                     variant="outline" 
//                     size="sm"
//                     onClick={() => setUrl('')}
//                   >
//                     Change Video
//                   </Button>
//                 </div>
//                 <UrlVideoPreview url={url} onTimeRangeChange={handleTimeRangeChange} />
//               </div>
//             ) : (
//               <VideoInput
//                 onFileSelect={setFile}
//                 onUrlSubmit={setUrl}
//                 file={file}
//                 url={url}
//                 disabled={processing}
//               />
//             )}

//             <div className="space-y-2">
//               <Label htmlFor="query">What would you like to extract?</Label>
//               <Textarea
//                 id="query"
//                 placeholder="e.g., Extract the main highlights of the speech"
//                 value={query}
//                 onChange={(e) => setQuery(e.target.value)}
//                 disabled={processing}
//                 rows={4}
//                 className="resize-none"
//               />
//             </div>

//             <Button
//               onClick={handleProcess}
//               disabled={processing || (!file && !url)}
//               className="w-full gradient-primary hover:opacity-90 transition-opacity"
//               size="lg"
//             >
//               {processing ? (
//                 <>
//                   <div className="mr-2 h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
//                   Processing...
//                 </>
//               ) : (
//                 <>
//                   <Play className="mr-2 h-4 w-4" />
//                   Process Video
//                 </>
//               )}
//             </Button>
//           </CardContent>
//         </Card>
//         </motion.div>
//       ) : (
//         <motion.div
//           initial={{ opacity: 0, y: 20 }}
//           animate={{ opacity: 1, y: 0 }}
//           exit={{ opacity: 0, y: -20 }}
//           className="max-w-4xl mx-auto space-y-6"
//         >
//           <Button
//             variant="ghost"
//             onClick={handleReset}
//             className="mb-4"
//           >
//             <ArrowLeft className="mr-2 h-4 w-4" />
//             Back to Input
//           </Button>

//           <div className="flex items-center gap-3">
//             <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-gradient-to-br from-violet-500 to-purple-600 text-white shadow-lg">
//               <ScissorsLineDashed className="h-6 w-6" />
//             </div>
//             <div>
//               <h1 className="text-3xl font-bold">Processing Results</h1>
//               <p className="text-muted-foreground">
//                 Your extracted video is ready
//               </p>
//             </div>
//           </div>

//           <Card className="gradient-card border-border/50">
//             <CardHeader>
//               <CardTitle>Processed Video</CardTitle>
//               <CardDescription>
//                 {result.videoInfo 
//                   ? `${result.videoInfo.title} (${result.duration.toFixed(1)}s segment)`
//                   : `Extracted segment: ${startTime.toFixed(1)}s - ${endTime.toFixed(1)}s`
//                 }
//               </CardDescription>
//             </CardHeader>
//             <CardContent className="space-y-6">
//               <VideoPlayer 
//                 videoUrl={result.videoUrl || url} 
//                 autoPlay={false} 
//               />

//               <div className="flex gap-3">
//                 <Button className="flex-1 gradient-primary hover:opacity-90 transition-opacity">
//                   <Download className="mr-2 h-4 w-4" />
//                   Download Video
//                 </Button>
//                 <Button variant="outline" onClick={handleReset}>
//                   Process Another
//                 </Button>
//               </div>
//             </CardContent>
//           </Card>

//           {/* Transcript Section */}
//           {result.transcript && (
//             <Card className="gradient-card border-border/50">
//               <CardHeader>
//                 <CardTitle>Video Transcript</CardTitle>
//                 <CardDescription>AI-generated speech-to-text transcription</CardDescription>
//               </CardHeader>
//               <CardContent>
//                 <div className="prose prose-sm max-w-none dark:prose-invert">
//                   <p className="text-foreground whitespace-pre-wrap leading-relaxed bg-muted/30 p-4 rounded-lg">
//                     {result.transcript}
//                   </p>
//                 </div>
//               </CardContent>
//             </Card>
//           )}

//           {/* Frames Section */}
//           {result.framesDir && (
//             <Card className="gradient-card border-border/50">
//               <CardHeader>
//                 <CardTitle className="flex items-center gap-2">
//                   <ImageIcon className="h-5 w-5" />
//                   Extracted Frames
//                 </CardTitle>
//                 <CardDescription>Key visual moments from the video</CardDescription>
//               </CardHeader>
//               <CardContent>
//                 <FramesDisplay framesDir={result.framesDir} />
//               </CardContent>
//             </Card>
//           )}

//           {/* Summary Section */}
//           <Card className="gradient-card border-border/50">
//             <CardHeader>
//               <CardTitle>Video Analysis</CardTitle>
//               <CardDescription>AI-generated summary and insights</CardDescription>
//             </CardHeader>
//             <CardContent>
//               <div className="prose prose-sm max-w-none dark:prose-invert">
//                 <p className="text-foreground whitespace-pre-wrap leading-relaxed">
//                   {result.summary}
//                 </p>
//               </div>
//             </CardContent>
//           </Card>
//         </motion.div>
//       )}
//     </DashboardLayout>
//   );
// }



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
  const [result, setResult] = useState<any>(null);
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

      // Save to history with timestamp info
      const { error } = await supabase.from('processing_history').insert({
        user_id: user?.id,
        module: 'summarization',
        input_type: file ? 'file' : 'url',
        input_url: validated.url || validated.fileName,
        query: validated.query,
        status: 'completed',
      });

      if (error) throw error;

      // Simulate processing with longer delay
      await new Promise(resolve => setTimeout(resolve, 5000));

      // Mock result with processed video (keep original URL for YouTube videos)
      const processedVideoUrl = file ? URL.createObjectURL(file) : url;
      
      setResult({
        videoUrl: processedVideoUrl,
        summary: `Based on your query "${query}", here's the extracted summary:\n\nThe video segment from ${startTime.toFixed(1)}s to ${endTime.toFixed(1)}s contains the key highlights you requested. This portion captures the most relevant content, focusing on the main points that match your criteria.\n\nKey Insights:\n• The extracted segment emphasizes the core message\n• Important visual elements have been preserved\n• Audio clarity has been maintained throughout\n• The content flows naturally within the selected timeframe\n\nThe processing has successfully isolated the requested portion while maintaining video quality and coherence.`,
        duration: endTime - startTime,
        originalDuration: endTime,
      });

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
            <h1 className="text-3xl font-bold">Video Summarization</h1>
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
              <CardTitle>Processed Video</CardTitle>
              <CardDescription>
                Extracted segment: {startTime.toFixed(1)}s - {endTime.toFixed(1)}s
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <VideoPlayer videoUrl={result.videoUrl} autoPlay={false} />

              <div className="flex gap-3">
                <Button className="flex-1 gradient-primary hover:opacity-90 transition-opacity">
                  <Download className="mr-2 h-4 w-4" />
                  Download Video
                </Button>
                <Button variant="outline" onClick={handleReset}>
                  Process Another
                </Button>
              </div>
            </CardContent>
          </Card>

          <Card className="gradient-card border-border/50">
            <CardHeader>
              <CardTitle>Video Analysis</CardTitle>
              <CardDescription>AI-generated summary and insights</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="prose prose-sm max-w-none dark:prose-invert">
                <p className="text-foreground whitespace-pre-wrap leading-relaxed">
                  {result.summary}
                </p>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      )}
    </DashboardLayout>
  );
}
