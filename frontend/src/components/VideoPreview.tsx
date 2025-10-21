import { useEffect, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import { Scissors, Play, Pause } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Label } from '@/components/ui/label';

interface VideoPreviewProps {
  file: File;
  onTimeRangeChange: (startTime: number, endTime: number) => void;
}

export function VideoPreview({ file, onTimeRangeChange }: VideoPreviewProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const timelineRef = useRef<HTMLDivElement>(null);
  const [duration, setDuration] = useState(0);
  const [startTime, setStartTime] = useState(0);
  const [endTime, setEndTime] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [videoUrl, setVideoUrl] = useState<string>('');
  const [isDraggingStart, setIsDraggingStart] = useState(false);
  const [isDraggingEnd, setIsDraggingEnd] = useState(false);
  const [thumbnails, setThumbnails] = useState<string[]>([]);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const url = URL.createObjectURL(file);
    setVideoUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  useEffect(() => {
    onTimeRangeChange(startTime, endTime);
  }, [startTime, endTime, onTimeRangeChange]);

  const handleLoadedMetadata = () => {
    if (videoRef.current) {
      const dur = videoRef.current.duration;
      setDuration(dur);
      setEndTime(dur);
      generateThumbnails(dur);
    }
  };

  const generateThumbnails = async (duration: number) => {
    if (!videoRef.current || !canvasRef.current) return;
    
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const thumbnailCount = 10;
    const interval = duration / thumbnailCount;
    const thumbs: string[] = [];

    canvas.width = 160;
    canvas.height = 90;

    for (let i = 0; i < thumbnailCount; i++) {
      const time = i * interval;
      video.currentTime = time;
      
      await new Promise((resolve) => {
        video.onseeked = () => {
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          thumbs.push(canvas.toDataURL('image/jpeg', 0.7));
          resolve(null);
        };
      });
    }

    setThumbnails(thumbs);
    video.currentTime = 0;
  };

  const handleTimeUpdate = () => {
    if (videoRef.current) {
      const time = videoRef.current.currentTime;
      setCurrentTime(time);
      
      // Stop at end marker
      if (time >= endTime && isPlaying) {
        videoRef.current.currentTime = startTime;
        setCurrentTime(startTime);
      }
    }
  };

  const togglePlayPause = () => {
    if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.pause();
      } else {
        // Start from the selected start time
        if (videoRef.current.currentTime < startTime || videoRef.current.currentTime >= endTime) {
          videoRef.current.currentTime = startTime;
        }
        videoRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  const handleTimelineClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!timelineRef.current || duration === 0) return;
    
    const rect = timelineRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const percentage = Math.max(0, Math.min(1, x / rect.width));
    const time = percentage * duration;
    
    if (videoRef.current) {
      videoRef.current.currentTime = time;
      setCurrentTime(time);
    }
  };

  const handleStartDrag = (e: React.MouseEvent | React.TouchEvent) => {
    e.stopPropagation();
    setIsDraggingStart(true);
  };

  const handleEndDrag = (e: React.MouseEvent | React.TouchEvent) => {
    e.stopPropagation();
    setIsDraggingEnd(true);
  };

  useEffect(() => {
    const handleMove = (e: MouseEvent | TouchEvent) => {
      if (!timelineRef.current || duration === 0) return;
      
      if (isDraggingStart || isDraggingEnd) {
        const rect = timelineRef.current.getBoundingClientRect();
        const clientX = 'touches' in e ? e.touches[0].clientX : e.clientX;
        const x = clientX - rect.left;
        const percentage = Math.max(0, Math.min(1, x / rect.width));
        const time = percentage * duration;
        
        if (isDraggingStart) {
          setStartTime(Math.min(time, endTime - 0.5));
        } else if (isDraggingEnd) {
          setEndTime(Math.max(time, startTime + 0.5));
        }
      }
    };

    const handleEnd = () => {
      setIsDraggingStart(false);
      setIsDraggingEnd(false);
    };

    if (isDraggingStart || isDraggingEnd) {
      document.addEventListener('mousemove', handleMove);
      document.addEventListener('mouseup', handleEnd);
      document.addEventListener('touchmove', handleMove, { passive: false });
      document.addEventListener('touchend', handleEnd);
    }

    return () => {
      document.removeEventListener('mousemove', handleMove);
      document.removeEventListener('mouseup', handleEnd);
      document.removeEventListener('touchmove', handleMove);
      document.removeEventListener('touchend', handleEnd);
    };
  }, [isDraggingStart, isDraggingEnd, duration, startTime, endTime]);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const selectedDuration = endTime - startTime;

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className="space-y-4"
    >
      <Card className="p-4 space-y-4">
        <div className="aspect-video bg-black rounded-lg overflow-hidden relative">
          <video
            ref={videoRef}
            src={videoUrl}
            onLoadedMetadata={handleLoadedMetadata}
            onTimeUpdate={handleTimeUpdate}
            className="w-full h-full object-contain"
          />
          <Button
            onClick={togglePlayPause}
            variant="secondary"
            size="icon"
            className="absolute bottom-4 right-4 rounded-full"
          >
            {isPlaying ? <Pause className="h-5 w-5" /> : <Play className="h-5 w-5" />}
          </Button>
        </div>

        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <Label className="flex items-center gap-2">
              <Scissors className="h-4 w-4" />
              Select Video Timeline
            </Label>
            <span className="text-sm text-muted-foreground">
              Duration: {formatTime(selectedDuration)}
            </span>
          </div>

          <div className="space-y-2">
            {/* Timeline with thumbnails */}
            <div
              ref={timelineRef}
              onClick={handleTimelineClick}
              className="relative h-16 bg-muted rounded-lg cursor-pointer overflow-hidden"
            >
              {/* Video thumbnails */}
              <div className="absolute inset-0 flex">
                {thumbnails.map((thumb, index) => (
                  <div
                    key={index}
                    className="flex-1 h-full border-r border-border/20 last:border-r-0"
                    style={{
                      backgroundImage: `url(${thumb})`,
                      backgroundSize: 'cover',
                      backgroundPosition: 'center',
                    }}
                  />
                ))}
              </div>

              {/* Dark overlay */}
              <div className="absolute inset-0 bg-black/40" />

              {/* Selected range */}
              <div
                className="absolute top-0 h-full bg-primary/30 border-l-2 border-r-2 border-primary"
                style={{
                  left: `${(startTime / duration) * 100}%`,
                  width: `${((endTime - startTime) / duration) * 100}%`,
                }}
              />

              {/* Current time indicator */}
              <div
                className="absolute top-0 w-0.5 h-full bg-white shadow-lg z-10"
                style={{
                  left: `${(currentTime / duration) * 100}%`,
                }}
              />

              {/* Start handle */}
              <div
                onMouseDown={handleStartDrag}
                onTouchStart={handleStartDrag}
                className="absolute top-1/2 -translate-y-1/2 w-4 sm:w-3 h-full bg-primary rounded-l cursor-ew-resize z-20 hover:scale-x-125 transition-transform touch-none"
                style={{
                  left: `calc(${(startTime / duration) * 100}% - 8px)`,
                }}
              />

              {/* End handle */}
              <div
                onMouseDown={handleEndDrag}
                onTouchStart={handleEndDrag}
                className="absolute top-1/2 -translate-y-1/2 w-4 sm:w-3 h-full bg-primary rounded-r cursor-ew-resize z-20 hover:scale-x-125 transition-transform touch-none"
                style={{
                  left: `calc(${(endTime / duration) * 100}% - 6px)`,
                }}
              />
            </div>

            {/* Time labels */}
            <div className="flex justify-between text-xs text-muted-foreground px-1">
              <span>{formatTime(startTime)}</span>
              <span>{formatTime(currentTime)}</span>
              <span>{formatTime(endTime)}</span>
            </div>
          </div>
        </div>
      </Card>

      {/* Hidden canvas for thumbnail generation */}
      <canvas ref={canvasRef} className="hidden" />

      <div className="flex items-center gap-2 text-sm text-muted-foreground bg-muted/50 p-3 rounded-lg">
        <Scissors className="h-4 w-4" />
        <span>
          Selected: {formatTime(startTime)} - {formatTime(endTime)} ({formatTime(selectedDuration)})
        </span>
      </div>
    </motion.div>
  );
}
