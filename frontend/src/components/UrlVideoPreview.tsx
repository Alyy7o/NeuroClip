import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { isYouTubeUrl, getYouTubeEmbedUrl } from '@/lib/videoUtils';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';

interface UrlVideoPreviewProps {
  url: string;
  onTimeRangeChange?: (start: number, end: number) => void;
}

export function UrlVideoPreview({ url, onTimeRangeChange }: UrlVideoPreviewProps) {
  const [videoUrl, setVideoUrl] = useState('');
  const isYouTube = isYouTubeUrl(url);

  useEffect(() => {
    if (isYouTube) {
      const embedUrl = getYouTubeEmbedUrl(url);
      setVideoUrl(embedUrl || url);
    } else {
      setVideoUrl(url);
    }

    // For URL videos, set default time range
    if (onTimeRangeChange) {
      onTimeRangeChange(0, 60); // Default 60 seconds
    }
  }, [url, isYouTube, onTimeRangeChange]);

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className="space-y-4"
    >
      <Card className="overflow-hidden bg-muted/30">
        <div className="aspect-video w-full">
          {isYouTube ? (
            <iframe
              src={videoUrl}
              className="w-full h-full"
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
              allowFullScreen
              title="YouTube video preview"
            />
          ) : (
            <video
              src={videoUrl}
              controls
              className="w-full h-full object-contain bg-black"
            >
              Your browser does not support the video tag.
            </video>
          )}
        </div>
      </Card>
      
      <div className="text-sm text-muted-foreground text-center">
        <p>Video loaded from URL</p>
        {isYouTube && (
          <p className="text-xs mt-1">Note: Time range selection not available for YouTube videos</p>
        )}
      </div>
    </motion.div>
  );
}
