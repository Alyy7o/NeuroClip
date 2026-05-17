import { useEffect, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import { Play, Pause, Volume2, VolumeX, Maximize, SkipForward, SkipBack } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { getYouTubeVideoId, getYouTubeEmbedUrl } from '@/lib/videoUtils';

interface VideoPlayerProps {
  videoUrl: string;
  autoPlay?: boolean;
}

export function VideoPlayer({ videoUrl, autoPlay = false }: VideoPlayerProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [isPlaying, setIsPlaying] = useState(autoPlay);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(1);
  const [isMuted, setIsMuted] = useState(false);
  const [showControls, setShowControls] = useState(true);
  const [isFullscreen, setIsFullscreen] = useState(false);

  // Check if the video is a YouTube URL
  const youtubeVideoId = getYouTubeVideoId(videoUrl);
  const isYouTubeVideo = !!youtubeVideoId;

  // For non-YouTube and non-blob URLs, fetch with ngrok header and convert to blob
  const [blobUrl, setBlobUrl] = useState<string | null>(null);
  const [loadError, setLoadError] = useState(false);
  const [videoError, setVideoError] = useState(false);

  useEffect(() => {
    // Skip blob fetching for YouTube, blob:, or data: URLs
    if (isYouTubeVideo || videoUrl.startsWith('blob:') || videoUrl.startsWith('data:')) {
      setBlobUrl(videoUrl);
      return;
    }

    let cancelled = false;
    setVideoError(false);
    setLoadError(false);
    const fetchVideo = async () => {
      try {
        const resp = await fetch(videoUrl, {
          headers: { 'ngrok-skip-browser-warning': 'true' },
        });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const blob = await resp.blob();
        if (!cancelled) {
          const url = URL.createObjectURL(blob);
          setBlobUrl(url);
          // #region agent log
          fetch('http://127.0.0.1:7349/ingest/b5b03500-6997-4666-8a59-a196e0f10b38',{method:'POST',headers:{'Content-Type':'application/json','X-Debug-Session-Id':'743c18'},body:JSON.stringify({sessionId:'743c18',location:'VideoPlayer.tsx:fetch',message:'video blob loaded',data:{bytes:blob.size,type:blob.type},timestamp:Date.now(),hypothesisId:'H1'})}).catch(()=>{});
          // #endregion
        }
      } catch (err) {
        console.error('[VideoPlayer] Failed to fetch video:', err);
        if (!cancelled) {
          setBlobUrl(videoUrl);
          setLoadError(true);
        }
      }
    };
    fetchVideo();

    return () => {
      cancelled = true;
      if (blobUrl && blobUrl.startsWith('blob:')) {
        URL.revokeObjectURL(blobUrl);
      }
    };
  }, [videoUrl, isYouTubeVideo]);

  // Keyboard controls
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      // Don't intercept if user is typing in an input/textarea
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;

      switch (e.key) {
        case ' ':
        case 'Spacebar':
          e.preventDefault();
          togglePlayPause();
          break;
        case 'ArrowLeft':
          e.preventDefault();
          skip(-5);
          break;
        case 'ArrowRight':
          e.preventDefault();
          skip(5);
          break;
        case 'f':
        case 'F':
          e.preventDefault();
          toggleFullscreen();
          break;
      }
    };

    container.addEventListener('keydown', handleKeyDown);
    return () => container.removeEventListener('keydown', handleKeyDown);
  }, [isPlaying, blobUrl]);

  // Sync fullscreen state when user exits via Escape
  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };
    document.addEventListener('fullscreenchange', handleFullscreenChange);
    return () => document.removeEventListener('fullscreenchange', handleFullscreenChange);
  }, []);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    if (autoPlay) {
      video.play();
    }
  }, [autoPlay, blobUrl]);

  const togglePlayPause = () => {
    if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.pause();
      } else {
        videoRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  const handleTimeUpdate = () => {
    if (videoRef.current) {
      setCurrentTime(videoRef.current.currentTime);
    }
  };

  const handleLoadedMetadata = () => {
    if (videoRef.current) {
      setDuration(videoRef.current.duration);
    }
  };

  const handleSeek = (value: number[]) => {
    if (videoRef.current) {
      videoRef.current.currentTime = value[0];
      setCurrentTime(value[0]);
    }
  };

  const skip = (seconds: number) => {
    if (videoRef.current) {
      videoRef.current.currentTime += seconds;
    }
  };

  const handleVolumeChange = (value: number[]) => {
    const newVolume = value[0];
    setVolume(newVolume);
    if (videoRef.current) {
      videoRef.current.volume = newVolume;
    }
    if (newVolume === 0) {
      setIsMuted(true);
    } else if (isMuted) {
      setIsMuted(false);
    }
  };

  const toggleMute = () => {
    if (videoRef.current) {
      videoRef.current.muted = !isMuted;
      setIsMuted(!isMuted);
    }
  };

  const toggleFullscreen = () => {
    if (containerRef.current) {
      if (document.fullscreenElement) {
        document.exitFullscreen();
      } else {
        containerRef.current.requestFullscreen();
      }
    }
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  // If it's a YouTube video, render an iframe
  if (isYouTubeVideo) {
    const embedUrl = getYouTubeEmbedUrl(youtubeVideoId);
    return (
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        className="relative bg-black rounded-lg overflow-hidden"
      >
        <iframe
          src={embedUrl}
          className="w-full aspect-video"
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
          allowFullScreen
        />
      </motion.div>
    );
  }

  return (
    <motion.div
      ref={containerRef}
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className="relative bg-black rounded-lg overflow-hidden group"
      tabIndex={0}
      style={{ outline: 'none' }}
    >
      {blobUrl && !videoError ? (
        <video
          ref={videoRef}
          src={blobUrl}
          type="video/mp4"
          onTimeUpdate={handleTimeUpdate}
          onLoadedMetadata={handleLoadedMetadata}
          onError={() => {
            setVideoError(true);
            // #region agent log
            fetch('http://127.0.0.1:7349/ingest/b5b03500-6997-4666-8a59-a196e0f10b38',{method:'POST',headers:{'Content-Type':'application/json','X-Debug-Session-Id':'743c18'},body:JSON.stringify({sessionId:'743c18',location:'VideoPlayer.tsx:onError',message:'video element error',data:{videoUrl:videoUrl.slice(0,80)},timestamp:Date.now(),hypothesisId:'H1'})}).catch(()=>{});
            // #endregion
          }}
          className="w-full aspect-video object-contain"
          onClick={togglePlayPause}
          playsInline
          controls={false}
        />
      ) : videoError ? (
        <motion.div className="w-full aspect-video flex flex-col items-center justify-center gap-2 bg-black/80 p-4">
          <p className="text-sm text-white/80 text-center">Preview unavailable (codec). Use Download to view the file.</p>
        </motion.div>
      ) : (
        <div className="w-full aspect-video flex items-center justify-center bg-black/80">
          <div className="flex flex-col items-center gap-2">
            <div className="h-8 w-8 animate-spin rounded-full border-2 border-violet-400 border-t-transparent" />
            <span className="text-xs text-white/60">Loading video…</span>
          </div>
        </div>
      )}

      {/* Controls Overlay */}
      <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/90 via-black/60 to-transparent p-2 sm:p-4 space-y-2 sm:space-y-3">
        {/* Timeline */}
        <div className="space-y-1">
          <Slider
            value={[currentTime]}
            min={0}
            max={duration || 100}
            step={0.1}
            onValueChange={handleSeek}
            className="cursor-pointer"
          />
          <div className="flex justify-between text-xs text-white/80">
            <span>{formatTime(currentTime)}</span>
            <span>{formatTime(duration)}</span>
          </div>
        </div>

        {/* Control Buttons */}
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-1 sm:gap-2">
            <Button
              variant="ghost"
              size="icon"
              onClick={togglePlayPause}
              className="text-white hover:bg-white/20 h-8 w-8 sm:h-10 sm:w-10"
            >
              {isPlaying ? <Pause className="h-4 w-4 sm:h-5 sm:w-5" /> : <Play className="h-4 w-4 sm:h-5 sm:w-5" />}
            </Button>

            <Button
              variant="ghost"
              size="icon"
              onClick={() => skip(-5)}
              className="text-white hover:bg-white/20 h-8 w-8 sm:h-10 sm:w-10"
            >
              <SkipBack className="h-3 w-3 sm:h-4 sm:w-4" />
            </Button>

            <Button
              variant="ghost"
              size="icon"
              onClick={() => skip(5)}
              className="text-white hover:bg-white/20 h-8 w-8 sm:h-10 sm:w-10"
            >
              <SkipForward className="h-3 w-3 sm:h-4 sm:w-4" />
            </Button>

            <div className="hidden sm:flex items-center gap-2 ml-2">
              <Button
                variant="ghost"
                size="icon"
                onClick={toggleMute}
                className="text-white hover:bg-white/20"
              >
                {isMuted || volume === 0 ? (
                  <VolumeX className="h-4 w-4" />
                ) : (
                  <Volume2 className="h-4 w-4" />
                )}
              </Button>
              <Slider
                value={[isMuted ? 0 : volume]}
                min={0}
                max={1}
                step={0.1}
                onValueChange={handleVolumeChange}
                className="w-16 md:w-20 cursor-pointer"
              />
            </div>
          </div>

          <div className="flex items-center gap-1">
            <Button
              variant="ghost"
              size="icon"
              onClick={toggleMute}
              className="text-white hover:bg-white/20 h-8 w-8 sm:hidden"
            >
              {isMuted || volume === 0 ? (
                <VolumeX className="h-4 w-4" />
              ) : (
                <Volume2 className="h-4 w-4" />
              )}
            </Button>
            <Button
              variant="ghost"
              size="icon"
              onClick={toggleFullscreen}
              className="text-white hover:bg-white/20 h-8 w-8 sm:h-10 sm:w-10"
            >
              <Maximize className="h-3 w-3 sm:h-4 sm:w-4" />
            </Button>
          </div>
        </div>
      </div>

      {/* Play button overlay */}
      {!isPlaying && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="absolute inset-0 flex items-center justify-center"
        >
          <Button
            onClick={togglePlayPause}
            size="icon"
            className="h-16 w-16 rounded-full bg-white/20 hover:bg-white/30 backdrop-blur-sm"
          >
            <Play className="h-8 w-8 text-white" />
          </Button>
        </motion.div>
      )}
    </motion.div>
  );
}
