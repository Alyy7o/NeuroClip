import { useState } from 'react';
import { motion } from 'framer-motion';
import { Upload, Link2, X } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Card } from '@/components/ui/card';

interface VideoInputProps {
  onFileSelect: (file: File) => void;
  onUrlSubmit: (url: string) => void;
  file: File | null;
  url?: string;
  disabled?: boolean;
  hideUrl?: boolean;
}

export function VideoInput({ onFileSelect, onUrlSubmit, file, url = '', disabled, hideUrl }: VideoInputProps) {
  const [dragActive, setDragActive] = useState(false);
  const [inputUrl, setInputUrl] = useState(url);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const droppedFile = e.dataTransfer.files[0];
      if (droppedFile.type.startsWith('video/')) {
        onFileSelect(droppedFile);
      }
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      onFileSelect(e.target.files[0]);
    }
  };

  const handleUrlSubmit = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (inputUrl.trim()) {
      onUrlSubmit(inputUrl.trim());
    }
  };

  return (
    <Tabs defaultValue="upload" className="w-full">
      <TabsList className={`grid w-full ${hideUrl ? 'grid-cols-1' : 'grid-cols-2'}`}>
        <TabsTrigger value="upload">Upload File</TabsTrigger>
        {!hideUrl && <TabsTrigger value="url">Video URL</TabsTrigger>}
      </TabsList>

      <TabsContent value="upload">
        <Card className="p-6">
          <div
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            className={`relative border-2 border-dashed rounded-xl p-8 transition-all ${
              dragActive ? 'border-primary bg-primary/5' : 'border-border'
            } ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
          >
            <input
              type="file"
              accept="video/*"
              onChange={handleFileChange}
              disabled={disabled}
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
            />

            <div className="text-center space-y-4">
              {file ? (
                <motion.div
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  className="space-y-2"
                >
                  <div className="mx-auto w-16 h-16 rounded-full bg-primary/10 flex items-center justify-center">
                    <Upload className="h-8 w-8 text-primary" />
                  </div>
                  <div>
                    <p className="font-medium">{file.name}</p>
                    <p className="text-sm text-muted-foreground">
                      {(file.size / 1024 / 1024).toFixed(2)} MB
                    </p>
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={(e) => {
                      e.stopPropagation();
                      onFileSelect(null as any);
                    }}
                  >
                    <X className="h-4 w-4 mr-2" />
                    Remove
                  </Button>
                </motion.div>
              ) : (
                <>
                  <div className="mx-auto w-16 h-16 rounded-full bg-muted flex items-center justify-center">
                    <Upload className="h-8 w-8 text-muted-foreground" />
                  </div>
                  <div>
                    <p className="font-medium">Drop your video here</p>
                    <p className="text-sm text-muted-foreground">
                      or click to browse
                    </p>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Supports MP4, MOV, and more
                  </p>
                </>
              )}
            </div>
          </div>
        </Card>
      </TabsContent>

      {!hideUrl && (
        <TabsContent value="url">
          <Card className="p-6">
            <form onSubmit={handleUrlSubmit} className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="video-url">Video URL</Label>
                <div className="flex gap-2">
                  <div className="relative flex-1">
                    <Link2 className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                    <Input
                      id="video-url"
                      type="url"
                      placeholder="https://youtube.com/watch?v=..."
                      value={inputUrl}
                      onChange={(e) => setInputUrl(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') {
                          e.preventDefault();
                          handleUrlSubmit();
                        }
                      }}
                      disabled={disabled}
                      className="pl-10"
                    />
                  </div>
                  <Button type="button" onClick={handleUrlSubmit} disabled={disabled || !inputUrl.trim()}>
                    Load
                  </Button>
                </div>
              </div>
              <p className="text-xs text-muted-foreground">
                Supports YouTube, Vimeo, and direct video links
              </p>
            </form>
          </Card>
        </TabsContent>
      )}
    </Tabs>
  );
}
