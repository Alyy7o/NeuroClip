import { useRef } from 'react';
import { motion } from 'framer-motion';
import { ImagePlus, X } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';

interface ReferenceImageUploadProps {
  images: File[];
  onChange: (files: File[]) => void;
  disabled?: boolean;
}

export function ReferenceImageUpload({ images, onChange, disabled }: ReferenceImageUploadProps) {
  const inputRef = useRef<HTMLInputElement>(null);

  const addFiles = (fileList: FileList | null) => {
    if (!fileList?.length) return;
    const incoming = Array.from(fileList).filter((f) => f.type.startsWith('image/'));
    onChange([...images, ...incoming]);
  };

  const removeAt = (index: number) => {
    onChange(images.filter((_, i) => i !== index));
  };

  return (
    <motion.div className="space-y-3">
      <Label>Reference images (persons to blur)</Label>
      <p className="text-xs text-muted-foreground">
        Upload clear photos of each person whose face should be blurred in the video.
      </p>
      <motion.div className="flex flex-wrap gap-3" layout>
        {images.map((file, i) => (
          <motion.div
            key={`${file.name}-${i}`}
            className="relative h-20 w-20 rounded-lg overflow-hidden border border-border bg-muted"
            layout
          >
            <img
              src={URL.createObjectURL(file)}
              alt={file.name}
              className="h-full w-full object-cover"
              onLoad={(e) => URL.revokeObjectURL((e.target as HTMLImageElement).src)}
            />
            <button
              type="button"
              disabled={disabled}
              className="absolute top-0.5 right-0.5 rounded-full bg-background/80 p-0.5 hover:bg-destructive hover:text-destructive-foreground"
              onClick={() => removeAt(i)}
              aria-label="Remove image"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          </motion.div>
        ))}
        <Button
          type="button"
          variant="outline"
          className="h-20 w-20 flex flex-col gap-1 shrink-0"
          disabled={disabled}
          onClick={() => inputRef.current?.click()}
        >
          <ImagePlus className="h-5 w-5" />
          <span className="text-[10px]">Add</span>
        </Button>
      </motion.div>
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        multiple
        className="hidden"
        disabled={disabled}
        onChange={(e) => {
          addFiles(e.target.files);
          e.target.value = '';
        }}
      />
    </motion.div>
  );
}
