import { motion } from 'framer-motion';
import { ScissorsLineDashed } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';

const processingSteps = [
  'Uploading video...',
  'Analyzing content...',
  'Processing frames...',
  'Applying effects...',
  'Finalizing output...',
];

export function ProcessingLoader() {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      className="fixed inset-0 z-50 flex items-center justify-center bg-background/80 backdrop-blur-sm p-4"
    >
      <Card className="gradient-card border-border/50 w-full max-w-md mx-auto">
        <CardContent className="p-4 sm:p-6 md:p-8 space-y-4 sm:space-y-6">
          <div className="flex flex-col items-center space-y-3 sm:space-y-4">
            <motion.div
              animate={{
                rotate: 360,
                scale: [1, 1.2, 1],
              }}
              transition={{
                rotate: { duration: 2, repeat: Infinity, ease: 'linear' },
                scale: { duration: 1, repeat: Infinity, ease: 'easeInOut' },
              }}
              className="flex h-12 w-12 sm:h-16 sm:w-16 items-center justify-center rounded-xl bg-gradient-to-br from-violet-500 to-purple-600 text-white shadow-lg"
            >
              <ScissorsLineDashed className="h-6 w-6 sm:h-8 sm:w-8" />
            </motion.div>

            <h3 className="text-lg sm:text-xl font-bold text-center">Processing Your Video</h3>
            <p className="text-xs sm:text-sm text-muted-foreground text-center">
              Please wait while we process your video. This may take a few moments...
            </p>
          </div>

          <div className="space-y-2 sm:space-y-3">
            {processingSteps.map((step, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.3 }}
                className="flex items-center gap-2 sm:gap-3"
              >
                <motion.div
                  animate={{
                    scale: [1, 1.2, 1],
                    backgroundColor: ['hsl(var(--primary))', 'hsl(var(--primary) / 0.5)', 'hsl(var(--primary))'],
                  }}
                  transition={{
                    duration: 1.5,
                    repeat: Infinity,
                    delay: index * 0.3,
                  }}
                  className="h-2 w-2 rounded-full bg-primary flex-shrink-0"
                />
                <span className="text-xs sm:text-sm text-muted-foreground">{step}</span>
              </motion.div>
            ))}
          </div>

          <div className="space-y-2">
            <div className="relative h-2 w-full overflow-hidden rounded-full bg-muted">
              <motion.div
                animate={{
                  x: ['-100%', '100%'],
                }}
                transition={{
                  duration: 1.5,
                  repeat: Infinity,
                  ease: 'linear',
                }}
                className="h-full w-1/3 bg-gradient-to-r from-transparent via-primary to-transparent"
              />
            </div>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}
