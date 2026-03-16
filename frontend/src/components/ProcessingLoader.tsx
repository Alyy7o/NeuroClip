import { motion } from 'framer-motion';
import { ScissorsLineDashed, CheckCircle2, CircleDashed, Loader2 } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';

interface ProcessingLoaderProps {
  steps?: string[];
  currentStep?: number;
  uploadProgress?: number;
}

const defaultSteps = [
  'Uploading video...',
  'Analyzing content...',
  'Processing frames...',
  'Finalizing output...',
];

export function ProcessingLoader({ 
  steps = defaultSteps, 
  currentStep = 0, 
  uploadProgress = 0 
}: ProcessingLoaderProps) {
  
  // Calculate percentage for the overall progress bar
  // If we are on step 0 (Uploading), use uploadProgress
  // If we are beyond step 0, distribute remaining percentage
  const totalSteps = steps.length;
  let overallPercentage = 0;
  
  if (currentStep === 0) {
    overallPercentage = uploadProgress * 0.4; // Uploading is 40% of the visual bar
  } else {
    overallPercentage = 40 + ((currentStep / (totalSteps - 1)) * 60);
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="fixed inset-0 z-50 flex items-center justify-center bg-background/80 backdrop-blur-sm p-4"
    >
      <Card className="gradient-card border-border/50 w-full max-w-md mx-auto shadow-2xl">
        <CardContent className="p-4 sm:p-6 md:p-8 space-y-4 sm:space-y-6">
          <div className="flex flex-col items-center space-y-3 sm:space-y-4">
            <motion.div
              animate={{
                rotate: 360,
                scale: [1, 1.1, 1],
              }}
              transition={{
                rotate: { duration: 4, repeat: Infinity, ease: 'linear' },
                scale: { duration: 2, repeat: Infinity, ease: 'easeInOut' },
              }}
              className="flex h-12 w-12 sm:h-16 sm:w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-violet-500 to-purple-600 text-white shadow-lg"
            >
              <ScissorsLineDashed className="h-6 w-6 sm:h-8 sm:w-8" />
            </motion.div>

            <div className="text-center">
              <h3 className="text-lg sm:text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-white to-white/70">
                {currentStep === 0 ? `Uploading Video (${uploadProgress}%)` : 'Processing Video'}
              </h3>
              <p className="text-xs sm:text-sm text-muted-foreground mt-1">
                Step {currentStep + 1} of {totalSteps}: {steps[currentStep]}
              </p>
            </div>
          </div>

          <div className="space-y-3 py-2">
            {steps.map((step, index) => {
              const isCompleted = index < currentStep;
              const isProcessing = index === currentStep;
              
              return (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className={`flex items-center gap-3 p-2 rounded-lg transition-colors ${
                    isProcessing ? 'bg-white/5 border border-white/10' : ''
                  }`}
                >
                  <div className="flex-shrink-0">
                    {isCompleted ? (
                      <CheckCircle2 className="h-5 w-5 text-green-500" />
                    ) : isProcessing ? (
                      <Loader2 className="h-5 w-5 text-primary animate-spin" />
                    ) : (
                      <CircleDashed className="h-5 w-5 text-muted-foreground/40" />
                    )}
                  </div>
                  <span className={`text-sm ${
                    isCompleted ? 'text-muted-foreground/60 line-through' : 
                    isProcessing ? 'text-primary font-medium' : 
                    'text-muted-foreground/40'
                  }`}>
                    {step}
                  </span>
                </motion.div>
              );
            })}
          </div>

          <div className="space-y-2">
            <div className="relative h-2 w-full overflow-hidden rounded-full bg-muted/30">
              <motion.div
                className="h-full bg-gradient-to-r from-violet-500 to-purple-600"
                initial={{ width: '0%' }}
                animate={{ width: `${overallPercentage}%` }}
                transition={{ duration: 0.5 }}
              />
            </div>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}
