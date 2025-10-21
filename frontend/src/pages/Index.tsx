import { motion } from 'framer-motion';
import { ArrowRight, Sparkles, EyeOff, Minimize2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useNavigate } from 'react-router-dom';
import { ThemeToggle } from '@/components/ThemeToggle';
import { useAuth } from '@/contexts/AuthContext';

export default function Index() {
  const navigate = useNavigate();
  const { user } = useAuth();

  if (user) {
    navigate('/dashboard');
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-muted">
      <header className="container mx-auto px-2 py-1 flex items-center justify-between">
        <div className="flex items-center gap-2">
          {/* <div className="flex h-18 w-18 items-center justify-center rounded-xl bg-white shadow-glow overflow-hidden"> */}
            <img src="/logo-n-1080.png" alt="NeuroClip" className="h-32 w-32 object-contain" />
          {/* </div> */}
          {/* <span className="text-2xl font-bold">NeuroClip</span> */}
        </div>
        <div className="flex items-center gap-4">
          <ThemeToggle />
          <Button onClick={() => navigate('/auth')}>Get Started</Button>
        </div>
      </header>

      <main className="container mx-auto px-6 py-20">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center max-w-4xl mx-auto space-y-8"
        >
          <h1 className="text-6xl font-bold leading-tight">
            Professional Video Processing{' '}
            <span className="bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
              Made Simple
            </span>
          </h1>
          
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            AI-powered video summarization, intelligent blurring, and smart compression—all in one platform.
          </p>

          <Button size="lg" className="gradient-primary hover:opacity-90" onClick={() => navigate('/auth')}>
            Start Processing <ArrowRight className="ml-2 h-5 w-5" />
          </Button>

          <div className="grid md:grid-cols-3 gap-6 mt-20">
            {[
              { icon: Sparkles, title: 'Smart Summarization', desc: 'Extract key highlights instantly' },
              { icon: EyeOff, title: 'Auto Blurring', desc: 'Protect sensitive content' },
              { icon: Minimize2, title: 'Compression', desc: 'Reduce size, keep quality' },
            ].map((feature, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 + i * 0.1 }}
                className="p-6 rounded-2xl gradient-card border border-border/50"
              >
                <feature.icon className="h-8 w-8 text-primary mb-4" />
                <h3 className="font-semibold mb-2">{feature.title}</h3>
                <p className="text-sm text-muted-foreground">{feature.desc}</p>
              </motion.div>
            ))}
          </div>
        </motion.div>
      </main>
    </div>
  );
}
