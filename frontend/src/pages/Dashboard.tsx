import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { Sparkles, EyeOff, Minimize2, ArrowRight } from 'lucide-react';
import { DashboardLayout } from '@/components/DashboardLayout';
import { Card, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { useAuth } from '@/contexts/AuthContext';

const modules = [
  {
    icon: Sparkles,
    title: 'Video Summarization',
    description: 'Extract key highlights from your videos with AI-powered analysis',
    path: '/summarization',
    gradient: 'from-violet-500 to-purple-600',
  },
  {
    icon: EyeOff,
    title: 'Video Blurring',
    description: 'Automatically blur faces, objects, or sensitive content',
    path: '/blurring',
    gradient: 'from-blue-500 to-cyan-600',
  },
  {
    icon: Minimize2,
    title: 'Video Compression',
    description: 'Reduce file size while maintaining maximum video quality',
    path: '/compression',
    gradient: 'from-emerald-500 to-teal-600',
  },
];

export default function Dashboard() {
  const navigate = useNavigate();
  const { user } = useAuth();

  const getGreeting = () => {
    const hour = new Date().getHours();
    if (hour < 12) return 'Good morning';
    if (hour < 18) return 'Good afternoon';
    return 'Good evening';
  };

  return (
    <DashboardLayout>
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="max-w-6xl mx-auto space-y-8"
      >
        {/* Welcome section */}
        <div>
          {/* <h1 className="text-4xl font-bold mb-2">
            {getGreeting()}, {user?.email?.split('@')[0]}!
          </h1> */}
          <p className="text-muted-foreground text-lg">
            What would you like to process today?
          </p>
        </div>

        {/* Module cards */}
        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
          {modules.map((module, index) => (
            <motion.div
              key={module.path}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
            >
              <Card
                className="group cursor-pointer hover-lift gradient-card border-border/50 overflow-hidden relative"
                onClick={() => navigate(module.path)}
              >
                <div className={`absolute inset-0 bg-gradient-to-br ${module.gradient} opacity-0 group-hover:opacity-10 transition-opacity`} />
                
                <CardHeader className="relative">
                  <div className={`mb-4 flex h-12 w-12 items-center justify-center rounded-lg bg-gradient-to-br ${module.gradient} text-white shadow-lg`}>
                    <module.icon className="h-6 w-6 " />
                  </div>
                  
                  <CardTitle className="flex items-center justify-between">
                    {module.title}
                    <ArrowRight className="h-5 w-5 opacity-0 -translate-x-2 group-hover:opacity-100 group-hover:translate-x-0 transition-all" />
                  </CardTitle>
                  
                  <CardDescription className="mt-2">
                    {module.description}
                  </CardDescription>
                </CardHeader>
              </Card>
            </motion.div>
          ))}
        </div>

        {/* Quick stats or tips */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.4 }}
          className="mt-12 p-6 rounded-2xl gradient-card border border-border/50"
        >
          <h2 className="text-xl font-semibold mb-4">💡 Pro Tips</h2>
          <ul className="space-y-2 text-muted-foreground">
            <li>• Use specific queries for better summarization results</li>
            <li>• Blurring works best with clear, well-lit videos</li>
            <li>• Compressed videos maintain quality up to 70% size reduction</li>
          </ul>
        </motion.div>
      </motion.div>
    </DashboardLayout>
  );
}
