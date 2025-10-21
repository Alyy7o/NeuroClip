import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { History as HistoryIcon, ScissorsLineDashed, EyeOff, Minimize2, Calendar, FileVideo, Play } from 'lucide-react';
import { DashboardLayout } from '@/components/DashboardLayout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { supabase } from '@/integrations/supabase/client';
import { useAuth } from '@/contexts/AuthContext';
import { useNavigate } from 'react-router-dom';
import { Badge } from '@/components/ui/badge';

const moduleIcons = {
  summarization: ScissorsLineDashed,
  blurring: EyeOff,
  compression: Minimize2,
};

const moduleColors = {
  summarization: 'from-violet-500 to-purple-600',
  blurring: 'from-blue-500 to-cyan-600',
  compression: 'from-emerald-500 to-teal-600',
};

export default function History() {
  const [history, setHistory] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const { user } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    fetchHistory();
  }, []);

  const fetchHistory = async () => {
    try {
      const { data, error } = await supabase
        .from('processing_history')
        .select('*')
        .eq('user_id', user?.id)
        .order('created_at', { ascending: false });

      if (error) throw error;
      setHistory(data || []);
    } catch (error) {
      console.error('Error fetching history:', error);
    } finally {
      setLoading(false);
    }
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  if (loading) {
    return (
      <DashboardLayout>
        <div className="flex items-center justify-center min-h-[400px]">
          <div className="h-12 w-12 animate-spin rounded-full border-4 border-primary border-t-transparent" />
        </div>
      </DashboardLayout>
    );
  }

  return (
    <DashboardLayout>
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="max-w-6xl mx-auto space-y-4 sm:space-y-6 px-4 sm:px-6"
      >
        <div className="flex flex-col sm:flex-row sm:items-center gap-3 sm:gap-4">
          <div className="flex h-10 w-10 sm:h-12 sm:w-12 items-center justify-center rounded-xl bg-gradient-to-br from-slate-500 to-gray-600 text-white shadow-lg flex-shrink-0">
            <HistoryIcon className="h-5 w-5 sm:h-6 sm:w-6" />
          </div>
          <div className="min-w-0">
            <h1 className="text-2xl sm:text-3xl font-bold truncate">Processing History</h1>
            <p className="text-sm sm:text-base text-muted-foreground">
              View all your past video processing jobs
            </p>
          </div>
        </div>

        {history.length === 0 ? (
          <Card className="gradient-card border-border/50">
            <CardContent className="py-12 text-center">
              <FileVideo className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
              <h3 className="text-lg font-semibold mb-2">No history yet</h3>
              <p className="text-muted-foreground">
                Your processed videos will appear here
              </p>
            </CardContent>
          </Card>
        ) : (
          <div className="grid gap-4">
            {history.map((item, index) => {
              const Icon = moduleIcons[item.module as keyof typeof moduleIcons];
              const gradient = moduleColors[item.module as keyof typeof moduleColors];

              return (
                <motion.div
                  key={item.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.05 }}
                >
                  <Card className="gradient-card border-border/50 hover-lift transition-all hover:scale-[1.01]">
                    <CardHeader className="p-4 sm:p-6">
                      <div className="flex flex-col gap-3 sm:gap-4">
                        <div className="flex items-start justify-between gap-3">
                          <div className="flex items-center gap-2 sm:gap-3 min-w-0 flex-1">
                            <div className={`flex h-8 w-8 sm:h-10 sm:w-10 items-center justify-center rounded-lg bg-gradient-to-br ${gradient} text-white shadow-md flex-shrink-0`}>
                              <Icon className="h-4 w-4 sm:h-5 sm:w-5" />
                            </div>
                            <div className="min-w-0 flex-1">
                              <CardTitle className="capitalize text-base sm:text-lg">{item.module}</CardTitle>
                              <CardDescription className="flex items-center gap-1.5 sm:gap-2 mt-1 text-xs sm:text-sm">
                                <Calendar className="h-3 w-3 flex-shrink-0" />
                                <span className="truncate">{formatDate(item.created_at)}</span>
                              </CardDescription>
                            </div>
                          </div>
                          <div className="flex items-center gap-2 flex-shrink-0">
                            <Badge
                              variant={item.status === 'completed' ? 'default' : 'secondary'}
                              className="capitalize text-xs"
                            >
                              {item.status}
                            </Badge>
                          </div>
                        </div>
                        <Button
                          size="sm"
                          onClick={() => navigate(`/${item.module}`)}
                          className="gradient-primary w-full sm:w-auto"
                        >
                          <Play className="h-4 w-4 mr-1" />
                          <span>Open Module</span>
                        </Button>
                      </div>
                    </CardHeader>
                    <CardContent className="p-4 sm:p-6 pt-0">
                      <div className="space-y-2 text-xs sm:text-sm">
                        {item.input_url && (
                          <p className="text-muted-foreground break-all">
                            <span className="font-medium">Input:</span> {item.input_url}
                          </p>
                        )}
                        {item.query && (
                          <p className="text-muted-foreground line-clamp-2">
                            <span className="font-medium">Query:</span> {item.query}
                          </p>
                        )}
                        {item.module === 'compression' && item.original_size && (
                          <div className="flex flex-col sm:flex-row sm:flex-wrap gap-2">
                            <p className="text-muted-foreground">
                              <span className="font-medium">Original:</span>{' '}
                              {(item.original_size / 1024 / 1024).toFixed(2)} MB
                            </p>
                            <p className="text-muted-foreground">
                              <span className="font-medium">Compressed:</span>{' '}
                              {(item.processed_size / 1024 / 1024).toFixed(2)} MB
                            </p>
                            <p className="text-accent font-medium">
                              {Math.round(((item.original_size - item.processed_size) / item.original_size) * 100)}% reduction
                            </p>
                          </div>
                        )}
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>
              );
            })}
          </div>
        )}
      </motion.div>
    </DashboardLayout>
  );
}
