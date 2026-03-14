
import { useSearchParams, useNavigate } from 'react-router-dom';
import { DashboardLayout } from '@/components/DashboardLayout';
import { VideoPlayer } from '@/components/VideoPlayer';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Download, ArrowLeft } from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8040';

export default function DownloadClip() {
    const [params] = useSearchParams();
    const navigate = useNavigate();

    const clipUrl = params.get('url');
    const score = params.get('score');
    const start = params.get('start');
    const end = params.get('end');

    if (!clipUrl) {
        return (
            <DashboardLayout>
                <div className="flex flex-col items-center justify-center min-h-[50vh] space-y-4">
                    <h2 className="text-2xl font-bold">No Clip Selected</h2>
                    <Button onClick={() => navigate(-1)}>Go Back</Button>
                </div>
            </DashboardLayout>
        );
    }

    const handleDownload = () => {
        // Handle clip_url that might be absolute or relative, and strip query params
        const rawUrl = clipUrl.split('?')[0];
        let path = '';
        if (rawUrl.includes('/static/')) {
            path = rawUrl.split('/static/')[1];
        } else {
            // Fallback if structure is unexpected
            path = rawUrl.split('/').pop() || '';
            if (path) path = `clips/${path}`;
        }

        if (!path) return;

        const link = document.createElement('a');
        link.href = `${API_BASE}/download?path=output_data/${path}`;
        link.download = path.split('/').pop() || 'clip.mp4';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    };

    return (
        <DashboardLayout>
            <div className="max-w-4xl mx-auto space-y-6">
                <Button variant="ghost" onClick={() => navigate(-1)} className="mb-4">
                    <ArrowLeft className="mr-2 h-4 w-4" />
                    Back to Results
                </Button>

                <Card className="gradient-card border-border/50">
                    <CardHeader>
                        <CardTitle>Download Clip</CardTitle>
                        <CardDescription>
                            Review your extracted clip and download it.
                        </CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-6">
                        <div className="rounded-lg overflow-hidden border border-border bg-black/50">
                            <VideoPlayer videoUrl={clipUrl} autoPlay={false} />
                        </div>

                        <div className="flex flex-col md:flex-row justify-between items-center gap-4 p-4 bg-card/30 rounded-lg border border-border/50">
                            <div className="space-y-1 text-center md:text-left">
                                <div className="text-sm text-muted-foreground">
                                    Time Range: <span className="text-foreground font-medium">{start}s - {end}s</span>
                                </div>
                                {score && (
                                    <div className="text-sm text-muted-foreground">
                                        Relevance Score: <span className="text-foreground font-medium">{Number(score).toFixed(3)}</span>
                                    </div>
                                )}
                            </div>

                            <Button size="lg" onClick={handleDownload} className="w-full md:w-auto gradient-primary">
                                <Download className="mr-2 h-5 w-5" />
                                Download Video
                            </Button>
                        </div>
                    </CardContent>
                </Card>
            </div>
        </DashboardLayout>
    );
}
