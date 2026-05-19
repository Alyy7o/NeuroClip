const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8040';

/** Prefer completed when output exists; never downgrade completed → processing. */
export function effectiveHistoryStatus(
  historyStatus?: string | null,
  videoStatus?: string | null,
  hasMedia?: boolean
): string {
  if (historyStatus === 'failed' || videoStatus === 'failed') return 'failed';
  if (hasMedia) return 'completed';
  if (historyStatus === 'completed' || videoStatus === 'completed') return 'completed';
  return historyStatus || videoStatus || 'processing';
}

export function resolveMediaPath(
  resultUrl?: string | null,
  videoUrl?: string | null
): string | null {
  const raw = (resultUrl || videoUrl || '').trim();
  if (!raw) return null;
  if (raw.startsWith('http://') || raw.startsWith('https://')) return raw;
  return raw.replace(/^\/static\//, '').replace(/^\//, '');
}

export function mediaFilename(pathOrUrl: string, fallback = 'video.mp4'): string {
  const part = pathOrUrl.split(/[/\\?#]/).filter(Boolean).pop();
  return part && part.includes('.') ? part : fallback;
}

export async function downloadHistoryMedia(pathOrUrl: string, filename?: string): Promise<void> {
  const name = filename || mediaFilename(pathOrUrl);
  const headers = { 'ngrok-skip-browser-warning': 'true' };

  if (pathOrUrl.startsWith('http://') || pathOrUrl.startsWith('https://')) {
    try {
      const resp = await fetch(pathOrUrl, { headers });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const blob = await resp.blob();
      triggerBlobDownload(blob, name);
      return;
    } catch {
      window.open(pathOrUrl, '_blank');
      return;
    }
  }

  const downloadUrl = `${API_BASE}/download?path=${encodeURIComponent(pathOrUrl)}`;
  try {
    const resp = await fetch(downloadUrl, { headers });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const blob = await resp.blob();
    triggerBlobDownload(blob, name);
  } catch {
    window.open(downloadUrl, '_blank');
  }
}

function triggerBlobDownload(blob: Blob, filename: string): void {
  const blobUrl = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = blobUrl;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(blobUrl);
}
