/** Client-side fallback when API returns raw transcript before Groq summary. */

const ON_SCREEN_BRACKET_RE = /\s*\[On Screen:\s*[^\]]*\]/gi;
const ON_SCREEN_PAREN_RE = /\s*\(On Screen:\s*[^)]*\)/gi;
const ON_SCREEN_LOOSE_RE = /\s*On Screen:\s*[^.[\(]{4,200}/gi;
const VISUAL_RE = /\s*\[Visual Content\]:\s*[^\[]*/gi;

function collapseRepeatedPhrases(text: string): string {
  const t = text.trim();
  if (t.length < 80) return t;
  const parts = t.split(/(?<=[.!?])\s+/);
  const seen = new Set<string>();
  const unique: string[] = [];
  for (const p of parts) {
    const key = p.trim().toLowerCase().slice(0, 100);
    if (!key || seen.has(key)) continue;
    seen.add(key);
    unique.push(p.trim());
  }
  return unique.length < parts.length ? unique.join(' ') : t;
}

export function stripOcrMarkup(text: string): string {
  if (!text) return '';
  let out = text;
  for (let i = 0; i < 8; i++) {
    const prev = out;
    out = out
      .replace(ON_SCREEN_BRACKET_RE, '')
      .replace(ON_SCREEN_PAREN_RE, '')
      .replace(ON_SCREEN_LOOSE_RE, '')
      .replace(VISUAL_RE, '');
    if (out === prev) break;
  }
  return collapseRepeatedPhrases(out.replace(/\s+/g, ' ').trim());
}

export function clipDescription(
  text: string,
  llmSummary?: string | null,
  maxLen = 400
): string {
  const summary = (llmSummary || '').trim();
  const body = summary || stripOcrMarkup(text) || 'No description available.';
  if (body.length <= maxLen) return body;
  return `${body.slice(0, maxLen - 3).trim()}...`;
}
