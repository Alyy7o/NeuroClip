import os
from pathlib import Path
from typing import Optional, Union, Dict

import assemblyai as aai

# --- Load environment: Kaggle Secrets first, then .env files ---
# Priority 1: Kaggle Secrets (exact API pattern from Kaggle)
if not os.environ.get("ASSEMBLYAI_API_KEY"):
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        os.environ["ASSEMBLYAI_API_KEY"] = user_secrets.get_secret("ASSEMBLYAI_API_KEY")
    except Exception:
        pass

# Priority 2: .env files (local dev)
env_loaded = False
try:
    from dotenv import load_dotenv
    candidates = [
        Path(__file__).resolve().parent / "Semantic-search-app" / ".env",
        Path(__file__).resolve().parent.parent / ".env",
    ]
    for p in candidates:
        try:
            if p.exists():
                load_dotenv(dotenv_path=str(p))
                env_loaded = True
                break
        except Exception:
            continue
except Exception:
    env_loaded = False
if not env_loaded:
    for p in [Path(__file__).resolve().parent / "Semantic-search-app" / ".env", Path(__file__).resolve().parent.parent / ".env"]:
        try:
            if p.exists():
                for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
                    s = line.strip()
                    if not s or s.startswith("#"):
                        continue
                    if "=" in s:
                        k, v = s.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip('"').strip("'")
                        if k and v and k not in os.environ:
                            os.environ[k] = v
                break
        except Exception:
            continue


def generate_transcript_from_video(
    source: Union[str, Path],
    output_srt_path: Optional[Union[str, Path]] = None,
    language_code: Optional[str] = None,
) -> Dict[str, Union[str, Path]]:
    """
    Transcribe a video/audio from a local file path or URL using AssemblyAI's SDK.

    Args:
        source: Local file path to the media, or a public URL.
        output_srt_path: If provided, saves subtitles (SRT) to this path.
        language_code: Optional language code (e.g., "en"), if you want to force a language.

    Returns:
        A dict with keys: "id", "status", "text", and optionally "srt_path" if exported.

    Raises:
        ValueError: If the API key is missing or the source is invalid.
        Exception: If transcription fails; the exception message will describe the cause.
    """

    # Configure SDK with API key from environment
    api_key = os.getenv("ASSEMBLYAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("ASSEMBLYAI_API_KEY is not set. Add it to .env or environment.")
    aai.settings.api_key = api_key

    # Build input for SDK: recent AssemblyAI SDK accepts a local path or URL string directly
    src_str = str(source)
    input_obj = src_str

    # Optional transcription configuration
    config = None
    if language_code:
        try:
            config = aai.TranscriptionConfig(language_code=language_code)
        except Exception:
            # Fallback in case SDK version does not recognize the parameter
            config = None

    transcriber = aai.Transcriber()

    # Perform synchronous transcription
    transcript = transcriber.transcribe(input_obj, config=config) if config else transcriber.transcribe(input_obj)

    # Build result payload
    result: Dict[str, Union[str, Path]] = {
        "id": getattr(transcript, "id", ""),
        "status": getattr(transcript, "status", ""),
        "text": getattr(transcript, "text", ""),
    }

    # Optionally export subtitles in SRT format, and always try to capture srt_text
    # so callers can convert without persisting to disk.
    srt_text_out: Optional[str] = None
    try:
        tmp = transcript.export_subtitles_srt()
        if isinstance(tmp, str) and tmp:
            srt_text_out = tmp
    except TypeError:
        # Some SDK versions require a path; handle below.
        pass

    if output_srt_path:
        out_path = Path(output_srt_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if srt_text_out:
            out_path.write_text(srt_text_out, encoding="utf-8")
        else:
            # Fallback to path-based export and read back
            try:
                transcript.export_subtitles_srt(str(out_path))
                try:
                    srt_text_out = out_path.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    srt_text_out = None
            except Exception:
                # If export fails, leave without SRT
                pass
        result["srt_path"] = out_path
    else:
        # No output path requested; attempt ephemeral export if we still lack text
        if srt_text_out is None:
            try:
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=".srt") as tf:
                    temp_path = Path(tf.name)
                try:
                    transcript.export_subtitles_srt(str(temp_path))
                    srt_text_out = temp_path.read_text(encoding="utf-8", errors="replace")
                finally:
                    try:
                        temp_path.unlink(missing_ok=True)
                    except Exception:
                        pass
            except Exception:
                srt_text_out = None

    if srt_text_out is not None:
        result["srt_text"] = srt_text_out

    return result


__all__ = ["generate_transcript_from_video"]