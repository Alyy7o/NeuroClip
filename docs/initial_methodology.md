# NeuroClip: Initial Methodology (For Research Paper Submission)

## 1. Problem Statement
Long-form educational and informational videos are difficult to navigate efficiently. Users need a system that can:
- locate semantically relevant moments from natural-language queries,
- combine spoken content with on-screen visual text,
- return concise, playable clips instead of full videos,
- protect privacy by masking sensitive visual content,
- support practical post-processing (compression) for easy sharing/deployment.

## 2. Research Objective
Design and implement a multimodal video understanding pipeline that transforms raw video into searchable semantic units and returns top-ranked short clips for a given query.

## 3. System Overview
NeuroClip follows a full-stack pipeline:
- Frontend: React + TypeScript interface for upload/URL input and result playback.
- Backend: FastAPI service for transcription, OCR, embedding generation, retrieval, clip extraction, and video compression.
- Storage/Retrieval: Supabase tables with vector-enabled semantic search.
- Video processing: FFmpeg for clipping/compression and a query-driven blurring workflow for sensitive content.

## 4. Input Data and Modalities
The implemented system accepts:
- Local uploaded videos (e.g., `.mp4`, `.mov`, `.webm`).
- Video URLs (downloaded through `yt-dlp` in backend workflow).

Two information channels are fused:
- Audio-text channel: Speech-to-text transcript generated with AssemblyAI (SRT + text).
- Visual-text channel: OCR text extracted from sampled frames using EasyOCR.

## 5. Data Processing Methodology

### 5.1 Video Ingestion
Video is received through API endpoints (`/upload-video`, `/upload-via-url`, or `/upload-and-search`) and stored with a generated job identifier.

### 5.2 Transcription
AssemblyAI generates time-aligned transcript content (SRT format). SRT blocks are parsed into sentence units with:
- sentence text,
- start time,
- end time.

### 5.3 OCR-based Visual Signal Enrichment
Frames are sampled at fixed intervals (currently 3s). EasyOCR extracts visible text from each sampled frame. OCR snippets are merged into nearest transcript sentence by timestamp matching (or added as synthetic visual sentences if no close match exists).

### 5.4 Semantic Representation
Each sentence (speech + merged OCR context) is converted into dense vector embeddings using `sentence-transformers/all-MiniLM-L6-v2` (default backend) with optional OpenCLIP fallback.

### 5.5 Data Persistence
Processed artifacts are saved as:
- `.v4.json` (metadata + enriched sentences),
- `.embeddings.json` (sentence vectors),
- Supabase records (`video_embeddings`, `video_sentence_embeddings`, processing history tables).

## 6. Retrieval and Clip Generation

### 6.1 Query Understanding
User query text is embedded in the same vector space as sentence vectors.

### 6.2 Candidate Search
Two-stage retrieval is implemented:
- Primary route: LLM-assisted intelligent segment selection.
- Fallback route: cosine similarity retrieval over sentence/window embeddings.

### 6.3 Re-ranking
Optional cross-encoder re-ranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`) refines semantic relevance of shortlisted segments.

### 6.4 Clip Extraction
Top segments are expanded with temporal margins and constrained by minimum/maximum clip duration. FFmpeg extracts final clips and serves them through playback endpoints.

## 7. Additional Module (Blurring)
NeuroClip includes a query-driven blurring module for privacy-aware redaction. The user provides instructions (for example, blur faces or license plates) and an optional time range; the system applies targeted blur on relevant visual regions while preserving non-sensitive areas.

Methodologically, this module follows:
- instruction parsing from natural language query,
- region/object localization on selected frames,
- temporal propagation of blur masks across the selected interval,
- rendering of a privacy-preserved output clip.

## 8. Additional Module (Compression)
NeuroClip includes a practical post-processing module (`/compress-video`) using FFmpeg to reduce output size while preserving acceptable quality for distribution.

Current compression strategy:
- codec-aware pipeline (GPU `hevc_nvenc` when available, CPU `libx265` fallback),
- resolution capping for efficient bitrate control,
- quality/size trade-off through encoder parameters,
- output statistics (original size, compressed size, reduction percent, processing time).

## 9. Experimental Setup (Minimum Reproducible)
- Runtime: Python backend with FastAPI.
- Key libraries: `assemblyai`, `easyocr`, `sentence-transformers`, `torch`, `opencv-python`, `ffmpeg`.
- Storage/Search: Supabase with vector search RPC.
- Output unit: ranked short clips with timestamps, transcript text, optional LLM summary, plus blurred/compressed video outputs.

## 10. Evaluation Plan (Only Essential Metrics)
For initial paper submission, report:
- Retrieval quality: Precision@K, Recall@K, nDCG@K on manually labeled relevant moments.
- Temporal quality: mean absolute boundary error (seconds) between predicted and reference clip boundaries.
- Utility: average watch-time reduction vs full video and user satisfaction score (Likert scale).
- Efficiency: end-to-end latency per video and per query.
- Privacy module quality: detection/redaction accuracy for sensitive objects and user-rated privacy adequacy.
- Compression quality: compression ratio and perceptual quality retention.

## 11. Minimal Contribution Claims
The current implementation supports these defensible claims:
1. Practical multimodal fusion of ASR transcript and OCR visual text for video search.
2. End-to-end pipeline from raw video input to query-driven short clip output.
3. Query-driven privacy-preserving video editing through targeted blurring.
4. Hybrid retrieval strategy combining embedding similarity with optional LLM-guided segment reasoning.
5. Practical compression pipeline for lightweight sharing and playback.

## 12. Current Limitations (State Honestly)
- OCR quality depends on frame clarity and font visibility.
- Multilingual support is currently limited by configured OCR/ASR language settings.
- LLM-assisted retrieval quality can vary with API availability and model behavior.
- Blurring quality may vary with object visibility, motion speed, and occlusion.
- Benchmark dataset protocol and large-scale quantitative results are still to be finalized.

## 13. Ethics and Compliance (Short Statement)
NeuroClip processes user-provided media and transcript content. Deployment should ensure user consent, secure storage of uploaded files, and compliance with platform terms and data privacy requirements.

---

## Paper-Ready Figure Caption (Optional)
"NeuroClip pipeline: video ingestion -> ASR transcription -> OCR enrichment -> sentence embedding -> vector retrieval/reranking -> timestamped clip extraction -> sensitive-region blurring -> compressed output delivery."
