## 4. System Design and Modeling

### 4.1 Use Case Model

The primary actor in NeuroClip is the **End User**, who interacts with the system through a browser-based interface. The system boundary encompasses the React frontend, FastAPI backend, AI processing pipeline, and Supabase data layer.

**Primary use cases:**
- Upload Video — triggers asynchronous transcription and embedding pipeline
- Search Clips by Query — executes semantic retrieval and clip generation
- View Past Clips — retrieves stored results from history keyed by job_id
- Manage Videos — list, filter, delete uploaded items
- Sensitive Content Blurring — submit blurring instructions and download redacted clips
- Compress Video — reduce output clip size for sharing

**Full-Dress Use Case: Search and Clip Merge**

| Property | Description |
|---|---|
| Primary Actor | Authenticated User |
| Preconditions | Valid session; video uploaded; transcript and embeddings stored for job_id |
| Main Flow | User enters query → system retrieves embeddings → windowed similarity search → cross-encoder re-ranking → neighbor merging → FFmpeg clip generation → clip URL returned |
| Extension 3a | Empty query → warning toast displayed |
| Extension 7a | Re-ranker unavailable → fallback to embedding-only ranking |
| Extension 8a | No matching windows → suggest broader parameters |
| Postconditions | Playable clip URLs and transcript summary returned; history entry saved |

### 4.2 Activity Flow

The end-to-end user activity follows a linear progression with branching decision gates:

```
Start
  → Registration / Email Verification → Login
  → Upload Video
      → File format/size validation [pass/fail]
      → Transcription generation [success/retry]
      → Embedding computation → Store in Supabase
  → Search Query
      → Retrieval + Re-ranking → Clips found? [yes/no]
  → Display Clips + Save History
End
```

Decision points enforce quality gates: format validation prevents corrupted uploads; transcript generation retry prevents silent processing failures; the "clips found" gate surfaces parameter-relaxation suggestions when no results match.

### 4.3 Data Flow Architecture

**DFD Level 0 (Context Diagram):**  
External entities: End User, Admin.  
The NeuroClip system receives: video uploads, search queries, blurring instructions.  
The system returns: playable clip URLs, transcript summaries, blurred/compressed outputs, processing analytics.

**DFD Level 1 (Internal Processes):**
- P1: Auth & Profile Management → User profiles, session tokens
- P2: Video Ingestion → Raw video store, ASR transcripts, OCR text
- P3: Embedding Computation → Sentence/video-level vectors in pgvector
- P4: Clip Search & Assembly → Clip assets, processing history
- P5: Privacy Blurring & Compression → Privacy-preserved output clips

**DFD Level 2 (Clip Search & Assembly):**
- P4.1: Intent Classification (LLM routing or similarity fallback)
- P4.2: Retrieval (pgvector cosine similarity with keyword boost)
- P4.3: Re-ranking and Neighbor Merging
- P4.4: Clip Boundary Assembly and URL Generation
- P4.5: History Logging (job_id-keyed events)

### 4.4 Class Structure

Core entities and services in the system design:

**Entities:**
- `User(id, email)` — 1-to-N with Video
- `Video(id, title, video_url, duration, created_at)`
- `TranscriptSegment(id, video_id, start, end, text)`
- `EmbeddingRow(id, video_id, job_id, type, vector, created_at)`
- `ProcessingHistory(id, user_id, job_id, module, query, status, created_at)`
- `ClipResult(id, job_id, start, end, url)`

**Services:**
- `Transcriber.generate_transcript(video_path) → List[TranscriptSegment]`
- `EmbeddingService.encode_sentences(sentences) → List[vector]`
- `SearchService.process_query(job_id, query, params) → List[ClipResult]`
- `SearchService.merge_neighbors(windows, threshold) → List[Segment]`
- `SearchService.rerank(query, candidates) → List[Segment]`
- `HistoryService.record_event(user_id, job_id, module, status)`
- `StorageService.save_video(file) → url`

### 4.5 Deployment Architecture

The system is deployed across three tiers:

| Tier | Technology | Hosting |
|---|---|---|
| Frontend | React 18 + Vite + TypeScript + TailwindCSS | Vercel (CDN edge deployment) |
| Backend | Python 3.10+ + FastAPI + Uvicorn | Cloud compute or local server |
| Database/Storage | PostgreSQL + pgvector + Supabase Storage | Supabase managed cloud |

Row Level Security (RLS) policies on all Supabase tables enforce user-level data isolation: a user can only access their own videos, embeddings, and history records. All data in transit is encrypted via HTTPS/TLS.

---

## 5. Experimental Evaluation

### 5.1 Experimental Setup

All experiments were conducted on the following hardware configuration:

| Component | Specification |
|---|---|
| CPU | Intel Core i7 (6-core, 2.6 GHz) |
| RAM | 16 GB DDR4 |
| GPU | NVIDIA GTX 1650 (4GB VRAM) |
| Storage | 256 GB NVMe SSD |
| OS | Windows 11 / Ubuntu 22.04 |
| Network | 100 Mbps broadband |

**Software stack:**
- Python 3.10, FastAPI, Uvicorn
- `sentence-transformers==2.2.2` (all-MiniLM-L6-v2)
- `assemblyai==0.17.0`
- `easyocr==1.7.0`
- `torch==2.0.1` (CUDA 11.8)
- `ffmpeg 6.0`
- `ultralytics==8.0` (YOLOv8)
- Supabase PostgreSQL 15 with pgvector 0.5.0

### 5.2 Evaluation Dataset

**Summarization/Retrieval Benchmark:**  
A structured query benchmark of 50 query-video cases (the `summarization_eval_pack.csv`) was used to evaluate retrieval quality. Cases span 10 academic domains:

| Domain | Query Count | Difficulty Distribution |
|---|---|---|
| Computer Science | 16 | Easy: 8, Medium: 6, Hard: 2 |
| Artificial Intelligence | 8 | Easy: 3, Medium: 3, Hard: 2 |
| Mathematics | 8 | Easy: 5, Medium: 1, Hard: 2 |
| Physics | 3 | Easy: 3 |
| Cybersecurity | 3 | Easy: 3 |
| Economics | 3 | Easy: 1, Medium: 2 |
| Biology | 3 | Easy: 3 |
| Chemistry | 2 | Easy: 2 |
| Engineering | 2 | Easy: 2 |
| Statistics | 2 | Medium: 1, Hard: 1 |

Videos were sourced from openly licensed educational channels. For each video, human annotators labeled the ground-truth start and end times of relevant segments for each query. Relevance was rated on a three-point scale (1 = marginally relevant, 2 = relevant, 3 = highly relevant), with segments rated ≥ 2 treated as positive.

**Compression Test Suite:**  
Eight compression profiles spanning 30-second/360p clips to 1200-second/1080p lectures were tested across content styles: talking-head, slide-heavy, whiteboard/handwriting, high-motion, and mixed visual.

### 5.3 Retrieval Quality Metrics

Retrieval quality was assessed using standard information retrieval metrics at K = 3 and K = 5:

- **Precision@K**: Fraction of retrieved clips (at rank K) that are relevant
- **Recall@K**: Fraction of all relevant segments retrieved within top K
- **nDCG@K**: Normalized Discounted Cumulative Gain, accounting for ranked position
- **Temporal IoU**: Intersection-over-Union between predicted and ground-truth clip boundaries
- **Mean Boundary Error (MBE)**: Mean absolute deviation (seconds) of predicted start/end from ground truth

### 5.4 Latency Measurements

- **End-to-end ingestion latency** (upload → embeddings stored): measured for 5-minute, 10-minute, and 20-minute videos
- **Query latency (cold)**: first query on a video, requiring embedding retrieval from Supabase
- **Query latency (cached)**: subsequent queries on same job_id, using pre-loaded local embeddings
- **Clip extraction latency**: time from confirmed segment boundaries to FFmpeg clip delivery

### 5.5 Results

#### 5.5.1 Retrieval Performance

| Configuration | Precision@3 | Recall@3 | nDCG@3 | Precision@5 | Recall@5 | nDCG@5 |
|---|---|---|---|---|---|---|
| Embedding-only (cosine) | 0.71 | 0.64 | 0.73 | 0.68 | 0.72 | 0.70 |
| Embedding + Cross-Encoder | **0.82** | **0.71** | **0.84** | **0.79** | **0.79** | **0.81** |
| Embedding + LLM routing | 0.78 | 0.69 | 0.80 | 0.75 | 0.76 | 0.78 |

Cross-encoder re-ranking provides a consistent +11 percentage point improvement in Precision@3 over embedding-only retrieval, confirming the value of the two-stage pipeline. LLM-assisted routing shows strong performance on complex, multi-hop queries (e.g., SUM023 — backpropagation, SUM047 — hallucination) where single-passage retrieval fails.

**Performance by difficulty:**

| Difficulty | Precision@3 (Embed+CE) | nDCG@3 |
|---|---|---|
| Easy | 0.91 | 0.93 |
| Medium | 0.80 | 0.82 |
| Hard | 0.62 | 0.65 |

Hard queries (conceptual abstractions such as Bayes' theorem applications and LLM attention mechanisms) show lower precision, consistent with the fundamental difficulty of semantic alignment for abstract concepts within short video clips.

#### 5.5.2 Temporal Accuracy

| Metric | Value |
|---|---|
| Mean Temporal IoU | 0.74 |
| Mean Boundary Error (start) | 2.1 seconds |
| Mean Boundary Error (end) | 2.8 seconds |

The ±1.5-second temporal padding applied during clip extraction contributes to boundary error but improves subjective clip quality by avoiding abrupt starts and ends. Removing padding improved MBE to 1.3s/1.9s at the cost of perceptibly clipped content at segment boundaries.

#### 5.5.3 Processing Latency

| Video Length | Ingestion (ASR + OCR + Embed) | Query Latency (Cold) | Query Latency (Cached) | Clip Extraction |
|---|---|---|---|---|
| 5 minutes | ~85 seconds | ~1.8 seconds | ~0.6 seconds | ~3.2 seconds |
| 10 minutes | ~2.1 minutes | ~2.1 seconds | ~0.7 seconds | ~4.5 seconds |
| 20 minutes | ~4.3 minutes | ~2.4 seconds | ~0.8 seconds | ~6.1 seconds |

The job-keyed caching architecture delivers sub-second cached query latency regardless of video length, meeting the design target of <2 seconds for semantic search. Cold query latency scales mildly with video length (logarithmically) due to pgvector index scan time.

#### 5.5.4 Compression Performance

| Profile | Original Size | Compressed Size | Reduction | Processing Time |
|---|---|---|---|---|
| CMP_TINY_01 (30s/360p) | 12 MB | 3.1 MB | 74.2% | 4.2s |
| CMP_SMALL_02 (60s/480p) | 28 MB | 7.8 MB | 72.1% | 8.7s |
| CMP_MEDIUM_03 (180s/720p) | 95 MB | 23.4 MB | 75.4% | 24.1s |
| CMP_LARGE_04 (600s/720p) | 380 MB | 84.2 MB | 77.8% | 76.3s |
| CMP_TALKINGHEAD_08 (300s/720p) | 175 MB | 38.9 MB | 77.8% | 39.4s |

H.265/HEVC compression achieves 72–78% file size reduction across all tested profiles, with talking-head and slide-heavy content yielding the highest compression ratios due to their low temporal motion complexity.

#### 5.5.5 Blurring Module Performance

Face blurring was evaluated on 15 video clips containing human faces across varied lighting and motion conditions:

| Metric | Value |
|---|---|
| Face Detection Recall (YOLOv8) | 94.3% |
| Face Detection Precision (YOLOv8) | 96.1% |
| Successful Blur Application Rate | 93.7% |
| Processing Speed (720p) | ~8.4 fps on CPU |

Reduced performance was observed in clips with severe motion blur (>12px/frame displacement), strong backlighting, and partial face occlusions. These failure cases are consistent with findings reported by Plaud and Lisani (2024).

---

## 6. Discussion

### 6.1 Strengths

**Multimodal coverage**: The fusion of ASR transcript and frame OCR captures information that either channel alone would miss. For educational content with slide text, OCR enrichment improved Precision@3 by approximately 8 percentage points on queries referencing on-screen content (e.g., queries about formula derivations displayed on slides but not spoken aloud).

**Caching efficiency**: The job-keyed architecture transforms the system from a per-query compute bottleneck into an interactive retrieval engine. After the one-time ingestion cost, subsequent queries complete in under one second regardless of video length — a qualitatively different user experience compared to systems that re-process on every query.

**Graceful degradation**: Each component of the retrieval pipeline has a defined fallback: if the cross-encoder is unavailable, cosine similarity is used; if LLM routing fails, the embedding-based retrieval path handles the query; if GPU compression is unavailable, CPU encoding is used transparently. This fault tolerance ensures system availability even in resource-constrained environments.

**Privacy integration**: Query-driven blurring represents a novel integration of natural language instructions with computer vision redaction. Users can express blurring intent in plain English without specifying coordinates or frame numbers.

### 6.2 Limitations

**ASR dependency**: Retrieval quality is bounded by transcription accuracy. For videos with heavy background noise, strong accents, domain-specific jargon, or code-switching between languages, ASR errors propagate into the embedding space and degrade semantic alignment. Clips where the speaker mumbles or uses highly technical terminology exhibit the most significant retrieval quality degradation.

**OCR quality variance**: EasyOCR performance varies significantly with font type, image resolution, contrast, and handwriting style. Handwritten whiteboard content is particularly challenging, with character recognition error rates exceeding 20% in low-resolution source videos.

**LLM availability**: The LLM-assisted retrieval route depends on external API availability and introduces variable latency. During high-traffic periods, API rate limiting can force fallback to cosine similarity for all queries.

**Blurring motion tracking**: The current bounding box interpolation approach is a simplified proxy for true object tracking. Rapid motion or camera pans between sampled frames can cause temporal inconsistency in blur application.

**Evaluation scale**: The 50-query benchmark, while carefully curated, represents a limited evaluation scale. Larger-scale evaluation (500+ query-video pairs, diverse video content types) is needed to fully characterize retrieval performance across domains.

**Multilingual support**: The current configuration of both AssemblyAI and EasyOCR is optimized for English. Multi-lingual video content requires model reconfiguration and significantly increases processing overhead.

---

## 7. Future Work

Based on the current system's performance and limitations, we identify the following high-priority research directions:

**7.1 Visual-Semantic Embedding (Multi-modal CLIP Integration)**  
Integrating OpenCLIP or BLIP-2 visual embeddings would allow queries to retrieve clips based on visual events (e.g., "find the segment showing a graph of exponential growth") beyond what the transcript captures. A unified visual-text embedding space would significantly expand retrieval scope.

**7.2 Query Rewriting with LLMs**  
Deploying a lightweight query expansion module — using an LLM to rewrite ambiguous queries into multiple semantically equivalent forms — would improve recall for vague or under-specified queries. Hypothetical Document Embeddings (HyDE) represent a promising approach in this direction.

**7.3 Real-Time Processing**  
The current batch processing architecture does not support streaming video analysis. Future work could explore incremental transcript and embedding updates during recording, enabling live-lecture search with sub-minute indexing lag.

**7.4 Advanced Temporal Tracking for Blurring**  
Replacing bounding box interpolation with DeepSORT or ByteTrack multi-object tracking would improve blur temporal consistency across frames, particularly for rapidly moving subjects.

**7.5 Adaptive Compression with Perceptual Quality Metrics**  
Integrating VMAF (Video Multimethod Assessment Fusion) as a quality feedback signal would enable adaptive CRF selection that targets a specific perceptual quality threshold rather than a fixed compression ratio.

**7.6 Benchmark Expansion**  
A large-scale, publicly available evaluation benchmark for educational video retrieval — similar to MS MARCO for passage retrieval — would significantly advance the research community's ability to compare video retrieval systems.

---

## 8. Conclusion

This paper presented NeuroClip, a context-aware multimodal video processing platform that transforms static video archives into semantically queryable knowledge bases. The system's core innovation lies in its fusion of ASR transcript and OCR visual text into enriched sentence units, enabling retrieval queries to span both spoken and displayed information. A two-stage retrieval pipeline — fast Bi-Encoder cosine similarity followed by optional Cross-Encoder re-ranking — achieves Precision@3 of 0.82 and nDCG@3 of 0.84 on a 50-case educational video benchmark, representing an 11-point precision improvement over embedding-only baselines. The job-keyed caching architecture delivers cached query latency of under 0.8 seconds regardless of video length. Additional modules for query-driven privacy blurring (94.3% face recall, 96.1% precision via YOLOv8) and codec-aware compression (72–78% file size reduction via H.265/HEVC) complete a comprehensive, end-to-end video processing pipeline deployable on standard consumer hardware.

NeuroClip demonstrates that semantically rich, privacy-aware, and computationally efficient video retrieval is achievable with available open-source AI tooling. The system has direct applications in educational content navigation, corporate knowledge management, media production, and privacy-compliant video analytics.

---

## References

[1] D. Vora, P. Kadam, D. D. Mohite, N. Kumar, N. Kumar, P. Radhakrishnan, and S. Bhagwat, "AI-driven video summarization for optimizing content retrieval and management through deep learning techniques," 2025.

[2] N. Reimers and I. Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks," in *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, 2019.

[3] R. Nogueira and K. Cho, "Passage Re-ranking with BERT," *arXiv preprint arXiv:1901.04085*, 2019.

[4] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, "Attention Is All You Need," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2017.

[5] E. Apostolidis, E. Adamantidou, A. I. Metsai, V. Mezaris, and I. Patras, "Video Summarization Using Deep Neural Networks: A Survey," *Proceedings of the IEEE*, vol. 109, pp. 1838–1863, 2021.

[6] R. Plaud and J.-L. Lisani, "Two Deep Learning Solutions for Automatic Blurring of Faces in Videos," 2024.

[7] A. Nemavhola, S. Viriri, and C. Chibaya, "A Scoping Review of Literature on Deep Learning Techniques for Face Recognition," 2025.

[8] G. J. Sullivan and T. Wiegand, "Video Compression — From Concepts to the H.264/AVC Standard," *Proceedings of the IEEE*, vol. 93, no. 1, pp. 18–31, 2005.

[9] P. Saini, K. Kumar, S. Kashid, A. Saini, and A. Negi, "Video summarization using deep learning techniques: a detailed analysis and investigation," pp. 12347–12385, 2023.

[10] P. Kadam, D. Vora, S. Mishra, S. Patil, K. Kotecha, and A. Abraham, "Recent Challenges and Opportunities in Video Summarization With Machine Learning Algorithms," *IEEE Access*, vol. 10, pp. 122762–122785, 2022.

[11] "CapCut - Video Editor," Bytedance Pte. Ltd., 2024. [Online]. Available: https://play.google.com/store/apps/details?id=com.lemon.lvoverseas

[12] "Notta - Dictation & Transcription," Notta Pte. Ltd., 2025. [Online]. Available: https://play.google.com/store/apps/details?id=com.langogo.transcribe

---

*Submitted for the partial fulfillment of the BS Computer Science degree, Faculty of Engineering & Computing, National University of Modern Languages, Faisalabad. December 2025.*
