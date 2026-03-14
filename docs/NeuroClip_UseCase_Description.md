NeuroClip Use Case Diagram — Description and Guidelines

NeuroClip’s use case diagram models how the End User interacts with the system to upload videos, derive searchable knowledge from transcripts and embeddings, and retrieve context-aware clips. The diagram, as shown in Figure 4.1, presents the system boundary for "NeuroClip" with its frontend UI and backend services, and emphasizes user-centric functionality: uploading media, generating transcripts, computing and storing embeddings, searching for relevant segments, and viewing previously saved clips. This view is essential because it depicts the functionalities performed with direct user interaction and helps see the system from the user’s perspective (as shown in Figure 4.1).

Actors
- End User: interacts with the UI to upload videos, submit queries, review results, and manage saved clips.
- NeuroClip System (Frontend): presents forms, query boxes, and playback components; forwards actions to backend.
- Backend Service (FastAPI): processes uploads, generates transcripts, computes/stores embeddings, performs search.
- Supabase (Database/Storage): persists video metadata, embeddings, and processing history; serves stored assets.
- Admin (optional): oversees content, reviews logs, and manages configuration and policies.

Primary Use Cases
- Upload Video: user provides a video; the system accepts and stores it.
- Generate Transcript: backend converts audio to text and segments sentences with timing.
- Compute and Store Embeddings: sentence and video-level vectors persisted for later retrieval.
- Search Clips by Query: user submits a query; backend ranks segments and returns playable clips.
- View Past Clips: user sees previous results linked by job_id; can replay and reuse saved context.
- Save Processing History: the system records user actions and outcomes with job_id linkage.
- Play Retrieved Clips: the player loads clip URLs and plays merged contextual segments.

Secondary Use Cases
- Manage Videos: list, filter, and remove uploaded items.
- Manage Account and Authentication: login, logout, and profile management.
- Error Handling and Notifications: show upload/search errors and success toasts.

General Guidelines
- Use-Case diagram can be skipped only if there is absolutely no user interaction with the system.
- Diagrams should be clear and uncluttered; preferred tools to draw the diagrams are Rational Rose and MS Visio.
- Caption styles: use Arial Narrow, size 10. Provide table titles at the top and figure titles below the figure.
- Number figures and tables with chapter number as prefix (e.g., 4.1, 4.2, 4.3) and reference them consistently in text.
- In-text references should use the syntax "as shown in Figure 4.1" when describing the diagram.
- Ensure alignment between the diagram’s actors/use cases and the features listed in the text.

Figure Caption Template
Figure 4.1: NeuroClip Use Case Diagram (caption style: Arial Narrow, size 10; place below the figure)

4.3 Full-Dress Use Case
Primary Actor: User
Preconditions: Verified session; video accessible; storage quota available.
Main Flow:
1. User selects a video for analysis on the Upload page.
2. System validates file type/size and stores the video.
3. System generates transcript and sentence timings.
4. System computes and stores sentence/video embeddings linked by job_id.
5. User opens the Video page and enters a query.
6. System searches embeddings, ranks segments, and assembles clips.
7. System saves processing history with job_id and returns playable clip URLs.
Alternatives:
* A1: Invalid file → prompt correction and show accepted formats.
* A2: Transcript missing → fallback to metadata-only search; prompt reprocessing.
Postconditions: Upload, transcript, embeddings, and personalized search results saved.

4.4 Activity Diagram (text description)
Start → Registration → Email Verification → Login → Onboarding → Upload Video → Generate Transcript → Compute Embeddings → Search Clips → View History → End
Decisions:
* Verified? If no, resend email; if yes, proceed.
* Transcript generated? If no, re-queue processing; if yes, compute embeddings.
* Clips found? If yes, display results and save history; if no, suggest broader query.

4.5 Data Flow Diagram (DFD 0, 1, 2)
DFD Level 0 (Context):
* Entities: User, Admin.
* System: NeuroClip.
* Data Stores: Videos, Transcripts, Embeddings, Processing History.
* Flows: Uploads, queries/responses, clip assets, analytics.

DFD Level 1 (Processes):
* P1: Auth & Profile Management → Profiles.
* P2: Video Ingestion → Videos + Transcripts.
* P3: Embedding Computation → Embeddings (sentence/video-level).
* P4: Clip Search & Assembly → Clip Assets + History.
* P5: Admin Content & Ingestion → Knowledge + Vectors.

DFD Level 2 (Clip Search & Assembly):
* P4.1: Intent Classification (optional).
* P4.2: Retrieval (semantic with keyword boost).
* P4.3: Reranking and Neighbor Merging.
* P4.4: Clip Boundary Assembly and URL Generation.
* P4.5: History Logging.

4.6 System Sequence Diagram (text description)
Scenario: Clip search flow.
User → UI: enter query.
UI → API: POST `/clips/search-db`.
API → SearchService: process_query(job_id, params).
SearchService → VectorStore: search top-k.
SearchService → Storage: resolve clip URLs.
SearchService → API: response payload.
API → UI: JSON response.
UI: render clips; persist search in history.

4.7 Sequence Diagram (text description)
Scenario: Email verification redirect.
User → Auth: signUp(email, password).
Auth → Email Service: send link with redirect to production domain.
User → Email: click link.
Email Service → Auth: verify token.
Auth → Frontend Redirect: domain root or callback.
Frontend: establish session; route to onboarding/dashboard.

4.8 Design Class Diagram (text description)
Entities:
* User(id, email)
* Video(id, title, video_url, duration, created_at)
* TranscriptSegment(id, video_id, start, end, text)
* EmbeddingRow(id, video_id, job_id, type, vector, created_at)
* ProcessingHistory(id, user_id, job_id, module, query, status, created_at)
* ClipRequest(id, job_id, query, params)
* ClipResult(id, job_id, start, end, url)
Services:
* Transcriber(generate_transcript)
* EmbeddingService(encode_sentences, store_vectors)
* SearchService(process_query, merge_neighbors, rerank)
* HistoryService(record_event)
* StorageService(save_video, get_clip_url)
Associations:
* User 1–N Video; Video 1–N TranscriptSegment; Video 1–N EmbeddingRow.
* ProcessingHistory N–1 User and references job_id.
* ClipRequest 1–N ClipResult; SearchService uses EmbeddingService and StorageService.

4.9 Architectural Diagrams
4.9.1 Interface Design
* Pages: Register, Login, Onboarding, History, VideoClips, Summarization, Admin.
* Components: Header, BottomNav, Cards, Inputs, Player, Toasts.

4.9.2 Component Level Design
* Frontend: modular React components with Supabase client; routes for `/video/:id` and history.
* Backend: FastAPI routes with Pydantic models; endpoints for upload, embeddings, and search.
* Pipeline: transcription utilities, embedding computation, vector storage (pgvector), clip generation.

4.9.3 Deployment Design
* Frontend on Vercel with `VITE_SITE_URL` and Supabase env vars.
* Supabase as Auth/DB/Storage; RLS enabled.
* Backend service (dev/local or cloud) for `/api` endpoints; supports hot-reload of vectors.
* Figure references should follow chapter numbering (e.g., Figure 4.2, 4.3) and use Arial Narrow size 10 captions below figures.
