// ================================
// 🚀 NeuroClip Backend Server
// ================================
import express from "express";
import cors from "cors";
import fs from "fs";
import path from "path";
import ffmpeg from "fluent-ffmpeg";
import { v4 as uuidv4 } from "uuid";
import { AssemblyAI } from "assemblyai";
import YTDlpWrap from "yt-dlp-wrap";
import dotenv from "dotenv";

dotenv.config();
const app = express();
const PORT = process.env.PORT || 5000;

// ================================
// ⚙️ Middleware Setup
// ================================
app.use(cors());
app.use(express.json());

// ================================
// 📁 Directories Setup
// ================================
const __dirname = path.resolve();
const tempDir = path.join(__dirname, "temp");

if (!fs.existsSync(tempDir)) {
  fs.mkdirSync(tempDir);
}

// ================================
// 🧠 Initialize Services
// ================================
const ytdlp = new YTDlpWrap();
const assembly = new AssemblyAI({
  apiKey: process.env.ASSEMBLYAI_API_KEY || "YOUR_ASSEMBLYAI_API_KEY",
});

// ================================
// 🧩 Utility: Sleep Helper
// ================================
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// ================================
// 🎥 Route: Process YouTube Video
// ================================
app.post("/api/process-youtube", async (req, res) => {
  const { url, startTime = 0, endTime = 10 } = req.body;

  if (!url || (!url.includes("youtube.com") && !url.includes("youtu.be"))) {
    return res.status(400).json({ error: "Invalid YouTube URL" });
  }

  const jobId = uuidv4();
  const jobDir = path.join(tempDir, jobId);
  fs.mkdirSync(jobDir);

  try {
    console.log("🎬 Getting video info...");
    const info = await ytdlp.getVideoInfo(url);
    const videoTitle = info.title.replace(/[^\w\s]/gi, "").trim() || "video";
    const audioPath = path.join(jobDir, `${videoTitle}.mp3`);
    const framesDir = path.join(jobDir, "frames");
    fs.mkdirSync(framesDir);

    // ================================
    // 🎧 Step 1: Download Audio
    // ================================
    console.log("⬇️ Downloading audio...");
    await ytdlp.execPromise([
      url,
      "-x",
      "--audio-format",
      "mp3",
      "--audio-quality",
      "0",
      "-o",
      audioPath,
    ]);

    // ================================
    // 🖼️ Step 2: Extract Frames
    // ================================
    console.log("🖼️ Extracting frames...");
    await new Promise((resolve, reject) => {
      ffmpeg(url)
        .setStartTime(startTime)
        .setDuration(endTime - startTime)
        .outputOptions("-vf", "fps=1,scale=640:480")
        .output(path.join(framesDir, "frame-%03d.png"))
        .on("end", resolve)
        .on("error", reject)
        .run();
    });

    // ================================
    // 🧠 Step 3: Transcription (AssemblyAI)
    // ================================
    console.log("🧠 Uploading audio to AssemblyAI...");
    let transcriptText = "Transcript not available";
    let transcriptId = "";

    try {
      const uploadUrl = await assembly.files.upload(audioPath);
      const transcript = await assembly.transcripts.create({
        audio_url: uploadUrl,
        language_code: "en_us",
        punctuate: true,
        format_text: true,
        speaker_labels: true,
      });

      transcriptId = transcript.id;
      console.log("⏳ Waiting for transcription...");

      let result = await assembly.transcripts.get(transcriptId);
      while (result.status !== "completed" && result.status !== "error") {
        await sleep(5000);
        result = await assembly.transcripts.get(transcriptId);
      }

      if (result.status === "completed") {
        transcriptText = result.text;
        console.log("✅ Transcription completed");
      } else {
        console.error("❌ Transcription failed:", result.error);
      }
    } catch (err) {
      console.error("Transcription error:", err.message);
    }

    // ================================
    // 📤 Step 4: Send Response
    // ================================
    res.json({
      jobId,
      video: {
        title: info.title,
        duration: info.duration,
        thumbnails: info.thumbnails,
      },
      transcript: transcriptText,
      transcriptId,
      audioUrl: `/api/audio/${jobId}/${encodeURIComponent(videoTitle)}.mp3`,
      framesDir: `/api/frames/${jobId}`,
    });
  } catch (error) {
    console.error("❌ Processing error:", error);
    res.status(500).json({ error: "Failed to process video: " + error.message });
  }
});

// ================================
// 🖼️ Serve Extracted Frames
// ================================
app.get("/api/frames/:jobId/:filename", (req, res) => {
  const { jobId, filename } = req.params;
  const filePath = path.join(tempDir, jobId, "frames", filename);

  if (!fs.existsSync(filePath)) {
    return res.status(404).send("Frame not found");
  }

  res.sendFile(filePath);
});

// ================================
// 🎧 Serve Audio File
// ================================
app.get("/api/audio/:jobId/:filename", (req, res) => {
  const { jobId, filename } = req.params;
  const filePath = path.join(tempDir, jobId, decodeURIComponent(filename));

  if (!fs.existsSync(filePath)) {
    return res.status(404).send("Audio file not found");
  }

  res.sendFile(filePath);
});

// ================================
// 🧩 Get Basic Video Info Only
// ================================
app.get("/api/video-info", async (req, res) => {
  const { url } = req.query;
  try {
    const info = await ytdlp.getVideoInfo(url);
    res.json({
      title: info.title,
      duration: info.duration,
      thumbnails: info.thumbnails,
    });
  } catch (error) {
    res.status(500).json({ error: "Failed to get video info: " + error.message });
  }
});

// ================================
// ⚠️ Global Error Handler
// ================================
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({ error: "Something went wrong!" });
});

// ================================
// 🚀 Start Server
// ================================
app.listen(PORT, () => {
  console.log(`✅ NeuroClip server running on port ${PORT}`);
});
