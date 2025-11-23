# Repository Scope Document

## Project Name
**VPS Video Processing Service**

## Purpose
This repository contains a Python-based video processing service designed to run on a VPS (16GB RAM, K3S cluster). It handles the computationally intensive tasks of video analysis and frame extraction for a larger video knowledge base system.

## What This Service Does

### Core Responsibilities
1. **Video Acquisition**
   - Downloads videos from YouTube (via yt-dlp)
   - Fetches videos from S3 storage
   - Temporarily stores videos for processing

2. **Intelligent Frame Extraction**
   - Analyzes videos using perceptual hashing (grid-based)
   - Detects static segments (slides, stable frames)
   - Identifies moving sections
   - Extracts representative frames from static segments

3. **Smart Image Compression**
   - Preserves original frame dimensions (no resizing)
   - Applies content-aware compression (text vs images)
   - Uses WebP format for optimal size/quality ratio
   - Reduces file size by ~70% vs standard JPEG

4. **S3 Storage Management**
   - Streams compressed images directly to S3 (no local storage bloat)
   - Organizes files in structured hierarchy: `video/{id}/images/`
   - Generates metadata JSON with timestamps and URLs
   - Handles deduplication (checks if video already processed)

5. **Progress Streaming**
   - Provides real-time progress via Server-Sent Events (SSE)
   - Enables reconnection on network failures
   - Supports job persistence with UUID-based tracking

## What This Service Does NOT Do

❌ **AI Processing** - No chapter generation, OCR, or LLM analysis (handled by Vercel)  
❌ **Frontend UI** - No web interface (handled by Vercel Next.js)  
❌ **Database Storage** - No video metadata persistence (handled by Vercel/Neon)  
❌ **RAG Pipeline** - No vector indexing or chat features (handled by Vercel/ChromaDB)  
❌ **Authentication** - No user management (handled by Vercel/Neon Auth)

## Architecture Position

```
User → Vercel (UI + Workflows) → [THIS SERVICE] → S3 Storage
                                       ↓
                                  Metadata JSON
                                       ↓
                          Vercel (AI Processing + RAG)
```

**This service is the "heavy lifter":**
- Runs on VPS with dedicated compute resources
- Handles video downloads and processing
- Offloads Vercel from expensive video operations
- Provides clean handoff via metadata JSON

## Technology Stack

### Core Dependencies
- **FastAPI** - Web framework for REST API + SSE
- **yt-dlp** - YouTube video downloading
- **OpenCV** - Video frame extraction and processing
- **Pillow** - Image compression (WebP)
- **boto3** - S3 client for o2switch
- **imagehash** - Perceptual hashing for frame comparison

### Integrated Packages
- **extract-slides-py** - Local package providing video analysis algorithms
  - FrameStreamer: Memory-efficient frame processing
  - SegmentDetector: Static segment detection state machine

## Key Features

### 1. Memory Efficiency
- Streams frames one-by-one (no RAM accumulation)
- Deletes temporary video files after processing
- Direct S3 upload (no intermediate local storage)

### 2. Content-Aware Processing
- Detects text/slides via edge density analysis
- Adjusts compression quality based on content type
- Preserves text readability with higher quality

### 3. Reconnection Support
- UUID-based job tracking
- SSE streams can reconnect on network failure
- Job state persists during processing

### 4. Production Ready
- Systemd service configuration
- Health check endpoints
- Comprehensive error handling and logging
- Environment-based configuration

## API Endpoints

### Processing
- `POST /process/youtube/{video_id}` - Process YouTube video
- `POST /process/s3` - Process video from S3

### Monitoring
- `GET /job/{job_id}/stream` - SSE progress stream
- `GET /job/{job_id}/result` - Final result (non-streaming)
- `GET /health` - Service health check

## Output Structure

### S3 Storage
```
video/
  {video_id}/
    images/
      segment_001.webp
      segment_002.webp
      segment_003.webp
    metadata.json
```

### Metadata JSON Format
```json
{
  "video_id": "dQw4w9WgXcQ",
  "source": "https://youtube.com/watch?v=...",
  "processed_at": "2024-11-23T10:30:00Z",
  "total_frames": 180,
  "static_segments": 42,
  "compression_stats": {
    "format": "webp",
    "frames_with_text": 15,
    "avg_quality": 87.5
  },
  "segments": [
    {
      "segment_id": 1,
      "start_time": 0.0,
      "end_time": 5.3,
      "duration": 5.3,
      "frame_count": 5,
      "image_url": "https://s3.../segment_001.webp",
      "compression": {
        "has_text": true,
        "quality": 90,
        "edge_density": 0.18
      }
    }
  ]
}
```

## Deployment Target

**Hardware Requirements:**
- VPS with 16GB RAM
- K3S cluster support
- 200GB disk space (temporary video storage)
- 10-20TB monthly egress (for S3 uploads)

**Network Requirements:**
- Outbound HTTPS (YouTube, S3)
- Inbound HTTPS (API access from Vercel)
- SSE/streaming support (no aggressive buffering)

**S3 Requirements:**
- S3-compatible storage (o2switch)
- Unlimited storage capacity
- Public read access for uploaded images

## Integration Points

### Upstream (Vercel Workflows)
**Vercel triggers this service:**
1. POST to `/process/youtube/{id}`
2. Connect to SSE stream for progress
3. Poll `/result` endpoint for completion
4. Fetch metadata.json from S3

### Downstream (S3 Storage)
**This service outputs to S3:**
1. Compressed WebP images
2. Metadata JSON with all segment info
3. Vercel reads from S3 for AI processing

## Development Workflow

```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure
cp .env.template .env
# Edit .env with S3 credentials

# Run
python video_service.py

# Test
curl -X POST http://localhost:8001/process/youtube/dQw4w9WgXcQ
curl -N http://localhost:8001/job/{job_id}/stream
```

## Repo Structure

```
.
├── video_service.py           # Main FastAPI service
├── requirements.txt           # Python dependencies
├── .env.template             # Environment configuration template
├── extract-slides-py/        # Video analysis package
│   └── src/
│       └── extract_slides/
│           ├── video_analyzer.py    # Frame extraction logic
│           └── video_output.py      # Output formatting
├── deploy.sh                 # VPS deployment script
├── systemd/                  # Service configuration
│   └── video-processor.service
└── README.md                 # Setup and usage docs
```

## Success Criteria

A successful deployment means:
1. ✅ Videos are downloaded and processed without manual intervention
2. ✅ Frames are extracted with 95%+ accuracy in detecting static segments
3. ✅ Images are compressed to ~30% original size while maintaining visual quality
4. ✅ All outputs land in S3 with correct structure
5. ✅ SSE streams provide real-time progress updates
6. ✅ Service auto-recovers from transient failures
7. ✅ Vercel can reliably consume the metadata JSON

## Out of Scope (Other Repos)

- **vercel-workflows-repo**: AI chapter generation, RAG indexing, UI
- **frontend-repo**: User interface, video player, chat interface
- **database-schema**: Video metadata, user data, conversations

This service is a **single-purpose microservice** focused exclusively on the compute-intensive video preprocessing pipeline.
