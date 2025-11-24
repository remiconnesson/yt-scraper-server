# Slide Extraction Pipeline Testing Guide

This document describes how to run and test the slide extraction pipeline.

## Overview

The slide extraction pipeline has three main components:

1. **Video Analysis** - Detects static segments (slides) vs moving segments in videos
2. **Frame Extraction** - Extracts representative frames from static segments
3. **S3 Upload** - Uploads extracted frames to S3 with metadata

## Prerequisites

1. Install dependencies:
   ```bash
   uv sync --dev
   ```

2. Configure environment variables in `.env`:
   ```bash
   S3_ENDPOINT=https://s3.remtoolz.ai
   S3_ACCESS_KEY=your_access_key
   S3_BUCKET_NAME=slides-extractor  # optional, defaults to slides-extractor
   ZYTE_API_KEY=your_zyte_api_key   # required for YouTube downloads
   ```

## Testing Methods

### Method 1: Synthetic Video Test (Recommended for Quick Testing)

Run the test script that creates a synthetic video with static slides:

```bash
# Analysis only (no S3 upload)
uv run python test_pipeline.py

# Full pipeline with S3 upload
uv run python test_pipeline.py --with-s3
```

**What it does:**
- Creates a synthetic video with alternating static slides and motion segments
- Analyzes the video to detect segments
- (Optional) Uploads static frame images to S3
- Displays detailed results

**Example output:**
```
================================================================================
Analysis Results:
================================================================================
Total segments detected: 2
Static segments: 1
Moving segments: 1
Video duration: 9.00 seconds
Total frames: 10

Segment 1:
  Type: static
  Start time: 0.00s
  End time: 1.00s
  Duration: 1.00s
  Frame count: 2
  Has representative frame: True

Segment 2:
  Type: moving
  Start time: 2.00s
  End time: 9.00s
  Duration: 7.00s
  Frame count: 8
  Has representative frame: False
```

### Method 2: YouTube Video Test

Process a real YouTube video:

```bash
uv run python test_youtube.py <youtube_video_id>

# Example:
uv run python test_youtube.py dQw4w9WgXcQ
```

**What it does:**
- Downloads the video from YouTube (requires ZYTE_API_KEY)
- Analyzes the video for static segments
- Extracts representative frames
- Uploads frames to S3
- Tracks progress via job tracking system

### Method 3: API Server (Production Method)

Start the FastAPI server:

```bash
uv run uvicorn slides_extractor.app_factory:app --host 0.0.0.0 --port 8000
```

Then make API requests:

```bash
# Start a video processing job
curl -X POST "http://localhost:8000/process/youtube/dQw4w9WgXcQ"

# Check progress
curl "http://localhost:8000/progress"

# View logs
curl "http://localhost:8000/logs"
```

## Pipeline Architecture

### 1. Video Analysis Phase

**File:** `src/slides_extractor/extract_slides/video_analyzer.py`

- **FrameStreamer**: Streams video frames with perceptual hashing
- **SegmentDetector**: Detects static vs moving segments based on hash similarity
- **analyze_video()**: Main entry point for analysis

**Configuration:**
```python
config: AnalysisConfig = {
    "grid_cols": 4,              # Hash grid columns
    "grid_rows": 4,              # Hash grid rows
    "cell_hash_threshold": 5,    # Hash difference threshold
    "min_static_cell_ratio": 0.8,# Minimum ratio of static cells
    "min_static_frames": 3,      # Minimum consecutive frames for static segment
}
```

### 2. Frame Extraction & Upload Phase

**File:** `src/slides_extractor/video_service.py`

Key functions:
- `_detect_static_segments()`: Analyzes video and updates job progress
- `_upload_segments()`: Uploads frames to S3 with metadata
- `extract_and_process_frames()`: Orchestrates the full pipeline

**S3 Storage Structure:**
```
video/
  {video_id}/
    images/
      segment_001.png
      segment_002.png
      ...
```

### 3. Job Tracking

**File:** `src/slides_extractor/video_jobs.py`

- Manages background processing of videos
- Downloads video from YouTube (via yt-dlp + Zyte proxy)
- Runs analysis and upload pipeline
- Tracks job status and progress

## Output Format

### Segment Metadata

Each uploaded segment includes:

```json
{
  "segment_id": 1,
  "start_time": 0.0,
  "end_time": 5.0,
  "duration": 5.0,
  "frame_count": 10,
  "image_url": "https://s3.remtoolz.ai/slides-extractor/video/VIDEO_ID/images/segment_001.png"
}
```

### Job Status

Jobs track progress through these states:
- `pending`: Job created, not started
- `downloading`: Downloading video from source
- `extracting`: Analyzing video for segments
- `uploading`: Uploading frames to S3
- `completed`: All steps finished
- `failed`: Error occurred

## Troubleshooting

### Issue: "S3 configuration missing"

**Solution:** Ensure `S3_ENDPOINT` and `S3_ACCESS_KEY` are set in `.env`:
```bash
S3_ENDPOINT=https://s3.remtoolz.ai
S3_ACCESS_KEY=your_key_here
```

### Issue: "Video file not found"

**Solution:** Verify the video file path exists and is accessible

### Issue: YouTube download fails

**Solution:** 
- Verify `ZYTE_API_KEY` is set correctly
- Check that the YouTube video ID is valid
- Ensure yt-dlp can access the video

### Issue: No static segments detected

**Possible causes:**
- Video has continuous motion (no static slides)
- Detection thresholds are too strict

**Solution:** Adjust analysis config:
```python
config: AnalysisConfig = {
    "cell_hash_threshold": 8,      # Increase for more lenient matching
    "min_static_cell_ratio": 0.6,  # Decrease to require fewer static cells
    "min_static_frames": 2,        # Decrease to detect shorter pauses
}
```

## Running Tests

Run the test suite:

```bash
# All tests
uv run pytest

# Specific test file
uv run pytest tests/test_video_analyzer.py

# Integration tests (requires S3 config)
uv run pytest -m integration
```

## Development

### Type Checking
```bash
uv run ty
```

### Linting
```bash
uv run ruff check .
```

### Formatting
```bash
uv run ruff format .
```

### All checks
```bash
make all
```

