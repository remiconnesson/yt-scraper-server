#!/usr/bin/env python3
"""Test script to run the slide extraction pipeline on a YouTube video."""

import sys

from slides_extractor.video_jobs import process_video_task


def main():
    """Run the pipeline on a YouTube video."""
    if len(sys.argv) < 2:
        print("Usage: python test_youtube.py <youtube_video_id>")
        print("Example: python test_youtube.py dQw4w9WgXcQ")
        sys.exit(1)
    
    video_id = sys.argv[1]
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    job_id = f"test_job_{video_id}"
    
    print("="*80)
    print(f"Processing YouTube Video: {video_id}")
    print("="*80)
    print(f"URL: {video_url}")
    print(f"Job ID: {job_id}")
    print("="*80)
    
    try:
        process_video_task(video_url, video_id, job_id)
        print("\n" + "="*80)
        print("Pipeline completed successfully!")
        print("="*80)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

