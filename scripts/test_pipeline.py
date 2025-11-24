#!/usr/bin/env python3
"""Test script to run the slide extraction pipeline on a synthetic video."""

import asyncio
import cv2
import numpy as np
import tempfile
import os

from slides_extractor.extract_slides.video_analyzer import analyze_video, AnalysisConfig
from slides_extractor.video_service import extract_and_process_frames


def create_test_video(output_path: str, duration_seconds: int = 10, fps: int = 5) -> None:
    """Create a synthetic test video with alternating static slides and motion.
    
    Args:
        output_path: Path where the video file will be saved
        duration_seconds: Total duration of the video in seconds
        fps: Frames per second for the video
    """
    print(f"Creating test video at {output_path}...")
    
    # Video properties
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = duration_seconds * fps
    
    # Create patterns: static slide (red), motion (random), static slide (blue), motion (random), static slide (green)
    for frame_num in range(total_frames):
        # Create a blank frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Divide video into segments
        segment = frame_num // (total_frames // 5)
        
        if segment == 0:  # Red static slide
            frame[:] = (0, 0, 255)  # BGR format: Red
            # Add some text
            cv2.putText(frame, "Slide 1", (width//2 - 100, height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        elif segment == 1:  # Random motion
            frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        elif segment == 2:  # Blue static slide
            frame[:] = (255, 0, 0)  # BGR format: Blue
            cv2.putText(frame, "Slide 2", (width//2 - 100, height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        elif segment == 3:  # Random motion
            frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        else:  # Green static slide
            frame[:] = (0, 255, 0)  # BGR format: Green
            cv2.putText(frame, "Slide 3", (width//2 - 100, height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
        out.write(frame)
    
    out.release()
    print(f"Test video created with {total_frames} frames at {fps} fps")


def run_analyze_video_only(video_path: str) -> None:
    """Run just the analysis part without S3 upload."""
    print("\n" + "="*80)
    print("Testing video analysis (without S3 upload)...")
    print("="*80 + "\n")
    
    config: AnalysisConfig = {
        "grid_cols": 4,
        "grid_rows": 4,
        "cell_hash_threshold": 5,
        "min_static_cell_ratio": 0.8,
        "min_static_frames": 3,
    }
    
    result = analyze_video(video_path, config, verbose=True)
    
    print(f"\n{'='*80}")
    print("Analysis Results:")
    print(f"{'='*80}")
    print(f"Total segments detected: {len(result.segments)}")
    print(f"Static segments: {len(result.static_segments)}")
    print(f"Moving segments: {len(result.moving_segments)}")
    print(f"Video duration: {result.video_duration:.2f} seconds")
    print(f"Total frames: {result.total_frames}")
    
    print(f"\n{'='*80}")
    print("Segment Details:")
    print(f"{'='*80}")
    for i, segment in enumerate(result.segments, 1):
        print(f"\nSegment {i}:")
        print(f"  Type: {segment.type}")
        print(f"  Start time: {segment.start_time:.2f}s")
        print(f"  End time: {segment.end_time:.2f}s")
        print(f"  Duration: {segment.duration:.2f}s")
        print(f"  Frame count: {segment.frame_count}")
        print(f"  Has representative frame: {segment.representative_frame is not None}")


async def run_full_pipeline(video_path: str) -> None:
    """Run the full pipeline including S3 upload."""
    print("\n" + "="*80)
    print("Testing full pipeline (with S3 upload)...")
    print("="*80 + "\n")
    
    video_id = "test_video_001"
    job_id = "test_job_001"
    
    try:
        metadata = await extract_and_process_frames(video_path, video_id, job_id)
        
        print(f"\n{'='*80}")
        print("Pipeline Results:")
        print(f"{'='*80}")
        print(f"Segments uploaded: {len(metadata)}")
        
        for i, segment_meta in enumerate(metadata, 1):
            print(f"\nSegment {i}:")
            print(f"  Segment ID: {segment_meta.get('segment_id')}")
            print(f"  Start time: {segment_meta.get('start_time'):.2f}s")
            print(f"  End time: {segment_meta.get('end_time'):.2f}s")
            print(f"  Duration: {segment_meta.get('duration'):.2f}s")
            print(f"  Frame count: {segment_meta.get('frame_count')}")
            print(f"  Image URL: {segment_meta.get('image_url')}")
    
    except Exception as e:
        print(f"\nError in full pipeline: {e}")
        print("This is expected if S3 credentials are not properly configured.")
        raise


def main(run_s3_upload: bool = False):
    """Main function to run the pipeline test.
    
    Args:
        run_s3_upload: If True, runs the full pipeline including S3 upload
    """
    print("="*80)
    print("Slide Extraction Pipeline Test")
    print("="*80)
    
    # Create temporary directory for test video
    temp_dir = tempfile.mkdtemp(prefix="slides_test_")
    video_path = os.path.join(temp_dir, "test_video.mp4")
    
    try:
        # Create test video
        create_test_video(video_path, duration_seconds=10, fps=5)
        
        # Test 1: Analyze video without S3 upload
        run_analyze_video_only(video_path)
        
        # Test 2: Full pipeline with S3 upload
        if run_s3_upload:
            print("\n" + "="*80)
            print("Running full pipeline with S3 upload...")
            print("="*80)
            asyncio.run(run_full_pipeline(video_path))
        else:
            print("\n" + "="*80)
            print("Skipping S3 upload test.")
            print("To test S3 upload, run: python test_pipeline.py --with-s3")
            print("="*80)
        
        print("\n" + "="*80)
        print("Test completed successfully!")
        print("="*80)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        if os.path.exists(video_path):
            os.remove(video_path)
        try:
            os.rmdir(temp_dir)
        except OSError:
            pass
        print(f"\nCleaned up temporary files in {temp_dir}")


if __name__ == "__main__":
    import sys
    run_s3 = "--with-s3" in sys.argv
    main(run_s3_upload=run_s3)

