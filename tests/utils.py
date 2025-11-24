import cv2
import numpy as np


def create_test_video(
    output_path: str, duration_seconds: int = 10, fps: int = 5
) -> None:
    """Create a synthetic test video with alternating static slides and motion."""
    # Video properties
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore # cv2.VideoWriter_fourcc is dynamically bound
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        # Try fallback codec
        print("Failed to open video writer with mp4v, trying avc1")
        fourcc = cv2.VideoWriter_fourcc(*"avc1")  # type: ignore # cv2.VideoWriter_fourcc is dynamically bound
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not out.isOpened():
            raise RuntimeError(f"Failed to create video writer at {output_path}")

    total_frames = duration_seconds * fps

    # Create patterns: static slide (red), motion (random), static slide (blue), motion (random), static slide (green)
    for frame_num in range(total_frames):
        # Create a blank frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Divide video into segments
        segment = frame_num // (total_frames // 5)

        if segment == 0:  # Red static slide
            frame[:] = (0, 0, 255)  # BGR format: Red
            cv2.putText(
                frame,
                "Slide 1",
                (width // 2 - 100, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (255, 255, 255),
                3,
            )
        elif segment == 1:  # Random motion
            frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        elif segment == 2:  # Blue static slide
            frame[:] = (255, 0, 0)  # BGR format: Blue
            cv2.putText(
                frame,
                "Slide 2",
                (width // 2 - 100, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (255, 255, 255),
                3,
            )
        elif segment == 3:  # Random motion
            frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        else:  # Green static slide
            frame[:] = (0, 255, 0)  # BGR format: Green
            cv2.putText(
                frame,
                "Slide 3",
                (width // 2 - 100, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (255, 255, 255),
                3,
            )

        out.write(frame)

    out.release()
