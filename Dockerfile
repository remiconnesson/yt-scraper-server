# Use a Python image with uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

# Install system dependencies required for OpenCV and other tools
# - libglib2.0-0, libsm6, libxext6, libxrender-dev: Required by opencv-python-headless
# - ffmpeg: Required by yt-dlp for merging video/audio and format conversion
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

RUN mkdir -p /app/models

# Download the EAST text detection model
# Using the raw GitHub link for direct access to the .pb file
ADD https://github.com/oyyd/frozen_east_text_detection.pb/raw/master/frozen_east_text_detection.pb /app/models/frozen_east_text_detection.pb

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
# Use --frozen to ensure exact versions from uv.lock
# Use --no-dev to skip development dependencies
# Use --no-install-project to skip installing the project itself (which requires README.md)
RUN uv sync --frozen --no-dev --no-install-project

# Copy source code
COPY src ./src
COPY README.md ./

# Install the project itself (if it has a package structure)
RUN uv sync --frozen --no-dev

# Create a non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Expose the port the app runs on
EXPOSE 8000

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

# Run the application directly from the virtual environment
CMD ["uvicorn", "slides_extractor.app_factory:app", "--host", "0.0.0.0", "--port", "8000"]

