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

# Create a non-root user for security and prepare the work directory
RUN useradd -m appuser && \
    mkdir -p /app/models && \
    chown -R appuser:appuser /app

# Switch to the non-root user before copying files/installing dependencies
USER appuser

# Set working directory
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

# Download the EAST text detection model with the correct ownership
ADD --chown=appuser:appuser https://github.com/oyyd/frozen_east_text_detection.pb/raw/master/frozen_east_text_detection.pb /app/models/frozen_east_text_detection.pb

# Copy dependency files
COPY --chown=appuser:appuser pyproject.toml uv.lock ./

# Install dependencies
# Use --frozen to ensure exact versions from uv.lock
# Use --no-dev to skip development dependencies
# Use --no-install-project to skip installing the project itself (which requires README.md)
RUN uv sync --frozen --no-dev --no-install-project

# Copy source code
COPY --chown=appuser:appuser src ./src
COPY --chown=appuser:appuser README.md ./

# Install the project itself (if it has a package structure)
RUN uv sync --frozen --no-dev

# Expose the port the app runs on
EXPOSE 8000

# Run the application directly from the virtual environment
CMD ["uvicorn", "slides_extractor.app_factory:app", "--host", "0.0.0.0", "--port", "8000"]

