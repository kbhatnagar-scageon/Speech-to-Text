# Build stage
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime AS builder

# Set working directory
WORKDIR /build

# Copy requirements file
COPY requirements.txt .

# Install system dependencies and PyInstaller
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies and PyInstaller
RUN pip install --no-cache-dir -r requirements.txt pyinstaller

# Copy application code
COPY sst_api.py whisper_transcriber.py ./

# Create PyInstaller spec file
RUN pyinstaller --name app \
    --onefile \
    --hidden-import=torch \
    --hidden-import=transformers \
    --hidden-import=librosa \
    --hidden-import=uvicorn \
    --hidden-import=fastapi \
    --add-data "whisper_transcriber.py:." \
    sst_api.py

# Runtime stage
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies for running the compiled app
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy Python environment from builder
COPY --from=builder /opt/conda /opt/conda

# Copy the compiled application
COPY --from=builder /build/dist/app /app/app

# Make port 8000 available
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Create temporary directory for uploads
RUN mkdir -p /tmp/transcription_uploads

# Set permissions
RUN chmod +x /app/app

# Run the compiled application
CMD ["/app/app"]