from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Query
from fastapi.responses import JSONResponse
import uvicorn
import os
import uuid
import tempfile
import shutil
from typing import Optional, Dict, Any
from pydantic import BaseModel
import time
import asyncio
from datetime import datetime
import json

# Import the WhisperTranscriber class from our previous file
# Make sure this file is in the same directory as your API file
from whisper_transcriber import WhisperTranscriber

# Create FastAPI app
app = FastAPI(
    title="Audio Transcription API",
    description="API for transcribing audio files using Whisper",
    version="1.0.0",
)

# Create a global transcriber instance
# We'll use medium as default but allow overriding via API
transcriber = WhisperTranscriber(model_size="medium", language="en")

# Store jobs in memory (in a production environment, you'd use a database)
transcription_jobs = {}

# Create temporary directory for uploaded files
UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "transcription_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


# Models for request/response
class TranscriptionResponse(BaseModel):
    job_id: str
    status: str
    created_at: str


class TranscriptionResult(BaseModel):
    job_id: str
    status: str
    created_at: str
    completed_at: Optional[str] = None
    transcript: Optional[str] = None
    segments: Optional[list] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# Helper function to load a different model if requested
async def load_model_if_needed(model_size: str, language: str):
    global transcriber

    # Only reload if different from current configuration
    if model_size != transcriber.model_size or language != transcriber.language:
        print(f"Loading new model: {model_size} for language: {language}")
        transcriber = WhisperTranscriber(model_size=model_size, language=language)
        await asyncio.sleep(0.1)  # Give a moment for the model to load


# Background task to process transcription
async def process_transcription(
    job_id: str, file_path: str, model_size: str, language: str
):
    try:
        # Update job status
        transcription_jobs[job_id]["status"] = "processing"

        # Ensure we have the right model loaded
        await load_model_if_needed(model_size, language)

        # Create a unique output path for the JSON
        output_path = os.path.join(UPLOAD_DIR, f"{job_id}_transcript.json")

        # Perform the transcription
        result = transcriber.transcribe_file(file_path, output_path)

        # Update job with results
        transcription_jobs[job_id].update(
            {
                "status": "completed",
                "completed_at": str(datetime.now()),
                "transcript": result.get("transcript", ""),
                "segments": result.get("segments", []),
                "metadata": result.get("metadata", {}),
            }
        )

        # Clean up the temporary file
        if os.path.exists(file_path):
            os.remove(file_path)

    except Exception as e:
        # Update job with error
        transcription_jobs[job_id].update(
            {"status": "failed", "completed_at": str(datetime.now()), "error": str(e)}
        )

        # Clean up on error
        if os.path.exists(file_path):
            os.remove(file_path)


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model_size: str = Query(
        "medium", enum=["tiny", "base", "small", "medium", "large"]
    ),
    language: str = Query("en"),
):
    """
    Upload an audio file to transcribe. Returns a job ID that can be used to fetch results.

    - **file**: Audio file to transcribe (supports various formats)
    - **model_size**: Size of the Whisper model to use
    - **language**: Language code (e.g., "en" for English)
    """
    # Generate a unique job ID
    job_id = str(uuid.uuid4())

    # Save the uploaded file
    file_extension = os.path.splitext(file.filename)[1] if file.filename else ".tmp"
    temp_file_path = os.path.join(UPLOAD_DIR, f"{job_id}{file_extension}")

    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    finally:
        file.file.close()

    # Create job record
    job_info = {
        "job_id": job_id,
        "status": "queued",
        "created_at": str(datetime.now()),
        "file_path": temp_file_path,
        "model_size": model_size,
        "language": language,
    }
    transcription_jobs[job_id] = job_info

    # Start background processing
    background_tasks.add_task(
        process_transcription, job_id, temp_file_path, model_size, language
    )

    # Return job ID immediately
    return TranscriptionResponse(
        job_id=job_id, status="queued", created_at=job_info["created_at"]
    )


@app.get("/jobs/{job_id}", response_model=TranscriptionResult)
async def get_job_status(job_id: str):
    """
    Get the status or result of a transcription job
    """
    if job_id not in transcription_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    return transcription_jobs[job_id]


@app.get("/")
async def root():
    """Root endpoint to verify the API is running"""
    return {"status": "online", "service": "Whisper Transcription API"}


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "model": transcriber.model_size,
        "language": transcriber.language,
        "device": transcriber.device,
    }


# Optional: endpoint to clean up old jobs (in a real app, you'd use a scheduled task)
@app.delete("/cleanup")
async def cleanup_old_jobs():
    """Remove completed jobs older than 1 hour"""
    current_time = time.time()
    removed_count = 0

    for job_id in list(transcription_jobs.keys()):
        job = transcription_jobs[job_id]
        if job["status"] in ["completed", "failed"]:
            created_time = datetime.fromisoformat(
                job["created_at"].replace("Z", "+00:00")
            )
            age_seconds = (datetime.now() - created_time).total_seconds()

            # If older than 1 hour, remove
            if age_seconds > 3600:
                transcription_jobs.pop(job_id)
                removed_count += 1

    return {"message": f"Removed {removed_count} old jobs"}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
