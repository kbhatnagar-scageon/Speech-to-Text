from fastapi import FastAPI, UploadFile, File, HTTPException, Query
import uvicorn
import os
import uuid
import tempfile
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from datetime import datetime

# Import the optimized WhisperTranscriber class
from whisper_transcriber import WhisperTranscriber

# Create FastAPI app
app = FastAPI(
    title="Speech Transcription API",
    description="API for transcr ibing speech in audio files using Whisper with Mac M4 Pro optimization",
    version="1.1.0",
)

# Create temporary directory for uploaded files
UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "transcription_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


# Model for response
class TranscriptionResponse(BaseModel):
    job_id: str
    status: str
    created_at: str
    text_output: str
    segments: list
    processing_time_ms: int
    device_used: str
    language: str  # Added field to show which language was used
    language_name: str  # Added field to show language name


# Model for language info
class LanguageInfo(BaseModel):
    code: str
    name: str


# Model for available languages response
class LanguagesResponse(BaseModel):
    available_languages: List[LanguageInfo]
    current_language: str
    current_language_name: str


# Initialize the transcriber as a global variable at startup
transcriber = None


@app.on_event("startup")
async def startup_event():
    global transcriber
    print("Loading Whisper model (large) with Apple Silicon acceleration...")
    # The optimized transcriber will automatically detect and use MPS on Mac M4
    transcriber = WhisperTranscriber(model_size="large", language="hi")
    print(f"Whisper model loaded successfully on device: {transcriber.device}")


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    language: Optional[str] = Query(
        None, description="Language code (e.g., 'en' for English, 'hi' for Hindi)"
    ),
):
    """
    Upload a speech audio file and get transcription JSON.

    - **file**: Audio file containing speech (supports various formats)
    - **language**: Optional language code to use for this transcription (e.g., 'en', 'hi')
    """
    start_time = datetime.now()

    # Generate a unique job ID
    job_id = str(uuid.uuid4())

    # Get original filename and extension
    original_filename = file.filename or "unknown_file"
    file_extension = (
        os.path.splitext(original_filename)[1] if original_filename else ".tmp"
    )

    # Create a safe filename with the job_id and original extension
    safe_filename = f"{job_id}{file_extension}"

    # Path for temporary processing
    temp_file_path = os.path.join(UPLOAD_DIR, safe_filename)

    try:
        # Save the uploaded file temporarily for processing
        file_contents = await file.read()
        with open(temp_file_path, "wb") as buffer:
            buffer.write(file_contents)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to save temporary file: {str(e)}"
        )

    try:
        # Create output path for JSON
        output_json_path = os.path.join(UPLOAD_DIR, f"{job_id}_transcript.json")

        # Get the language to use (either from query parameter or current default)
        transcription_language = language or transcriber.language

        # Perform transcription using the pre-loaded model
        result = transcriber.transcribe_file(
            temp_file_path, output_json_path, language=transcription_language
        )

        # Get transcript text
        transcript_text = result.get("transcript", "")

        # Process segments to include timing info but avoid duplicating full text
        segments = []
        for segment in result.get("segments", []):
            # Only keep timing information and segment ID if present
            processed_segment = {
                "start_time": segment.get("start_time", 0),
                "end_time": segment.get("end_time", 0),
                "timestamp": segment.get("timestamp", datetime.now().isoformat()),
            }
            # If there are individual segment texts that are different from the full transcript, keep those
            if "text" in segment and segment["text"] != transcript_text:
                processed_segment["text"] = segment["text"]

            segments.append(processed_segment)

        # Calculate processing time
        end_time = datetime.now()
        processing_time_ms = int((end_time - start_time).total_seconds() * 1000)

        # Get metadata
        metadata = result.get("metadata", {})
        device_used = metadata.get("device", "unknown")
        used_language = metadata.get("language", transcription_language)
        language_name = metadata.get(
            "language_name",
            transcriber.language_names.get(used_language, used_language),
        )

        # Create response
        response = TranscriptionResponse(
            job_id=job_id,
            status="completed",
            created_at=datetime.now().isoformat(),
            text_output=transcript_text,
            segments=segments,
            processing_time_ms=processing_time_ms,
            device_used=device_used,
            language=used_language,
            language_name=language_name,
        )

        # Clean up temporary files
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        if os.path.exists(output_json_path):
            os.remove(output_json_path)

        return response

    except Exception as e:
        # Clean up on error
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@app.get("/languages", response_model=LanguagesResponse)
async def get_available_languages():
    """Get a list of available languages for transcription"""
    if not transcriber:
        raise HTTPException(
            status_code=503, detail="Transcription service not initialized"
        )

    languages = transcriber.get_available_languages()
    language_list = [
        LanguageInfo(code=code, name=name) for code, name in languages.items()
    ]

    # Get current language
    current_lang = transcriber.language
    current_lang_name = transcriber.language_names.get(current_lang, current_lang)

    return LanguagesResponse(
        available_languages=language_list,
        current_language=current_lang,
        current_language_name=current_lang_name,
    )


@app.post("/set_language/{language_code}", response_model=Dict[str, str])
async def set_language(language_code: str):
    """Set the default language for transcription"""
    if not transcriber:
        raise HTTPException(
            status_code=503, detail="Transcription service not initialized"
        )

    # Get available languages
    available_languages = transcriber.get_available_languages()

    # Check if the language code is valid
    if language_code not in available_languages and language_code != "auto":
        raise HTTPException(
            status_code=400,
            detail=f"Invalid language code. Available codes: {', '.join(available_languages.keys())} or 'auto'",
        )

    # Change the language
    message = transcriber.change_language(language_code)

    return {"message": message}


@app.get("/")
async def root():
    """Root endpoint to verify the API is running"""
    if not transcriber:
        return {
            "status": "initializing",
            "service": "Speech Transcription API with Mac M4 Pro optimization",
        }

    return {
        "status": "online",
        "service": "Speech Transcription API with Mac M4 Pro optimization",
        "device": transcriber.device,
        "current_language": transcriber.language,
        "current_language_name": transcriber.language_names.get(
            transcriber.language, transcriber.language
        ),
        "endpoints": [
            {
                "path": "/transcribe",
                "method": "POST",
                "description": "Upload and transcribe an audio file",
            },
            {
                "path": "/languages",
                "method": "GET",
                "description": "Get available languages for transcription",
            },
            {
                "path": "/set_language/{language_code}",
                "method": "POST",
                "description": "Set the default language for transcription",
            },
        ],
        "model": f"Whisper large (loaded on {transcriber.device})",
    }


if __name__ == "__main__":
    uvicorn.run("sst_api:app", host="0.0.0.0", port=8000, reload=False)
