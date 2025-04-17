# Speech-to-English Transcription API

A high-performance FastAPI service that transcribes speech from audio files in any language to English text using OpenAI's Whisper model with optimizations for Apple Silicon (especially M4 Pro).

## Features

- 🎙️ Transcribes speech from audio files in **any language** to **English text**
- 🚀 Optimized for Apple Silicon (M-series) chips with MPS acceleration
- 💪 Supports NVIDIA CUDA for GPU acceleration on compatible hardware
- 🔄 Automatic language detection or manual language selection
- ⚡ Batch processing for efficient handling of long audio files
- 🧩 Segment-level timestamps for precise audio-text alignment
- 📊 Detailed metadata and processing statistics
- 🌐 REST API with comprehensive documentation

## System Requirements

- Python 3.8+
- For Apple Silicon optimization: Mac with M-series chip (M1, M2, M3, M4)
- Minimum 8GB RAM (16GB+ recommended for large model)
- 5GB+ disk space for model and dependencies

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/speech-transcription-api.git
cd speech-transcription-api
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Starting the API Server

Run the API with the following command:

```bash
python sst_api.py
```

This will:
- Load the Whisper large model with optimizations for your hardware
- Start a FastAPI server on http://localhost:8000
- Initialize with automatic language detection

You can also run it with Uvicorn directly:

```bash
uvicorn sst_api:app --host 0.0.0.0 --port 8000 --workers 4
```

### API Endpoints

#### Transcribe Audio

```
POST /transcribe
```

Upload an audio file to transcribe speech in any language to English text.

**Parameters:**
- `file`: Audio file (MP3, WAV, FLAC, etc.)
- `language` (optional): Source language code (e.g., 'en', 'hi') or 'auto'

**Example using curl:**

```bash
curl -X POST "http://localhost:8000/transcribe" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_audio_file.mp3" \
  -F "language=auto"
```

**Response:**

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "created_at": "2025-04-17T12:34:56.789012",
  "text_output": "This is the transcribed text in English.",
  "segments": [
    {
      "start_time": 0.0,
      "end_time": 3.5,
      "timestamp": "2025-04-17T12:34:56.789012"
    }
  ],
  "processing_time_ms": 1234,
  "device_used": "mps",
  "language": "en",
  "language_name": "English"
}
```

#### Get Available Languages

```
GET /languages
```

Returns a list of supported languages for transcription.

**Example:**

```bash
curl -X GET "http://localhost:8000/languages"
```

#### Set Default Language

```
POST /set_language/{language_code}
```

Set the default source language for transcription or use 'auto' for automatic detection.

**Example:**

```bash
curl -X POST "http://localhost:8000/set_language/hi"
```

#### API Status

```
GET /
```

Check the API status, current language, and device being used.

**Example:**

```bash
curl -X GET "http://localhost:8000/"
```

### Interactive API Documentation

After starting the server, visit:
- http://localhost:8000/docs - Swagger UI
- http://localhost:8000/redoc - ReDoc UI

## Architecture

This system consists of two main components:

1. **WhisperTranscriber Class** (`whisper_transcriber.py`):
   - Handles the Whisper model loading and optimization
   - Provides audio processing and transcription functionality
   - Manages language settings and device-specific optimizations

2. **FastAPI Application** (`sst_api.py`):
   - Provides REST API endpoints
   - Handles file uploads and request processing
   - Returns structured JSON responses

## Performance Optimization

The system includes several optimizations for Apple Silicon M-series chips:
- Uses Metal Performance Shaders (MPS) backend when available
- Applies float16 precision for faster inference on M-series chips
- Optimized batch processing for long audio files
- Memory management optimizations for Apple Silicon

## Customization

You can modify the following parameters in `whisper_transcriber.py`:
- `model_size`: Change to "tiny", "base", "small", "medium" for different performance/accuracy tradeoffs
- `batch_size`: Adjust based on your available memory
- Add more languages to the `language_names` dictionary

## License

[MIT License](LICENSE)

## Acknowledgements

This project uses the following key technologies:
- [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition model
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework
- [Transformers](https://huggingface.co/docs/transformers/index) - Hugging Face implementation of Whisper
- [PyTorch](https://pytorch.org/) - Deep learning framework
