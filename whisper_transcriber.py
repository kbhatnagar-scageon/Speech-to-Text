import torch
import os
import json
import numpy as np
from datetime import datetime
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa


class WhisperTranscriber:
    def __init__(self, model_size="large", device=None, language="hi"):
        """
        Initialize a Whisper-based transcription system optimized for file input.
        Supports Apple Silicon (M-series) GPU acceleration.

        Args:
            model_size: Size of Whisper model ("tiny", "base", "small", "medium", "large")
            device: Device to run the model on ('cpu', 'cuda', or 'mps')
            language: Language code for transcription (e.g., "en" for English, "hi" for Hindi)
        """
        # Set up the device with Apple Silicon (M-series) support
        if device is None:
            # Check for MPS (Metal Performance Shaders) for Mac M-series chips
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        print(f"Using device: {self.device}")

        # Set language
        self.language = language
        print(f"Transcribing in language: {language}")

        # Set up model
        self.model_size = model_size
        self.whisper_model_name = f"openai/whisper-{model_size}"
        print(f"Loading Whisper {model_size} model...")
        self._setup_whisper()
        print("Whisper model loaded!")

        # Create output directory
        os.makedirs("transcripts", exist_ok=True)

        # Language name mapping for reference
        self.language_names = {
            "en": "English",
            "hi": "Hindi",
            # Add more languages as needed
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "ja": "Japanese",
            "zh": "Chinese",
            "ru": "Russian",
        }

    def _setup_whisper(self):
        """Setup Whisper model with optimizations for different GPU types"""
        load_options = {}

        # Use optimizations based on device type
        if self.device == "cuda":
            # NVIDIA GPU optimizations
            load_options = {"torch_dtype": torch.float16, "low_cpu_mem_usage": True}
        elif self.device == "mps":
            # Apple Silicon optimizations
            # MPS works better with float32 for some operations
            load_options = {"low_cpu_mem_usage": True}

        # Load model
        self.processor = WhisperProcessor.from_pretrained(self.whisper_model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            self.whisper_model_name, **load_options
        )

        # Move model to device
        self.model.to(self.device)

        # Clear cache after loading if using GPU
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def change_language(self, language):
        """
        Change the language used for transcription.

        Args:
            language: Language code (e.g., "en" for English, "hi" for Hindi)

        Returns:
            str: Message confirming the language change
        """
        if language == self.language:
            return f"Already using {self.language_names.get(language, language)} for transcription."

        self.language = language
        lang_name = self.language_names.get(language, language)
        print(f"Changed transcription language to: {lang_name}")
        return f"Transcription language changed to: {lang_name}"

    def get_available_languages(self):
        """
        Return a list of available languages with their codes.

        Returns:
            dict: Dictionary mapping language codes to language names
        """
        return self.language_names

    def get_current_language(self):
        """
        Return the current language being used for transcription.

        Returns:
            str: Current language name and code
        """
        lang_name = self.language_names.get(self.language, self.language)
        return f"Current language: {lang_name} (code: {self.language})"

    def transcribe_file(self, file_path, output_json=None, language=None):
        """
        Transcribe an audio file and save the result as JSON.

        Args:
            file_path: Path to the audio file to transcribe
            output_json: Path for the output JSON file (default: based on input filename)
            language: Optional language override for this specific transcription

        Returns:
            Transcription result as a dictionary
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        # Handle language override for this specific transcription
        original_language = None
        if language and language != self.language:
            original_language = self.language
            print(f"Temporarily switching to {language} for this transcription")
            self.language = language

        print(f"Loading audio file: {file_path}")
        print(
            f"Transcribing in language: {self.language_names.get(self.language, self.language)}"
        )

        # Set default output filename if not provided
        if output_json is None:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            output_json = f"transcripts/{base_name}_transcript_{self.language}.json"

        # Create directory if needed
        os.makedirs(os.path.dirname(output_json), exist_ok=True)

        # Load and preprocess audio file with librosa (handles various formats)
        try:
            # Load audio with librosa which supports many formats
            audio_array, sample_rate = librosa.load(file_path, sr=16000, mono=True)

            # Normalize audio
            audio_array = (
                audio_array / np.max(np.abs(audio_array))
                if np.max(np.abs(audio_array)) > 0
                else audio_array
            )

            print(f"Audio loaded: {len(audio_array)/16000:.2f} seconds")

            # Process in segments for longer files
            segment_length = 30 * 16000  # 30 seconds at 16kHz
            transcript_segments = []

            # If audio is short enough, process it all at once
            if len(audio_array) <= segment_length:
                text = self._transcribe_with_whisper(audio_array)
                if text:
                    timestamp = str(datetime.now())
                    segment = {
                        "text": text,
                        "start_time": 0,
                        "end_time": len(audio_array) / 16000,
                        "timestamp": timestamp,
                    }
                    transcript_segments.append(segment)
            else:
                # Process longer audio in segments with overlap
                overlap = 0.5  # 0.5 second overlap
                overlap_samples = int(overlap * 16000)

                for i in range(0, len(audio_array), segment_length - overlap_samples):
                    segment_audio = audio_array[i : i + segment_length]

                    # Skip processing if segment is too short
                    if len(segment_audio) < 1.0 * 16000:  # at least 1 second
                        continue

                    text = self._transcribe_with_whisper(segment_audio)
                    if text and len(text.strip()) > 0:
                        timestamp = str(datetime.now())
                        segment = {
                            "text": text,
                            "start_time": i / 16000,
                            "end_time": (i + len(segment_audio)) / 16000,
                            "timestamp": timestamp,
                        }
                        transcript_segments.append(segment)

            # Create the full result
            result = {
                "transcript": " ".join([seg["text"] for seg in transcript_segments]),
                "segments": transcript_segments,
                "metadata": {
                    "file": file_path,
                    "model": f"whisper-{self.model_size}",
                    "language": self.language,
                    "language_name": self.language_names.get(
                        self.language, self.language
                    ),
                    "device": self.device,
                    "processing_date": str(datetime.now()),
                    "duration_seconds": len(audio_array) / 16000,
                },
            }

            # Save to JSON file
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            print(f"Transcription saved to {output_json}")

            # Restore original language if it was temporarily changed
            if original_language:
                self.language = original_language
                print(
                    f"Restored language to {self.language_names.get(self.language, self.language)}"
                )

            return result

        except Exception as e:
            # Restore original language even if an error occurred
            if original_language:
                self.language = original_language

            print(f"Error transcribing file: {e}")
            import traceback

            traceback.print_exc()
            return {"error": str(e)}

    def _transcribe_with_whisper(self, audio_array):
        """Transcribe audio using Whisper model"""
        # Clear GPU cache if using CUDA
        if self.device == "cuda":
            torch.cuda.empty_cache()

        # Convert audio to features
        input_features = self.processor(
            audio_array, sampling_rate=16000, return_tensors="pt"
        ).input_features

        # Convert to correct precision based on device
        if self.device == "cuda":
            input_features = input_features.half()

        # Move to device
        input_features = input_features.to(self.device)

        # Create attention mask to fix warning
        attention_mask = torch.ones(
            input_features.shape[:2], dtype=torch.long, device=input_features.device
        )

        # Generate with optimized parameters
        with torch.no_grad():
            # Force language to avoid language detection warning
            forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                language=self.language, task="transcribe"
            )

            # Optimize generation parameters based on device
            generation_kwargs = {
                "attention_mask": attention_mask,
                "max_length": 256,
                "forced_decoder_ids": forced_decoder_ids,
                "do_sample": False,
                "use_cache": True,
            }

            # Use more beams on GPU for better quality
            if self.device in ["cuda", "mps"]:
                generation_kwargs["num_beams"] = 5
            else:
                # Lower beam count on CPU for better performance
                generation_kwargs["num_beams"] = 2

            predicted_ids = self.model.generate(input_features, **generation_kwargs)

        # Decode to text
        transcription = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0].strip()

        return self._post_process_text(transcription)

    def _post_process_text(self, text):
        """Clean up transcribed text"""
        if not text:
            return ""

        # Fix common punctuation issues
        text = text.replace(" ,", ",")
        text = text.replace(" .", ".")
        text = text.replace(" ?", "?")
        text = text.replace(" !", "!")

        return text.strip()
