import torch
import os
import json
import numpy as np
from datetime import datetime
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import soundfile as sf  # Using soundfile instead of librosa
import gc
import wave


class WhisperTranscriber:
    def __init__(self, model_size="large", device=None, language="auto", batch_size=8):
        """
        Initialize a Whisper-based transcription system optimized for file input.
        Supports Apple Silicon (M-series) GPU acceleration.
        Always translates output to English regardless of input language.

        Args:
            model_size: Size of Whisper model ("tiny", "base", "small", "medium", "large")
            device: Device to run the model on ('cpu', 'cuda', or 'mps')
            language: Language code for input detection (or "auto" for automatic detection)
            batch_size: Batch size for processing segments (higher values use more memory but may be faster)
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

        # Set language mode
        self.language = language
        self.translation_mode = True  # Always translate to English
        print(f"Using input language: {language}, with translation to English enabled")

        # Set up model
        self.model_size = model_size
        self.whisper_model_name = f"openai/whisper-{model_size}"
        self.batch_size = batch_size

        print(f"Loading Whisper {model_size} model...")
        self._setup_whisper()
        print("Whisper model loaded!")

        # Create output directory
        os.makedirs("transcripts", exist_ok=True)

        # Language name mapping for reference
        self.language_names = {
            "en": "English",
            "hi": "Hindi",
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
            # Apple Silicon optimizations - use float16 for better performance on M4
            load_options = {"torch_dtype": torch.float16, "low_cpu_mem_usage": True}

        # Load model
        self.processor = WhisperProcessor.from_pretrained(self.whisper_model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            self.whisper_model_name, **load_options
        )

        # Move model to device
        self.model.to(self.device)

        # Set model to evaluation mode for faster inference
        self.model.eval()

        # Clear cache after loading if using GPU
        if self.device in ["cuda", "mps"]:
            torch.cuda.empty_cache() if self.device == "cuda" else gc.collect()

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

    def _load_audio(self, file_path):
        """
        Load audio without using librosa's resampling functionality
        """
        print(f"Loading audio file: {file_path}")

        # Check if file exists and has content
        if not os.path.exists(file_path):
            print(f"ERROR: File not found: {file_path}")
            return np.zeros(1600), 16000  # Return 0.1s of silence

        file_size = os.path.getsize(file_path)
        print(f"File size: {file_size} bytes")

        if file_size == 0:
            print("ERROR: File is empty (0 bytes)")
            return np.zeros(1600), 16000  # Return 0.1s of silence

        try:
            # Try using soundfile first
            audio_array, sample_rate = sf.read(file_path)
            print(
                f"Loaded audio with soundfile: shape={audio_array.shape}, sample_rate={sample_rate}Hz"
            )

            # Convert stereo to mono if needed
            if len(audio_array.shape) > 1 and audio_array.shape[1] > 1:
                audio_array = audio_array.mean(axis=1)
                print(f"Converted stereo to mono: {audio_array.shape}")

        except Exception as sf_error:
            print(f"Soundfile failed: {sf_error}, trying wav module...")

            try:
                # Fallback to wave module for WAV files
                with wave.open(file_path, "rb") as wav_file:
                    sample_rate = wav_file.getframerate()
                    n_channels = wav_file.getnchannels()
                    n_frames = wav_file.getnframes()
                    sample_width = wav_file.getsampwidth()

                    # Read all frames
                    frames = wav_file.readframes(n_frames)

                    # Convert bytes to numpy array
                    if sample_width == 2:  # 16-bit audio
                        dtype = np.int16
                    elif sample_width == 4:  # 32-bit audio
                        dtype = np.int32
                    else:  # Default to 16-bit
                        dtype = np.int16

                    audio_array = np.frombuffer(frames, dtype=dtype)

                    # Convert to float32 and normalize to [-1, 1]
                    audio_array = audio_array.astype(np.float32) / np.iinfo(dtype).max

                    # Convert stereo to mono if needed
                    if n_channels == 2:
                        audio_array = audio_array.reshape(-1, 2).mean(axis=1)

                    print(
                        f"Loaded audio with wave module: shape={audio_array.shape}, sample_rate={sample_rate}Hz"
                    )
            except Exception as wav_error:
                print(f"Wave module also failed: {wav_error}")
                print("Returning empty audio")
                return np.zeros(1600), 16000  # Return 0.1s of silence

        # Resample to 16kHz if needed (simple method)
        if sample_rate != 16000:
            print(f"Resampling from {sample_rate}Hz to 16000Hz")
            # Simple resampling using linear interpolation (not high quality but works without resampy)
            original_length = len(audio_array)
            target_length = int(original_length * 16000 / sample_rate)

            # Create a time array for the original and target sample rates
            original_time = np.linspace(
                0, original_length / sample_rate, original_length
            )
            target_time = np.linspace(0, original_length / sample_rate, target_length)

            # Interpolate to get the new waveform
            resampled_audio = np.interp(target_time, original_time, audio_array)

            audio_array = resampled_audio
            sample_rate = 16000
            print(f"Resampled audio shape: {audio_array.shape}")

        # Normalize audio
        max_amp = np.max(np.abs(audio_array))
        if max_amp > 0:
            # Check if audio is very quiet
            if max_amp < 0.01:
                print(
                    f"WARNING: Audio signal is very quiet (max amplitude: {max_amp:.6f}), amplifying..."
                )
                # Boost amplitude for quiet audio
                audio_array = audio_array * min(0.9 / max_amp, 10)
                print(
                    f"New max amplitude after boost: {np.max(np.abs(audio_array)):.6f}"
                )
            elif max_amp > 1.0:
                # Normalize if too loud
                audio_array = audio_array / max_amp
                print(
                    f"Normalized loud audio, new max amplitude: {np.max(np.abs(audio_array)):.6f}"
                )

        # Print final stats
        print(f"Final audio: {len(audio_array)/16000:.2f} seconds at {sample_rate}Hz")

        return audio_array, sample_rate

    def transcribe_file(self, file_path, output_json=None, language=None):
        """
        Transcribe an audio file and save the result as JSON.
        Always translates to English regardless of input language.

        Args:
            file_path: Path to the audio file to transcribe in any language
            output_json: Path for the output JSON file (default: based on input filename)
            language: Optional source language hint (or "auto" for detection)

        Returns:
            Transcription result as a dictionary with English text
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        # Handle language override for this specific transcription
        original_language = None
        if language and language != self.language:
            original_language = self.language
            print(f"Temporarily switching to {language} for this transcription")
            self.language = language

        print(f"Transcribing in language: {self.language}")

        # Set default output filename if not provided
        if output_json is None:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            output_json = f"transcripts/{base_name}_transcript.json"

        # Create directory if needed
        os.makedirs(os.path.dirname(output_json), exist_ok=True)

        # Load and preprocess audio file with optimized loading
        try:
            audio_array, _ = self._load_audio(file_path)

            print(f"Audio loaded: {len(audio_array)/16000:.2f} seconds")

            # Process in segments for longer files - adjust segment length for faster processing
            segment_length = (
                25 * 16000
            )  # 25 seconds at 16kHz (slightly shorter segments)
            overlap = 0.2  # 0.2 second overlap (reduced for speed)
            overlap_samples = int(overlap * 16000)

            transcript_segments = []

            # Check if audio is short enough to process all at once
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
                # For longer audio, process in batches
                segments_to_process = []
                segment_times = []

                # Divide audio into segments
                for i in range(0, len(audio_array), segment_length - overlap_samples):
                    segment_audio = audio_array[i : i + segment_length]

                    # Skip processing if segment is too short
                    if len(segment_audio) < 0.5 * 16000:  # at least 0.5 second
                        continue

                    segments_to_process.append(segment_audio)
                    segment_times.append((i / 16000, (i + len(segment_audio)) / 16000))

                    # When we reach the batch size or finish all segments, process them
                    if len(
                        segments_to_process
                    ) >= self.batch_size or i + segment_length >= len(audio_array):
                        batch_results = self._batch_transcribe(segments_to_process)

                        for idx, (text, (start_time, end_time)) in enumerate(
                            zip(batch_results, segment_times)
                        ):
                            if text and len(text.strip()) > 0:
                                timestamp = str(datetime.now())
                                segment = {
                                    "text": text,
                                    "start_time": start_time,
                                    "end_time": end_time,
                                    "timestamp": timestamp,
                                }
                                transcript_segments.append(segment)

                        # Clear for next batch
                        segments_to_process = []
                        segment_times = []

            # Create the full result with combined text
            full_transcript = " ".join([seg["text"] for seg in transcript_segments])
            print(
                f"Full transcript (length {len(full_transcript)}): '{full_transcript[:100]}...'"
            )

            detected_language = (
                self.language if self.language != "auto" else "detected automatically"
            )
            result = {
                "transcript": full_transcript,
                "segments": transcript_segments,
                "metadata": {
                    "file": file_path,
                    "model": f"whisper-{self.model_size}",
                    "source_language": detected_language,
                    "source_language_name": self.language_names.get(
                        detected_language, detected_language
                    ),
                    "output_language": "en",
                    "output_language_name": "English",
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

            # Force garbage collection to free memory
            if self.device == "mps":
                gc.collect()
            elif self.device == "cuda":
                torch.cuda.empty_cache()

            return result

        except Exception as e:
            # Restore original language even if an error occurred
            if original_language:
                self.language = original_language

            print(f"Error transcribing file: {e}")
            import traceback

            traceback.print_exc()
            return {"error": str(e), "transcript": "", "segments": []}

    def _batch_transcribe(self, audio_segments):
        """Process multiple audio segments at once"""
        results = []

        for segment in audio_segments:
            text = self._transcribe_with_whisper(segment)
            results.append(text)

        return results

    def _transcribe_with_whisper(self, audio_array):
        """Transcribe audio using Whisper model with optimizations and translate to English"""
        # Clear GPU cache if using CUDA
        if self.device == "cuda":
            torch.cuda.empty_cache()

        # Add debug logging
        print(
            f"Processing audio segment of length: {len(audio_array)/16000:.2f} seconds"
        )

        # Check if audio array is valid
        if len(audio_array) < 400:  # Very short audio might not have enough content
            print(
                f"WARNING: Audio segment too short ({len(audio_array)} samples), may be empty or corrupted"
            )
            return ""

        # Normalize audio to ensure it has sufficient volume
        if np.max(np.abs(audio_array)) < 0.001:
            print(
                f"WARNING: Audio volume is very low, max amplitude: {np.max(np.abs(audio_array))}"
            )
            # Amplify if too quiet
            audio_array = audio_array * 100

        # Convert audio to features
        try:
            input_features = self.processor(
                audio_array, sampling_rate=16000, return_tensors="pt"
            ).input_features
            print(f"Feature extraction successful: {input_features.shape}")
        except Exception as e:
            print(f"ERROR in feature extraction: {str(e)}")
            return ""

        # Convert to correct precision based on device
        if self.device in ["cuda", "mps"]:
            input_features = (
                input_features.half()
            )  # Use half precision for both CUDA and MPS

        # Move to device
        input_features = input_features.to(self.device)

        # Create attention mask for better performance
        attention_mask = torch.ones(
            input_features.shape[:2], dtype=torch.long, device=input_features.device
        )

        # Generate with optimized parameters
        try:
            with torch.no_grad():
                # Always use translate task for consistent English output
                forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                    task="translate",
                    language=None,  # Allow Whisper to detect input language
                )

                print(f"Using task: translate with auto language detection")

                # Optimize generation parameters based on device
                generation_kwargs = {
                    "attention_mask": attention_mask,
                    "max_length": 220,  # Slightly reduced for faster generation
                    "forced_decoder_ids": forced_decoder_ids,
                    "do_sample": False,
                    "use_cache": True,
                }

                # Optimize beam search parameters for M4 Pro
                if self.device == "mps":
                    # M4 Pro specific optimizations
                    generation_kwargs["num_beams"] = 3  # Reduced from 5
                elif self.device == "cuda":
                    generation_kwargs["num_beams"] = 4
                else:
                    # CPU optimizations
                    generation_kwargs["num_beams"] = 2

                print(
                    f"Starting generation with parameters: beams={generation_kwargs.get('num_beams', 'default')}"
                )
                predicted_ids = self.model.generate(input_features, **generation_kwargs)
                print(f"Generation completed, output length: {predicted_ids.shape}")

            # Decode to text
            transcription = self.processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0].strip()

            print(
                f"Transcribed text (length {len(transcription)}): '{transcription[:100]}...'"
            )

            if not transcription:
                print("WARNING: Empty transcription result")

            return self._post_process_text(transcription)

        except Exception as e:
            print(f"ERROR in transcription: {str(e)}")
            import traceback

            traceback.print_exc()
            return ""

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
