import argparse
import os
import tempfile
import time
import threading
import subprocess
import numpy as np
import gradio as gr
import librosa
import soundfile as sf
import torch
from pathlib import Path
from typing import Iterator, Dict, Any
from transformers.utils import logging
from transformers import set_seed
import sys
import pygame # Import Pygame for audio playback
from flask import Flask, request, jsonify, Response, stream_with_context
import uuid

# --- VibeVoice Integration (Copied and adapted from your script) ---

vibevoice_dir = Path('./VibeVoice')

if str(vibevoice_dir) not in sys.path:
    sys.path.insert(0, str(vibevoice_dir))

# --- Import VibeVoice modules ---
try:
    from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
    from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
    from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
    from vibevoice.modular.streamer import AudioStreamer
except ImportError:
    print("Import error: Could not import VibeVoice modules directly.")
    print("Attempting to load modules from VibeVoice directory...")
    try:
        import importlib.util

        def load_module(module_name, file_path):
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return module

        config_module = load_module("vibevoice_config", vibevoice_dir / "modular" / "configuration_vibevoice.py")
        VibeVoiceConfig = config_module.VibeVoiceConfig

        model_module = load_module("vibevoice_model", vibevoice_dir / "modular" / "modeling_vibevoice_inference.py")
        VibeVoiceForConditionalGenerationInference = model_module.VibeVoiceForConditionalGenerationInference

        processor_module = load_module("vibevoice_processor", vibevoice_dir / "processor" / "vibevoice_processor.py")
        VibeVoiceProcessor = processor_module.VibeVoiceProcessor

        streamer_module = load_module("vibevoice_streamer", vibevoice_dir / "modular" / "streamer.py")
        AudioStreamer = streamer_module.AudioStreamer

        print("✓ VibeVoice modules loaded dynamically.")

    except Exception as e:
        raise ImportError(
            f"VibeVoice module not found. Error: {e}\n"
            "Please ensure VibeVoice is properly installed:\n"
            "git clone https://github.com/vibevoice-community/VibeVoice\n"
            "cd VibeVoice/\n"
            "pip install -e .\n"
            "pip install wheel flash-attn --no-build-isolation\n"
        )

logging.set_verbosity_info()
logger = logging.get_logger(__name__)

# --- Global variables for VibeVoiceChat instance and parameters ---
# These will be initialized once when the Flask app starts.
vibevoice_chat_instance = None
default_args = {}

class VibeVoiceProcessorWrapper:
    def __init__(self, model_path: str, device: str = "cuda", inference_steps: int = 5, default_voice_path: str = None):
        """Initialize the VibeVoice processor and model."""
        self.model_path = model_path
        self.device = device if torch.cuda.is_available() else "cpu"
        self.inference_steps = inference_steps
        self.is_generating = False
        self.stop_generation = False
        self.current_streamer = None
        self.default_voice_path = default_voice_path

        self._initialize_device()
        self._process_default_voice()
        self.load_model()
        self.setup_voice_presets()

    def _initialize_device(self):
        """Initialize device and check GPU availability."""
        if torch.cuda.is_available():
            print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
            try:
                total_memory_gb = torch.cuda.get_device_memory_info(0).total / 1e9
            except AttributeError:
                total_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  Memory: {total_memory_gb:.2f} GB")
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  PyTorch CUDA: {torch.cuda.is_available()}")
            try:
                torch.cuda.set_per_process_memory_fraction(0.95, 0)
            except AttributeError:
                print("Warning: Could not set per-process memory fraction.")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        else:
            print("✗ No GPU detected, using CPU (generation will be VERY slow)")
            print("  For faster generation, ensure CUDA is properly installed")

    def _process_default_voice(self):
        """Process the default voice sample if provided."""
        self.processed_default_voice_sample = None
        if self.default_voice_path and Path(self.default_voice_path).exists():
            print(f"Processing default voice from: {self.default_voice_path}")
            try:
                self.processed_default_voice_sample = self.read_audio(self.default_voice_path)
                print("✓ Default voice processed successfully.")
            except Exception as e:
                print(f"Warning: Failed to process default voice '{self.default_voice_path}': {e}")
        elif self.default_voice_path:
            print(f"Warning: Default voice path '{self.default_voice_path}' does not exist or is invalid.")

    def load_model(self):
        """Load the VibeVoice model and processor."""
        print(f"Loading model from {self.model_path}")
        start_time = time.time()

        try:
            self.processor = VibeVoiceProcessor.from_pretrained(self.model_path)
            print("✓ Processor loaded.")
        except Exception as e:
            print(f"Error loading processor from {self.model_path}: {e}")
            raise

        if torch.cuda.is_available():
            print("Loading model with GPU acceleration...")
            try:
                self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.bfloat16,
                    device_map='auto',
                    attn_implementation="flash_attention_2",
                    low_cpu_mem_usage=True,
                )
                print("✓ Model loaded with Flash Attention 2.")
            except Exception as e:
                print(f"Warning: Could not load with flash_attention_2: {e}")
                print("Falling back to standard attention and device_map='cuda:0'...")
                try:
                    self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                        self.model_path,
                        torch_dtype=torch.bfloat16,
                        device_map='cuda:0',
                        low_cpu_mem_usage=True,
                    )
                    print("✓ Model loaded on GPU with standard attention.")
                except Exception as gpu_e:
                    print(f"Error loading model on GPU: {gpu_e}")
                    print("Falling back to CPU...")
                    self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                        self.model_path,
                        torch_dtype=torch.float32,
                        device_map='cpu',
                        low_cpu_mem_usage=True,
                    )
                    print("✓ Model loaded on CPU.")
        else:
            print("Loading model on CPU (this will be slow)...")
            self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32,
                device_map='cpu',
                low_cpu_mem_usage=True,
            )
            print("✓ Model loaded on CPU.")

        self.model.eval()

        try:
            self.model.model.noise_scheduler = self.model.model.noise_scheduler.from_config(
                self.model.model.noise_scheduler.config,
                algorithm_type='sde-dpmsolver++',
                beta_schedule='squaredcos_cap_v2'
            )
            self.model.set_ddpm_inference_steps(num_steps=self.inference_steps)
            print(f"✓ Noise scheduler configured with {self.inference_steps} DDPM inference steps.")
        except Exception as e:
            print(f"Warning: Could not configure noise scheduler: {e}")

        load_time = time.time() - start_time
        print(f"✓ Model loaded in {load_time:.2f} seconds")

        print(f"Model inferred device: {self.device}")

    def setup_voice_presets(self):
        """Setup voice presets from the voices directory."""
        voices_dir = Path('./voices')

        if not voices_dir.exists():
            try:
                voices_dir.mkdir()
                print(f"Created voices directory at {voices_dir}")
                print("Please add voice sample files (.wav, .mp3, etc.) to this directory")
            except OSError as e:
                print(f"Error creating voices directory {voices_dir}: {e}")

        self.available_voices = {}
        audio_extensions = ('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac')

        if voices_dir.exists():
            for file_path in voices_dir.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in audio_extensions:
                    name = file_path.stem
                    self.available_voices[name] = str(file_path)

        if self.default_voice_path and Path(self.default_voice_path).exists():
            default_voice_stem = Path(self.default_voice_path).stem
            if default_voice_stem not in self.available_voices:
                 self.available_voices[default_voice_stem] = self.default_voice_path
                 print(f"Added external default voice '{self.default_voice_path}' to available options.")

        self.available_voices = dict(sorted(self.available_voices.items()))

        if not self.available_voices:
            print(f"Warning: No voice files found in '{voices_dir}' or specified as default.")
            print("Using default (zero) voice samples. Add audio files to the 'voices' directory or specify --default_voice for better results.")
            self.available_voices = {"Default": None}
        else:
            print(f"Found {len(self.available_voices)} voice presets: {', '.join(self.available_voices.keys())}")

    def read_audio(self, audio_path: str, target_sr: int = 24000) -> np.ndarray:
        """Read and preprocess audio file."""
        try:
            wav, sr = sf.read(audio_path)
            if len(wav.shape) > 1:
                wav = np.mean(wav, axis=1)
            if sr != target_sr:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
            return wav
        except Exception as e:
            print(f"Error reading audio {audio_path}: {e}")
            return np.zeros(target_sr, dtype=np.float32)

    def format_script(self, message: str, num_speakers: int = 2) -> str:
        """Format input message into a script with speaker assignments."""
        lines = message.strip().split('\n')
        formatted_lines = []

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            if line.startswith('Speaker ') and ':' in line:
                try:
                    speaker_id_str = line.split(':')[0].split(' ')[1]
                    speaker_id = int(speaker_id_str)
                    if 0 <= speaker_id < num_speakers:
                        formatted_lines.append(line)
                        continue
                    else:
                        print(f"Warning: Speaker ID {speaker_id} out of range (0-{num_speakers-1}). Reassigning.")
                except (ValueError, IndexError):
                    print(f"Warning: Could not parse speaker ID from '{line}'. Reassigning.")

            speaker_id = i % num_speakers
            formatted_lines.append(f"Speaker {speaker_id}: {line}")

        return '\n'.join(formatted_lines)

    def _generate_audio_chunks(
        self,
        message: str,
        voice_1: str,
        voice_2: str,
        num_speakers: int,
        cfg_scale: float
    ) -> Iterator[tuple]:
        """Generate audio stream from text input as chunks."""
        try:
            self.stop_generation = False
            self.is_generating = True

            if not message.strip():
                print("Message is empty, aborting generation.")
                yield None
                return

            formatted_script = self.format_script(message, num_speakers)
            print(f"Formatted script:\n{formatted_script}")
            print(f"Using device: {self.device}")

            selected_voice_names = []
            if voice_1 and voice_1 != "Default":
                selected_voice_names.append(voice_1)
            if num_speakers > 1 and voice_2 and voice_2 != "Default":
                selected_voice_names.append(voice_2)

            voice_samples = []
            for i in range(num_speakers):
                audio_data = None
                voice_name_to_load = "Default"

                if i < len(selected_voice_names):
                    voice_name_to_load = selected_voice_names[i]
                elif selected_voice_names:
                    voice_name_to_load = selected_voice_names[0]

                is_default_voice_loaded_and_valid = (
                    self.processed_default_voice_sample is not None and
                    isinstance(self.processed_default_voice_sample, np.ndarray) and
                    self.processed_default_voice_sample.size > 0
                )
                
                is_selected_voice_the_default = (
                    self.default_voice_path and
                    voice_1 == Path(self.default_voice_path).stem
                )

                if i == 0 and is_default_voice_loaded_and_valid and is_selected_voice_the_default:
                    audio_data = self.processed_default_voice_sample
                    print(f"Using pre-loaded default voice sample for Speaker {i} ('{voice_1}').")
                else:
                    if voice_name_to_load in self.available_voices and self.available_voices[voice_name_to_load]:
                        try:
                            audio_data = self.read_audio(self.available_voices[voice_name_to_load])
                            print(f"Loaded voice sample for Speaker {i} ('{voice_name_to_load}') from: {self.available_voices[voice_name_to_load]}")
                        except Exception as e:
                            print(f"Error reading audio for '{voice_name_to_load}': {e}")
                            audio_data = None
                    else:
                        print(f"Voice '{voice_name_to_load}' not found in available voices or is 'Default'.")

                if audio_data is None or len(audio_data) == 0:
                    print(f"Using silence for Speaker {i} ('{voice_name_to_load}') as fallback.")
                    audio_data = np.zeros(24000, dtype=np.float32)

                voice_samples.append(audio_data)

            print(f"Prepared {len(voice_samples)} voice samples for generation.")

            inputs = self.processor(
                text=[formatted_script],
                voice_samples=[voice_samples],
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )

            if self.device == "cuda":
                print("Moving inputs to GPU...")
                inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
                print(f"✓ Inputs moved to GPU.")

            audio_streamer = AudioStreamer(batch_size=1, stop_signal=None, timeout=None)
            self.current_streamer = audio_streamer

            generation_thread = threading.Thread(
                target=self._generate_with_streamer,
                args=(inputs, cfg_scale, audio_streamer)
            )
            generation_thread.start()

            time.sleep(1.5) # Give the generation thread a moment to start

            sample_rate = 24000
            audio_stream_iter = audio_streamer.get_stream(0)

            chunk_count = 0
            for audio_chunk in audio_stream_iter:
                if self.stop_generation:
                    print("Stop generation requested, ending stream.")
                    audio_streamer.end()
                    break

                chunk_count += 1

                if torch.is_tensor(audio_chunk):
                    if audio_chunk.dtype == torch.bfloat16:
                        audio_chunk = audio_chunk.float()
                    audio_np = audio_chunk.cpu().numpy()
                else:
                    audio_np = np.array(audio_chunk, dtype=np.float32)

                if len(audio_np.shape) > 1:
                    audio_np = audio_np.squeeze()

                # Convert to 16-bit PCM WAV format for streaming
                audio_16bit = self.convert_to_16_bit_wav(audio_np)
                yield (sample_rate, audio_16bit)

            generation_thread.join(timeout=10.0)
            if generation_thread.is_alive():
                print("Warning: Generation thread did not finish within timeout.")

            print("Generation stream finished.")
            self.current_streamer = None
            self.is_generating = False

        except Exception as e:
            print(f"Error in _generate_audio_chunks: {e}")
            import traceback
            traceback.print_exc()
            self.is_generating = False
            self.current_streamer = None
            yield None # Indicate error or empty stream

    def _generate_with_streamer(self, inputs, cfg_scale, audio_streamer):
        """Helper method to run generation with streamer in a separate thread."""
        try:
            def check_stop():
                return self.stop_generation

            print("Starting generation thread...")
            if self.device == "cuda" and torch.cuda.is_available():
                print("Using autocast for mixed precision on CUDA.")
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=None,
                        cfg_scale=cfg_scale,
                        tokenizer=self.processor.tokenizer,
                        generation_config={'do_sample': False},
                        audio_streamer=audio_streamer,
                        stop_check_fn=check_stop,
                        verbose=False,
                        refresh_negative=True,
                    )
            else:
                print("Running generation on CPU or without CUDA.")
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=None,
                    cfg_scale=cfg_scale,
                    tokenizer=self.processor.tokenizer,
                    generation_config={'do_sample': False},
                    audio_streamer=audio_streamer,
                    stop_check_fn=check_stop,
                    verbose=False,
                    refresh_negative=True,
                )
            print("Generation thread finished model.generate().")
        except Exception as e:
            print(f"Error in generation thread: {e}")
            import traceback
            traceback.print_exc()
            audio_streamer.end()
        finally:
            print("Generation thread exiting.")

    def convert_to_16_bit_wav(self, data: np.ndarray) -> np.ndarray:
        """Convert audio data (float32) to 16-bit PCM numpy array."""
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        if data.dtype != np.float32:
            data = data.astype(np.float32)

        max_val = np.max(np.abs(data))
        if max_val > 1.0:
            data = data / max_val
        elif max_val == 0:
            return np.zeros_like(data, dtype=np.int16)

        data_int16 = (data * 32767).astype(np.int16)
        return data_int16

    def stop_audio_generation(self):
        """Stop the current audio generation."""
        print("Stop audio generation called.")
        self.stop_generation = True
        if self.current_streamer:
            try:
                self.current_streamer.end()
            except Exception as e:
                print(f"Error signaling streamer to end: {e}")
        self.is_generating = False
        self.current_streamer = None

# --- Flask App Setup ---
app = Flask(__name__)

# Create a temporary directory for audio files if it doesn't exist
AUDIO_DIR = Path("./generated_audio_flask")
AUDIO_DIR.mkdir(exist_ok=True)

# Initialize Pygame Mixer once for the server
try:
    pygame.mixer.init()
    print("Pygame mixer initialized.")
except pygame.error as e:
    print(f"Warning: Could not initialize Pygame mixer: {e}. Audio playback might not work.")
    # Handle this case appropriately, e.g., disable playback features or log an error.

def initialize_vibevoice_instance(model_path, device, inference_steps, default_voice):
    """Initializes the VibeVoiceProcessorWrapper instance globally."""
    global vibevoice_chat_instance
    global default_args
    
    try:
        print("Initializing VibeVoice processor and model...")
        vibevoice_chat_instance = VibeVoiceProcessorWrapper(
            model_path=model_path,
            device=device,
            inference_steps=inference_steps,
            default_voice_path=default_voice
        )
        default_args = {
            'voice_1': list(vibevoice_chat_instance.available_voices.keys())[0] if vibevoice_chat_instance.available_voices and list(vibevoice_chat_instance.available_voices.keys())[0] != "Default" else "Default",
            'voice_2': list(vibevoice_chat_instance.available_voices.keys())[1] if len(vibevoice_chat_instance.available_voices) > 1 and list(vibevoice_chat_instance.available_voices.keys())[1] != "Default" else ("Default" if vibevoice_chat_instance.available_voices and list(vibevoice_chat_instance.available_voices.keys())[0] == "Default" else (list(vibevoice_chat_instance.available_voices.keys())[0] if vibevoice_chat_instance.available_voices else "Default")),
            'num_speakers': 2,
            'cfg_scale': 1.3 # Default CFG scale
        }
        print("VibeVoice instance initialized successfully.")
    except Exception as e:
        print(f"FATAL ERROR: Failed to initialize VibeVoice: {e}")
        print("The server cannot start without VibeVoice.")
        sys.exit(1)

@app.route('/tts', methods=['POST'])
def tts_endpoint():
    """
    Receives text via POST request and returns generated audio.
    Expects JSON payload with 'text' key.
    Optional keys: 'voice_1', 'voice_2', 'num_speakers', 'cfg_scale'.
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 415 # Unsupported Media Type

    if vibevoice_chat_instance is None:
        return jsonify({"error": "VibeVoice model not initialized"}), 503 # Service Unavailable

    data = request.get_json()
    text_to_speak = data.get('text')

    if not text_to_speak:
        return jsonify({"error": "Missing 'text' field in JSON payload"}), 400 # Bad Request

    # Use default parameters if not provided in the request
    voice_1 = data.get('voice_1', default_args['voice_1'])
    voice_2 = data.get('voice_2', default_args['voice_2'])
    num_speakers = data.get('num_speakers', default_args['num_speakers'])
    cfg_scale = data.get('cfg_scale', default_args['cfg_scale'])

    # Basic validation for num_speakers
    if not (1 <= num_speakers <= 2):
        return jsonify({"error": "num_speakers must be between 1 and 2"}), 400

    # Ensure selected voices are valid
    available_voice_keys = list(vibevoice_chat_instance.available_voices.keys())
    if voice_1 not in available_voice_keys:
        return jsonify({"error": f"Invalid voice_1: '{voice_1}'. Available voices: {available_voice_keys}"}), 400
    if num_speakers == 2 and voice_2 not in available_voice_keys:
        return jsonify({"error": f"Invalid voice_2: '{voice_2}'. Available voices: {available_voice_keys}"}), 400

    # Ensure stop_generation is reset before starting a new generation
    vibevoice_chat_instance.stop_generation = False
    vibevoice_chat_instance.is_generating = False # Ensure flag is reset
    vibevoice_chat_instance.current_streamer = None # Clear any old streamer

    # Use a generator to yield audio chunks
    def generate_audio_stream_response(text, v1, v2, n_speakers, cfg):
        audio_generator = vibevoice_chat_instance._generate_audio_chunks(
            text, v1, v2, n_speakers, cfg
        )
        
        first_chunk = True
        audio_buffer = b""
        sample_rate = 24000 # Default sample rate from VibeVoice
        
        for sr, audio_chunk_np in audio_generator:
            if sr is None or audio_chunk_np is None:
                # Handle generation error or completion with no audio
                if first_chunk: # If no audio was generated at all
                    yield b"" # Send an empty response body
                break
            
            sample_rate = sr # Update sample rate if it's consistent
            
            # Convert numpy array (16-bit PCM) to bytes
            audio_bytes = audio_chunk_np.tobytes()
            
            if first_chunk:
                # For the first chunk, we need to write a WAV header
                # Using tempfile to get bytes for a WAV file, then slicing it.
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    sf.write(tmp_file.name, audio_chunk_np, sample_rate)
                    tmp_file.seek(0)
                    wav_header = tmp_file.read(44) # Standard WAV header size
                    audio_buffer += wav_header
                    audio_buffer += audio_bytes
                first_chunk = False
            else:
                audio_buffer += audio_bytes

            # Yield chunks to the client
            # We can yield the entire buffer accumulated so far or parts of it.
            # For simplicity, let's yield the header + current chunk, and then subsequent chunks.
            # A more sophisticated approach would be to stream WAV data without the header for subsequent chunks
            # and let the client handle it, or use a streaming WAV format.
            # For now, we'll send the header with the first chunk and then raw PCM data.
            
            # If we want to stream a valid WAV file, it's more complex.
            # For now, we'll try to send raw PCM data and let the client handle it,
            # or a full WAV file at the end.
            # A simpler approach for immediate playback in browsers might be to
            # send the full audio file once it's ready.
            # Let's reconsider: For streaming, it's better to send raw PCM data.
            # However, many clients expect a full WAV file.
            # Let's collect all audio and return a file at the end if streaming isn't feasible here.
            
            # --- Let's try to stream raw PCM data ---
            # The client will need to know the sample rate and format.
            # We'll set Content-Type to audio/wav and hope for the best,
            # or change to application/octet-stream and specify format via headers.
            # A more robust solution might involve MP3 or OGG streaming.
            yield audio_bytes # Yielding raw PCM bytes
            
            # If we want to send a complete WAV file at the end:
            # all_audio_chunks.append(audio_chunk_np)

        # --- If we were collecting all chunks to send a single WAV file at the end ---
        # if all_audio_chunks:
        #     final_audio_array = np.concatenate(all_audio_chunks)
        #     with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        #         sf.write(tmp_file.name, final_audio_array, sample_rate)
        #         tmp_file.seek(0)
        #         with open(tmp_file.name, 'rb') as f:
        #             audio_content = f.read()
        #     os.remove(tmp_file.name) # Clean up temp file
        #     return Response(audio_content, mimetype='audio/wav')
        # else:
        #     return Response(b"", mimetype='audio/wav') # Empty response if no audio


    # Set response headers for audio streaming
    # Using 'audio/wav' for simplicity, but raw PCM might require 'audio/x-raw' or 'application/octet-stream'
    # with proper client-side interpretation. For now, let's try to stream raw PCM data.
    # The client will need to know the sample rate (24000 Hz), channels (1), and bit depth (16-bit).
    # A common MIME type for raw PCM is 'audio/L16' or 'audio/opus' etc.
    # For simplicity and common browser support, let's try streaming WAV.
    # If streaming WAV directly is problematic, we can collect and return a full WAV file.

    # Let's aim to stream WAV data. This requires creating a WAV header for the first chunk
    # and then appending subsequent PCM data. This is complex for true streaming.
    # A simpler alternative for broad compatibility is to generate the full audio and return it.
    # However, to demonstrate streaming as requested, let's try to yield chunks.
    # For a robust streaming WAV, a library or manual header management is needed.

    # For this example, we'll yield the raw PCM data and set a common audio MIME type.
    # The client will need to interpret this (e.g., know it's 16-bit mono PCM at 24kHz).
    # Using 'audio/wav' implies a WAV file structure. If we only send PCM, it might fail.
    # Let's return a Response object that uses the generator.
    
    # To make it more robust for streaming, we'll yield raw PCM bytes.
    # The client should be configured to interpret this (e.g., using MediaRecorder API with specific mimeType).
    
    # The original Gradio code yields (sample_rate, audio_array). We need to convert this to bytes.
    # Let's modify the generator to yield bytes directly.

    # Use a unique temporary file to store the complete audio if we decide to return a full file later.
    # For now, let's try streaming PCM.
    
    try:
        # The generator yields (sample_rate, audio_chunk_np) which is (int, np.ndarray)
        # We need to convert np.ndarray to bytes.
        # And handle the WAV header if we want to stream a WAV file.
        # For simplicity in this example, let's stream the raw PCM data and set a generic audio type.
        # Clients might need explicit configuration to play this (e.g., sample rate, format).

        response = Response(stream_with_context(generate_audio_stream_response(
            text_to_speak, voice_1, voice_2, num_speakers, cfg_scale
        )), mimetype='audio/wav') # Try 'audio/wav' or 'audio/x-raw' or 'application/octet-stream'

        # If direct streaming of raw PCM causes issues, fallback to generating full WAV and returning it:
        # full_audio_response = generate_full_wav_response(text_to_speak, voice_1, voice_2, num_speakers, cfg_scale)
        # return full_audio_response

        return response

    except Exception as e:
        print(f"Error processing TTS request: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Internal server error during audio generation"}), 500

# Helper function for direct WAV file generation (fallback or alternative)
def generate_full_wav_response(text, v1, v2, n_speakers, cfg):
    all_audio_chunks = []
    sample_rate = 24000
    
    # Re-use the generator but collect all data
    audio_generator = vibevoice_chat_instance._generate_audio_chunks(
        text, v1, v2, n_speakers, cfg
    )
    
    for sr, audio_chunk_np in audio_generator:
        if sr is not None and audio_chunk_np is not None and len(audio_chunk_np) > 0:
            all_audio_chunks.append(audio_chunk_np)
            sample_rate = sr
        elif vibevoice_chat_instance.stop_generation:
            break # Stop if generation was interrupted
    
    if not all_audio_chunks:
        return Response(b"", mimetype='audio/wav') # Empty response if no audio

    final_audio_array = np.concatenate(all_audio_chunks)
    
    # Save to a temporary file and return its content as a WAV response
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        sf.write(tmp_file.name, final_audio_array, sample_rate)
        tmp_file.seek(0)
        audio_content = tmp_file.read()
        os.remove(tmp_file.name) # Clean up temp file
        
    return Response(audio_content, mimetype='audio/wav')

# --- Argument Parsing for Flask Server ---
def parse_args():
    parser = argparse.ArgumentParser(description="VibeVoice TTS Flask Server")
    parser.add_argument(
        "--model_path",
        type=str,
        default="microsoft/VibeVoice-1.5B",
        help="Path to the VibeVoice model (e.g., 'microsoft/VibeVoice-1.5B' or a local path)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference ('cuda' or 'cpu')",
    )
    parser.add_argument(
        "--inference_steps",
        type=int,
        default=5,
        help="Number of DDPM inference steps (lower = faster, higher = better quality)",
    )
    parser.add_argument(
        "--default_voice",
        type=str,
        default=None,
        help="Path to a default voice audio file (.wav, .mp3, etc.) to load and select initially.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address for the Flask server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=13000,
        help="Port for the Flask server.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable Flask debug mode.",
    )
    return parser.parse_args()

# --- Main Function to Start Flask Server ---
if __name__ == "__main__":
    args = parse_args()

    # Set seed for reproducibility if needed
    set_seed(42)

    # Initialize VibeVoice only once when the server starts
    initialize_vibevoice_instance(
        model_path=args.model_path,
        device=args.device,
        inference_steps=args.inference_steps,
        default_voice=args.default_voice
    )

    print(f"Starting Flask server on http://{args.host}:{args.port}")
    print(f"Model path: {args.model_path}")
    print(f"Device: {args.device}")
    print(f"Inference steps: {args.inference_steps}")
    print(f"Default voice path: {args.default_voice if args.default_voice else 'Not set'}")
    print(f"Default voice presets: {list(vibevoice_chat_instance.available_voices.keys())}")
    print(f"Default parameters: {default_args}")


    # Run Flask app
    # Use threaded=True to handle multiple requests concurrently if needed,
    # but be mindful of VibeVoice's state management if it's not thread-safe.
    # The current implementation uses global state, so concurrent generation might be tricky.
    # For simplicity, we'll run in debug mode which usually handles threading.
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)