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

class VibeVoiceChat:
    def __init__(self, model_path: str, device: str = "cuda", inference_steps: int = 5, default_voice_path: str = None):
        """Initialize the VibeVoice chat model."""
        self.model_path = model_path
        self.device = device if torch.cuda.is_available() else "cpu"
        self.inference_steps = inference_steps
        self.is_generating = False
        self.stop_generation = False
        self.current_streamer = None
        self.default_voice_path = default_voice_path

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

        # --- Process default voice before loading model ---
        self.processed_default_voice_sample = None
        if self.default_voice_path and Path(self.default_voice_path).exists():
            print(f"Processing default voice from: {self.default_voice_path}")
            try:
                self.processed_default_voice_sample = self.read_audio(self.default_voice_path)
                print("✓ Default voice processed successfully.")
            except Exception as e:
                print(f"Warning: Failed to process default voice '{self.default_voice_path}': {e}")
                self.processed_default_voice_sample = None
        elif self.default_voice_path:
            print(f"Warning: Default voice path '{self.default_voice_path}' does not exist or is invalid.")
            self.processed_default_voice_sample = None

        self.load_model()
        self.setup_voice_presets()

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

    def generate_audio_stream(
        self,
        message: str,
        history: list,
        voice_1: str,
        voice_2: str,
        num_speakers: int,
        cfg_scale: float
    ) -> Iterator[tuple]:
        """Generate audio stream from text input."""
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

            start_time = time.time()

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

                # --- FIX FOR ValueError ---
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
                # --- END FIX ---

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
                if torch.cuda.is_available():
                    mem_alloc = torch.cuda.memory_allocated() / 1e9
                    mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
                    print(f"GPU memory allocated: {mem_alloc:.2f} GB / {mem_total:.2f} GB")
            else:
                print("Inputs remain on CPU.")

            audio_streamer = AudioStreamer(
                batch_size=1,
                stop_signal=None,
                timeout=None
            )

            self.current_streamer = audio_streamer

            generation_thread = threading.Thread(
                target=self._generate_with_streamer,
                args=(inputs, cfg_scale, audio_streamer)
            )
            generation_thread.start()

            time.sleep(1.5)

            sample_rate = 24000
            audio_stream_iter = audio_streamer.get_stream(0)

            all_audio_chunks = []
            chunk_count = 0

            print("Streaming audio chunks...")
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

                audio_16bit = self.convert_to_16_bit_wav(audio_np)
                all_audio_chunks.append(audio_16bit)

                if all_audio_chunks:
                    complete_audio_for_yield = np.concatenate(all_audio_chunks)
                    yield (sample_rate, complete_audio_for_yield)

            generation_thread.join(timeout=10.0)
            if generation_thread.is_alive():
                print("Warning: Generation thread did not finish within timeout.")

            if all_audio_chunks:
                complete_audio = np.concatenate(all_audio_chunks)
                generation_time = time.time() - start_time
                audio_duration = len(complete_audio) / sample_rate
                print(f"✓ Generation complete:")
                print(f"  Time taken: {generation_time:.2f} seconds")
                print(f"  Audio duration: {audio_duration:.2f} seconds")
                if generation_time > 0:
                    print(f"  Real-time factor: {audio_duration/generation_time:.2f}x")
                yield (sample_rate, complete_audio)

            print("Generation stream finished.")
            self.current_streamer = None
            self.is_generating = False

        except Exception as e:
            print(f"Error in generate_audio_stream: {e}")
            import traceback
            traceback.print_exc()
            self.is_generating = False
            self.current_streamer = None
            yield None

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


# --- Gradio Interface Creation ---
def load_text_from_file(file_path: str | None) -> str:
    """Loads text content from a file."""
    if file_path and Path(file_path).exists():
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            print(f"Loaded content from default prompt file: {file_path}")
            return content
        except Exception as e:
            print(f"Error loading default prompt file '{file_path}': {e}")
            return ""
    elif file_path:
        print(f"Warning: Default prompt file '{file_path}' not found.")
        return ""
    return ""

# Add initial_prompt and initial_cfg_scale parameters
def create_chat_interface(chat_instance: VibeVoiceChat, initial_prompt: str = "", initial_cfg_scale: float = 1.3):
    """Create a simplified Gradio ChatInterface for VibeVoice."""

    voice_options = list(chat_instance.available_voices.keys())
    if not voice_options:
        voice_options = ["Default"]

    default_voice_selection = "Default"
    if chat_instance.default_voice_path:
        default_voice_stem = Path(chat_instance.default_voice_path).stem
        if default_voice_stem in chat_instance.available_voices:
            default_voice_selection = default_voice_stem
        else:
            print(f"Warning: Default voice file '{chat_instance.default_voice_path}' stem ('{default_voice_stem}') not found in available voices. Using fallback.")
    elif voice_options and voice_options[0] != "Default":
        default_voice_selection = voice_options[0]

    default_voice_2_selection = default_voice_selection
    if len(voice_options) > 1:
        potential_second_voice = voice_options[1]
        if potential_second_voice != default_voice_selection:
            default_voice_2_selection = potential_second_voice
        elif len(voice_options) > 2:
            default_voice_2_selection = voice_options[2]
    elif len(voice_options) == 1 and voice_options[0] != "Default":
        default_voice_2_selection = voice_options[0]


    def chat_fn(message: str, history: list, voice_1: str, voice_2: str, num_speakers: int, cfg_scale: float):
        """Process chat message and generate audio response. Returns a temporary file path."""
        if isinstance(message, dict):
            text = message.get("text", "")
        else:
            text = message

        if not text.strip():
            print("Received empty message in chat_fn.")
            return None

        try:
            print(f"Generating audio for: '{text[:50]}...'")
            audio_generator = chat_instance.generate_audio_stream(
                text, history, voice_1, voice_2, num_speakers, cfg_scale
            )

            final_audio_data = None
            for sample_rate, audio_array in audio_generator:
                if sample_rate is not None and audio_array is not None and len(audio_array) > 0:
                    final_audio_data = (sample_rate, audio_array)
                if chat_instance.stop_generation:
                    print("Stopping audio collection due to stop request.")
                    break

            if final_audio_data:
                sample_rate, audio_array = final_audio_data
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    sf.write(tmp_file.name, audio_array, sample_rate)
                    print(f"Audio saved to temporary file: {tmp_file.name}")
                    return tmp_file.name
            else:
                print("No audio data was generated.")
                return None

        except Exception as e:
            print(f"Error in chat_fn: {e}")
            import traceback
            traceback.print_exc()
            return None

    with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="purple"), fill_height=True) as interface:
        gr.Markdown("# 🎙️ VibeVoice Chat\nGenerate natural dialogue audio with AI voices")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Voice & Generation Settings")

                voice_1 = gr.Dropdown(
                    choices=voice_options,
                    value=default_voice_selection,
                    label="Voice 1",
                    info="Select voice for Speaker 0"
                )

                voice_2 = gr.Dropdown(
                    choices=voice_options,
                    value=default_voice_2_selection,
                    label="Voice 2",
                    info="Select voice for Speaker 1 (if using multiple speakers)"
                )

                num_speakers = gr.Slider(
                    minimum=1,
                    maximum=2,
                    value=2,
                    step=1,
                    label="Number of Speakers",
                    info="Number of speakers in the dialogue"
                )

                cfg_scale = gr.Slider(
                    minimum=1.0,
                    maximum=2.0,
                    value=initial_cfg_scale, # Use the passed value here
                    step=0.05,
                    label="CFG Scale",
                    info="Guidance strength (higher = more adherence to text)"
                )

                stop_button = gr.Button("🛑 Stop Generation", variant="stop", visible=False)

            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=400,
                    type="messages",
                    elem_id="chatbot"
                )

                msg = gr.Textbox(
                    label="Message",
                    placeholder="Type your message or paste a script...",
                    lines=3,
                    value=initial_prompt
                )

                audio_output = gr.Audio(
                    label="Generated Audio",
                    type="filepath",
                    autoplay=True,
                    visible=False
                )

                with gr.Row():
                    submit_button = gr.Button("🎵 Generate Audio", variant="primary")
                    clear_button = gr.Button("🗑️ Clear Chat")

                gr.Examples(
                    examples=[
                        "Hello! How are you doing today?",
                        "Speaker 0: Welcome to our podcast!\nSpeaker 1: Thanks for having me!",
                        "Tell me an interesting fact about space.",
                        "What's your favorite type of music and why?",
                        "Speaker 0: This is a test of the VibeVoice system.\nSpeaker 1: It sounds quite natural!",
                    ],
                    inputs=msg,
                    label="Example Prompts"
                )

        # --- Event Handlers with Corrected Button Updates ---
        def update_ui_after_generation(message, history, voice_1, voice_2, num_speakers, cfg_scale):
            """Processes the user message, triggers audio generation, and updates the UI."""
            if not message.strip():
                return history, None, gr.update(visible=False), "", {"interactive": True}, {"visible": False, "interactive": False}

            history = history or []
            history.append({"role": "user", "content": message})

            submit_button_update_args = {"interactive": False}
            stop_button_update_args = {"visible": True, "interactive": True}

            audio_path = chat_fn(message, history, voice_1, voice_2, num_speakers, cfg_scale)

            if audio_path:
                history.append({"role": "assistant", "content": f"🎵 Audio generated"})
                return history, audio_path, gr.update(visible=True), "", submit_button_update_args, stop_button_update_args
            else:
                history.append({"role": "assistant", "content": "❌ Failed to generate audio."})
                return history, None, gr.update(visible=False), "", submit_button_update_args, stop_button_update_args

        def reset_buttons_after_action():
            """Resets submit and stop buttons to their default states."""
            return {"interactive": True}, {"visible": False, "interactive": False}

        submit_event = submit_button.click(
            fn=update_ui_after_generation,
            inputs=[msg, chatbot, voice_1, voice_2, num_speakers, cfg_scale],
            outputs=[chatbot, audio_output, audio_output, msg, submit_button, stop_button],
            queue=True
        ).then(
            fn=reset_buttons_after_action,
            outputs=[submit_button, stop_button]
        )

        msg_submit_event = msg.submit(
            fn=update_ui_after_generation,
            inputs=[msg, chatbot, voice_1, voice_2, num_speakers, cfg_scale],
            outputs=[chatbot, audio_output, audio_output, msg, submit_button, stop_button],
            queue=True
        ).then(
            fn=reset_buttons_after_action,
            outputs=[submit_button, stop_button]
        )

        clear_button.click(
            lambda: ([], None, gr.update(visible=False), "", {"interactive": True}, {"visible": False, "interactive": False}),
            outputs=[chatbot, audio_output, audio_output, msg, submit_button, stop_button]
        )

        stop_event = stop_button.click(
            fn=chat_instance.stop_audio_generation,
            api_name="stop_generation"
        ).then(
            fn=reset_buttons_after_action,
            outputs=[submit_button, stop_button]
        )

    return interface

# --- Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description="VibeVoice Chat Interface")
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
        "--default_prompt_file",
        type=str,
        default=None,
        help="Path to a text file (.txt) to load as the initial prompt in the message box.",
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=1.3,
        help="Classifier-Free Guidance scale (higher = more adherence to text, default: 1.3)",
    )
    return parser.parse_args()

# --- Main Function ---
def main():
    """Main function to parse arguments, initialize VibeVoiceChat, and launch the Gradio interface."""
    args = parse_args()

    set_seed(42)

    print("🎙️ Initializing VibeVoice Chat Interface...")

    initial_prompt_content = load_text_from_file(args.default_prompt_file)
    print(f"Initial prompt content to be loaded: '{initial_prompt_content[:100]}...'")
    
    if args.default_prompt_file and initial_prompt_content:
        print(f"✓ Successfully loaded default prompt from: {args.default_prompt_file}")
    elif args.default_prompt_file and not initial_prompt_content:
        print(f"✗ Failed to load default prompt from: {args.default_prompt_file} (file might be empty or inaccessible)")

    # --- Initialize VibeVoiceChat instance ---
    try:
        chat_instance = VibeVoiceChat(
            model_path=args.model_path,
            device=args.device,
            inference_steps=args.inference_steps,
            default_voice_path=args.default_voice
        )
    except Exception as e:
        print(f"Fatal Error: Could not initialize VibeVoiceChat. {e}")
        exit(1)

    # --- Decide whether to launch Gradio or run generation directly ---
    if args.default_prompt_file and initial_prompt_content:
        print("\n--- Running Generation Directly (No Gradio UI) ---")
        prompt_text = initial_prompt_content
        cfg_scale_direct_gen = args.cfg_scale
        
        available_voices_list = list(chat_instance.available_voices.keys())
        selected_voice_1 = available_voices_list[0] if available_voices_list and available_voices_list[0] != "Default" else "Default"
        selected_voice_2 = available_voices_list[1] if len(available_voices_list) > 1 and available_voices_list[1] != "Default" else selected_voice_1

        print(f"Generating audio with prompt: '{prompt_text[:100]}...'")
        print(f"Using Voice 1: '{selected_voice_1}', Voice 2: '{selected_voice_2}'")
        print(f"Using CFG Scale: {cfg_scale_direct_gen}")

        output_filename = f"generated_audio_{int(time.time())}.wav"
        output_dir = Path("./generated_audio")
        output_dir.mkdir(exist_ok=True)
        output_filepath = output_dir / output_filename

        try:
            audio_generator = chat_instance.generate_audio_stream(
                prompt_text, [], selected_voice_1, selected_voice_2, num_speakers=2, cfg_scale=cfg_scale_direct_gen
            )
            
            final_audio_data = None
            for sample_rate, audio_array in audio_generator:
                if sample_rate is not None and audio_array is not None and len(audio_array) > 0:
                    final_audio_data = (sample_rate, audio_array)
                if chat_instance.stop_generation:
                    print("Generation stopped prematurely.")
                    break

            if final_audio_data:
                sample_rate, audio_array = final_audio_data
                sf.write(output_filepath, audio_array, sample_rate)
                print(f"✓ Audio successfully generated and saved to: {output_filepath}")
                
                # --- PLAYBACK LOGIC ---
                try:
                    # Initialize Pygame Mixer
                    pygame.mixer.init()
                    print("Pygame mixer initialized.")
                    
                    # Load the audio file
                    audio_sound = pygame.mixer.Sound(output_filepath)
                    print(f"Audio file '{output_filepath}' loaded into Pygame Sound object.")
                    
                    # Play the audio
                    audio_sound.play()
                    print("Playing audio...")
                    
                    # Keep the script running while audio plays
                    while pygame.mixer.get_busy():
                        pygame.time.Clock().tick(10) # Small delay to prevent high CPU usage

                    print("Audio playback finished.")
                    
                except pygame.error as pe:
                    print(f"Pygame error during playback: {pe}")
                    print("Please ensure Pygame is installed (`pip install pygame`) and your audio drivers are working correctly.")
                except FileNotFoundError:
                    print(f"Error: Audio file not found at {output_filepath} for playback.")
                except Exception as e:
                    print(f"An unexpected error occurred during Pygame playback: {e}")
                finally:
                    # Quit the mixer cleanly
                    if pygame.mixer.get_init():
                        pygame.mixer.quit()
                        print("Pygame mixer quit.")
                # --- END PLAYBACK LOGIC ---

            else:
                print("✗ Audio generation failed, no audio data produced.")

        except Exception as e:
            print(f"Error during direct audio generation: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if chat_instance.current_streamer:
                 try:
                     chat_instance.current_streamer.end()
                 except:
                     pass
            print("--- Direct Generation Finished ---")
            sys.exit(0)
            
    else: # If no prompt file is provided, launch Gradio UI
        print("\n--- Launching Gradio UI ---")
        try:
            # Pass the initial CFG scale to create_chat_interface
            interface = create_chat_interface(chat_instance, initial_prompt=initial_prompt_content, initial_cfg_scale=args.cfg_scale)
        except Exception as e:
            print(f"Fatal Error creating Gradio interface: {e}")
            exit(1)

        print(f"🚀 Launching chat interface...")
        print(f"   Model: {args.model_path}")
        print(f"   Device: {chat_instance.device}")
        print(f"   Inference steps: {args.inference_steps}")
        print(f"   Default voice path: {args.default_voice if args.default_voice else 'Not set'}")
        print(f"   Default prompt file: {args.default_prompt_file if args.default_prompt_file else 'Not set'}")
        print(f"   Default CFG Scale: {args.cfg_scale}")
        print(f"   Available voices: {len(chat_instance.available_voices)}")

        if chat_instance.device == "cpu":
            print("\n⚠️  WARNING: Running on CPU - generation will be VERY slow!")
            print("   For faster generation, ensure you have:")
            print("   1. NVIDIA GPU with CUDA support")
            print("   2. PyTorch with CUDA installed (check PyTorch website for correct command)")
            print("   3. VibeVoice installed with flash-attn support.")

        interface.queue(max_size=10).launch(
            show_error=True,
            quiet=False,
            share=False
        )

if __name__ == "__main__":
    main()