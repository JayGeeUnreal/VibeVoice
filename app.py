"""
VibeVoice with Fahd Mirza
"""
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
# Clone and setup VibeVoice if not already present
vibevoice_dir = Path('./VibeVoice')
if not vibevoice_dir.exists():
    print("Cloning VibeVoice repository...")
    subprocess.run(['git', 'clone', 'https://github.com/vibevoice-community/VibeVoice'], check=True)
    print("Installing VibeVoice...")
    subprocess.run(['pip', 'install', '-e', './VibeVoice'], check=True)
    print("Installing wheel (required for flash-attn)...")
    subprocess.run(['pip', 'install', 'wheel'], check=True)
    print("Installing flash-attn...")
    try:
        subprocess.run(['pip', 'install', 'flash-attn', '--no-build-isolation'], check=True)
    except subprocess.CalledProcessError:
        print("Warning: flash-attn installation failed. Continuing without it...")
# Add the VibeVoice directory to path
import sys
sys.path.insert(0, str(vibevoice_dir))
# Import VibeVoice modules
try:
    from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
    from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
    from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
    from vibevoice.modular.streamer import AudioStreamer
except ImportError:
    try:
        import importlib.util
   
        def load_module(module_name, file_path):
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return module
     
        config_module = load_module(
            "vibevoice_config",
            vibevoice_dir / "modular" / "configuration_vibevoice.py"
        )
        VibeVoiceConfig = config_module.VibeVoiceConfig
      
        model_module = load_module(
            "vibevoice_model",
            vibevoice_dir / "modular" / "modeling_vibevoice_inference.py"
        )
        VibeVoiceForConditionalGenerationInference = model_module.VibeVoiceForConditionalGenerationInference
      
        processor_module = load_module(
            "vibevoice_processor",
            vibevoice_dir / "processor" / "vibevoice_processor.py"
        )
        VibeVoiceProcessor = processor_module.VibeVoiceProcessor
       
        streamer_module = load_module(
            "vibevoice_streamer",
            vibevoice_dir / "modular" / "streamer.py"
        )
        AudioStreamer = streamer_module.AudioStreamer
       
    except Exception as e:
        raise ImportError(
            f"VibeVoice module not found. Error: {e}\n"
            "Please ensure VibeVoice is properly installed:\n"
            "git clone https://github.com/vibevoice-community/VibeVoice\n"
            "cd VibeVoice/\n"
            "pip install -e .\n"
        )
from transformers.utils import logging
from transformers import set_seed
logging.set_verbosity_info()
logger = logging.get_logger(__name__)
class VibeVoiceChat:
    def __init__(self, model_path: str, device: str = "cuda", inference_steps: int = 5):
        """Initialize the VibeVoice chat model."""
        self.model_path = model_path
        self.device = device if torch.cuda.is_available() else "cpu"
        self.inference_steps = inference_steps
        self.is_generating = False
        self.stop_generation = False
        self.current_streamer = None
        
        # Check GPU availability and CUDA version
        if torch.cuda.is_available():
            print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  PyTorch CUDA: {torch.cuda.is_available()}")
            # Set memory fraction to avoid OOM
            torch.cuda.set_per_process_memory_fraction(0.95)
            # Enable TF32 for faster computation on Ampere GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        else:
            print("✗ No GPU detected, using CPU (generation will be VERY slow)")
            print("  For faster generation, ensure CUDA is properly installed")
        
        self.load_model()
        self.setup_voice_presets()
        
    def load_model(self):
        """Load the VibeVoice model and processor."""
        print(f"Loading model from {self.model_path}")
        start_time = time.time()
        
        self.processor = VibeVoiceProcessor.from_pretrained(self.model_path)
        
        if torch.cuda.is_available():
            print("Loading model with GPU acceleration...")
            try:
                self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.bfloat16,
                    device_map='cuda:0',
                    attn_implementation="flash_attention_2",
                    low_cpu_mem_usage=True,
                )
                print("✓ Flash Attention 2 enabled for faster generation")
            except Exception as e:
                print(f"Warning: Could not load with flash_attention_2: {e}")
                print("Falling back to standard attention...")
                self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.bfloat16,
                    device_map='cuda:0',
                    low_cpu_mem_usage=True,
                )
        else:
            print("Loading model on CPU (this will be slow)...")
            self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32,
                device_map='cpu',
                low_cpu_mem_usage=True,
            )
        
        self.model.eval()
        
        # Configure noise scheduler for faster inference
        self.model.model.noise_scheduler = self.model.model.noise_scheduler.from_config(
            self.model.model.noise_scheduler.config, 
            algorithm_type='sde-dpmsolver++',
            beta_schedule='squaredcos_cap_v2'
        )
        self.model.set_ddpm_inference_steps(num_steps=self.inference_steps)
        
        load_time = time.time() - start_time
        print(f"✓ Model loaded in {load_time:.2f} seconds")
        
        # Print model device
        if hasattr(self.model, 'device'):
            print(f"Model device: {self.model.device}")
    
    def setup_voice_presets(self):
        """Setup voice presets from the voices directory."""
        voices_dir = os.path.join(os.path.dirname(__file__), "voices")
        
        # Create voices directory if it doesn't exist
        if not os.path.exists(voices_dir):
            os.makedirs(voices_dir)
            print(f"Created voices directory at {voices_dir}")
            print("Please add voice sample files (.wav, .mp3, etc.) to this directory")
        
        self.available_voices = {}
        audio_extensions = ('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac')
        
        # Scan for audio files
        for file in os.listdir(voices_dir):
            if file.lower().endswith(audio_extensions):
                name = os.path.splitext(file)[0]
                self.available_voices[name] = os.path.join(voices_dir, file)
        
        # Sort voices alphabetically
        self.available_voices = dict(sorted(self.available_voices.items()))
        
        if not self.available_voices:
            print(f"Warning: No voice files found in {voices_dir}")
            print("Using default (zero) voice samples. Add audio files to the voices directory for better results.")
            # Add a default "None" option
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
            return np.zeros(24000)  # Return 1 second of silence as fallback
    
    def format_script(self, message: str, num_speakers: int = 2) -> str:
        """Format input message into a script with speaker assignments."""
        lines = message.strip().split('\n')
        formatted_lines = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Check if already formatted
            if line.startswith('Speaker ') and ':' in line:
                formatted_lines.append(line)
            else:
                # Auto-assign speakers in rotation
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
            
            # Validate inputs
            if not message.strip():
                yield None
                return
            
            # Format the script
            formatted_script = self.format_script(message, num_speakers)
            print(f"Formatted script:\n{formatted_script}")
            print(f"Using device: {self.device}")
            
            # Start timing
            start_time = time.time()
            
            # Select voices based on number of speakers
            selected_voices = []
            if voice_1 and voice_1 != "Default":
                selected_voices.append(voice_1)
            if num_speakers > 1 and voice_2 and voice_2 != "Default":
                selected_voices.append(voice_2)
            
            # Load voice samples
            voice_samples = []
            for i in range(num_speakers):
                # Use the appropriate voice for each speaker
                if i < len(selected_voices):
                    voice_name = selected_voices[i]
                    if voice_name in self.available_voices and self.available_voices[voice_name]:
                        audio_data = self.read_audio(self.available_voices[voice_name])
                    else:
                        audio_data = np.zeros(24000)  # Default silence
                else:
                    # Use first voice or default if not enough voices selected
                    if selected_voices and selected_voices[0] in self.available_voices and self.available_voices[selected_voices[0]]:
                        audio_data = self.read_audio(self.available_voices[selected_voices[0]])
                    else:
                        audio_data = np.zeros(24000)  # Default silence
                
                voice_samples.append(audio_data)
            
            print(f"Loaded {len(voice_samples)} voice samples")
            
            # Process inputs
            inputs = self.processor(
                text=[formatted_script],
                voice_samples=[voice_samples],
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )
            
            # Move to device and ensure correct dtype
            if self.device == "cuda":
                inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
                print(f"✓ Inputs moved to GPU")
                # Check GPU memory
                if torch.cuda.is_available():
                    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            
            # Create audio streamer
            audio_streamer = AudioStreamer(
                batch_size=1,
                stop_signal=None,
                timeout=None
            )
            
            self.current_streamer = audio_streamer
            
            # Start generation in separate thread
            generation_thread = threading.Thread(
                target=self._generate_with_streamer,
                args=(inputs, cfg_scale, audio_streamer)
            )
            generation_thread.start()
            
            # Wait briefly for generation to start
            time.sleep(1)
            
            # Stream audio chunks
            sample_rate = 24000
            audio_stream = audio_streamer.get_stream(0)
            
            all_audio_chunks = []
            chunk_count = 0
            
            for audio_chunk in audio_stream:
                if self.stop_generation:
                    audio_streamer.end()
                    break
                
                chunk_count += 1
                
                # Convert to numpy
                if torch.is_tensor(audio_chunk):
                    if audio_chunk.dtype == torch.bfloat16:
                        audio_chunk = audio_chunk.float()
                    audio_np = audio_chunk.cpu().numpy().astype(np.float32)
                else:
                    audio_np = np.array(audio_chunk, dtype=np.float32)
                
                # Ensure 1D
                if len(audio_np.shape) > 1:
                    audio_np = audio_np.squeeze()
                
                # Convert to 16-bit
                audio_16bit = self.convert_to_16_bit_wav(audio_np)
                all_audio_chunks.append(audio_16bit)
                
                # Yield accumulated audio
                if all_audio_chunks:
                    complete_audio = np.concatenate(all_audio_chunks)
                    yield (sample_rate, complete_audio)
            
            # Wait for generation to complete
            generation_thread.join(timeout=5.0)
            
            # Final yield with complete audio
            if all_audio_chunks:
                complete_audio = np.concatenate(all_audio_chunks)
                generation_time = time.time() - start_time
                audio_duration = len(complete_audio) / sample_rate
                print(f"✓ Generation complete:")
                print(f"  Time taken: {generation_time:.2f} seconds")
                print(f"  Audio duration: {audio_duration:.2f} seconds")
                print(f"  Real-time factor: {audio_duration/generation_time:.2f}x")
                yield (sample_rate, complete_audio)
            
            self.current_streamer = None
            self.is_generating = False
            
        except Exception as e:
            print(f"Error in generation: {e}")
            import traceback
            traceback.print_exc()
            self.is_generating = False
            self.current_streamer = None
            yield None
    
    def _generate_with_streamer(self, inputs, cfg_scale, audio_streamer):
        """Helper method to run generation with streamer."""
        try:
            def check_stop():
                return self.stop_generation
            
            # Use torch.cuda.amp for mixed precision if available
            if self.device == "cuda" and torch.cuda.is_available():
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
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
        except Exception as e:
            print(f"Error in generation thread: {e}")
            import traceback
            traceback.print_exc()
            audio_streamer.end()
    
    def convert_to_16_bit_wav(self, data):
        """Convert audio data to 16-bit WAV format."""
        if torch.is_tensor(data):
            data = data.detach().cpu().numpy()
        
        data = np.array(data)
        
        if np.max(np.abs(data)) > 1.0:
            data = data / np.max(np.abs(data))
        
        data = (data * 32767).astype(np.int16)
        return data
    
    def stop_audio_generation(self):
        """Stop the current audio generation."""
        self.stop_generation = True
        if self.current_streamer:
            try:
                self.current_streamer.end()
            except:
                pass
def create_chat_interface(chat_instance: VibeVoiceChat):
    """Create a simplified Gradio ChatInterface for VibeVoice."""
    
    # Get available voices
    voice_options = list(chat_instance.available_voices.keys())
    if not voice_options:
        voice_options = ["Default"]
    
    default_voice_1 = voice_options[0] if len(voice_options) > 0 else "Default"
    default_voice_2 = voice_options[1] if len(voice_options) > 1 else voice_options[0]
    
    # Define the chat function that returns audio
    def chat_fn(message: str, history: list, voice_1: str, voice_2: str, num_speakers: int, cfg_scale: float):
        """Process chat message and generate audio response."""
        
        # Extract text from message
        if isinstance(message, dict):
            text = message.get("text", "")
        else:
            text = message
        
        if not text.strip():
            return ""
        
        try:
            # Generate audio stream
            audio_generator = chat_instance.generate_audio_stream(
                text, history, voice_1, voice_2, num_speakers, cfg_scale
            )
            
            # Collect all audio data
            audio_data = None
            for audio_chunk in audio_generator:
                if audio_chunk is not None:
                    audio_data = audio_chunk
            
            # Return audio file path or error message
            if audio_data:
                # Save audio to temporary file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    sample_rate, audio_array = audio_data
                    sf.write(tmp_file.name, audio_array, sample_rate)
                    # Return the file path directly
                    return tmp_file.name
            else:
                return "Failed to generate audio"
            
        except Exception as e:
            print(f"Error in chat_fn: {e}")
            import traceback
            traceback.print_exc()
            return f"Error: {str(e)}"
    
    # Create the interface using Blocks for more control
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="purple"), fill_height=True) as interface:
        gr.Markdown("# 🎙️ VibeVoice Chat\nGenerate natural dialogue audio with AI voices")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Voice & Generation Settings")
                
                voice_1 = gr.Dropdown(
                    choices=voice_options,
                    value=default_voice_1,
                    label="Voice 1",
                    info="Select voice for Speaker 0"
                )
                
                voice_2 = gr.Dropdown(
                    choices=voice_options,
                    value=default_voice_2,
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
                    value=1.3,
                    step=0.05,
                    label="CFG Scale",
                    info="Guidance strength (higher = more adherence to text)"
                )
            
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
                    lines=3
                )
                
                audio_output = gr.Audio(
                    label="Generated Audio",
                    type="filepath",
                    autoplay=True,
                    visible=False
                )
                
                with gr.Row():
                    submit = gr.Button("🎵 Generate Audio", variant="primary")
                    clear = gr.Button("🗑️ Clear")
                
                # Example messages
                gr.Examples(
                    examples=[
                        "Hello! How are you doing today?",
                        "Speaker 0: Welcome to our podcast!\nSpeaker 1: Thanks for having me!",
                        "Tell me an interesting fact about space.",
                        "What's your favorite type of music and why?",
                    ],
                    inputs=msg,
                    label="Example Messages"
                )
        
        # Set up event handlers
        def process_and_display(message, history, voice_1, voice_2, num_speakers, cfg_scale):
            """Process message and update both chatbot and audio."""
            # Add user message to history
            history = history or []
            history.append({"role": "user", "content": message})
            
            # Generate audio
            audio_path = chat_fn(message, history, voice_1, voice_2, num_speakers, cfg_scale)
            
            # Add assistant response with audio
            if audio_path and audio_path.endswith('.wav'):
                history.append({"role": "assistant", "content": f"🎵 Audio generated successfully"})
                return history, audio_path, gr.update(visible=True), ""
            else:
                history.append({"role": "assistant", "content": audio_path or "Failed to generate audio"})
                return history, None, gr.update(visible=False), ""
        
        submit.click(
            fn=process_and_display,
            inputs=[msg, chatbot, voice_1, voice_2, num_speakers, cfg_scale],
            outputs=[chatbot, audio_output, audio_output, msg],
            queue=True
        )
        
        msg.submit(
            fn=process_and_display,
            inputs=[msg, chatbot, voice_1, voice_2, num_speakers, cfg_scale],
            outputs=[chatbot, audio_output, audio_output, msg],
            queue=True
        )
        
        clear.click(lambda: ([], None, gr.update(visible=False)), outputs=[chatbot, audio_output, audio_output])
    
    return interface
def parse_args():
    parser = argparse.ArgumentParser(description="VibeVoice Chat Interface")
    parser.add_argument(
        "--model_path",
        type=str,
        default="microsoft/VibeVoice-1.5B",
        help="Path to the VibeVoice model",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference",
    )
    parser.add_argument(
        "--inference_steps",
        type=int,
        default=5,
        help="Number of DDPM inference steps (lower = faster, higher = better quality)",
    )
    
    return parser.parse_args()
def main():
    """Main function to run the chat interface."""
    args = parse_args()
    
    set_seed(42)
    
    print("🎙️ Initializing VibeVoice Chat Interface...")
    
    # Initialize chat instance
    chat_instance = VibeVoiceChat(
        model_path=args.model_path,
        device=args.device,
        inference_steps=args.inference_steps
    )
    
    # Create interface
    interface = create_chat_interface(chat_instance)
    
    print(f"🚀 Launching chat interface")
    print(f"📁 Model: {args.model_path}")
    print(f"💻 Device: {chat_instance.device}")
    print(f"🔢 Inference steps: {args.inference_steps}")
    print(f"🎭 Available voices: {len(chat_instance.available_voices)}")
    
    if chat_instance.device == "cpu":
        print("\n⚠️  WARNING: Running on CPU - generation will be VERY slow!")
        print("   For faster generation, ensure you have:")
        print("   1. NVIDIA GPU with CUDA support")
        print("   2. PyTorch with CUDA installed: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    
    # Launch the interface
    interface.queue(max_size=10).launch(
        show_error=True,
        quiet=False,
    )
if __name__ == "__main__":
    main()