#!/usr/bin/env python3
"""
Enhanced microphone to STT using the actual TCCC pipeline with visible terminal window.
This script launches a visible window on the Jetson's connected monitor and
captures real audio with improved quality settings, processing it through the actual STT model.
"""

import os
import sys
import time
import numpy as np
import wave
import pyaudio
import threading
import collections
import subprocess
import logging
import shlex
import signal as os_signal
from datetime import datetime
from scipy import signal
try:
    import tkinter as tk
    from tkinter import font as tkFont
    HAS_TKINTER = True
except ImportError:
    HAS_TKINTER = False

# Import battlefield audio enhancer if available
try:
    from battlefield_audio_enhancer import BattlefieldAudioEnhancer
    HAS_BATTLEFIELD_ENHANCER = True
except ImportError:
    HAS_BATTLEFIELD_ENHANCER = False

# Import FullSubNet enhancer if available
try:
    from fullsubnet_integration.fullsubnet_enhancer import FullSubNetEnhancer
    HAS_FULLSUBNET_ENHANCER = True
except ImportError:
    HAS_FULLSUBNET_ENHANCER = False

# Set up paths
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(project_dir, 'src'))

# Force use of real STT (not mock)
os.environ["USE_MOCK_STT"] = "0"

# TCCC imports
from tccc.audio_pipeline import AudioPipeline
from tccc.stt_engine import create_stt_engine

# Configuration
OUTPUT_FILE = "improved_transcription.txt"
AUDIO_FILE = "improved_audio.wav"
HIGHQUALITY_AUDIO = "highquality_audio.wav"
LOG_FILE = "microphone_capture.log"
RECORDING_DURATION = 15  # seconds - increased for better visibility

# Enhancement options
ENHANCEMENT_MODE = "auto"  # Options: "battlefield", "fullsubnet", "both", "none", "auto"

# Enhanced audio settings
SAMPLE_RATE = 44100      # Higher sample rate for better quality (will be resampled for STT)
FORMAT = pyaudio.paInt16 # 16-bit for good quality
CHANNELS = 1             # Mono for better STT compatibility
CHUNK_SIZE = 1024        # Increased for more stable buffer processing
CALIBRATION_TIME = 2     # Seconds to calibrate noise levels

# Enhancement configurations
BATTLEFIELD_CONFIG = {
    "audio": {
        "sample_rate": SAMPLE_RATE,
        "channels": CHANNELS,
        "format": "int16",
        "chunk_size": CHUNK_SIZE
    },
    "battlefield_filtering": {
        "enabled": True,
        "outdoor_mode": True,
        "transient_protection": True,
        "distance_compensation": True
    },
    "voice_isolation": {
        "enabled": True,
        "strength": 0.8,
        "focus_width": 200,
        "voice_boost_db": 6
    }
}

FULLSUBNET_CONFIG = {
    "fullsubnet": {
        "enabled": True,
        "use_gpu": True,
        "sample_rate": 16000,
        "chunk_size": CHUNK_SIZE,
        "n_fft": 512,
        "hop_length": 256,
        "win_length": 512,
        "normalized_input": True,
        "normalized_output": True,
        "gpu_acceleration": True,
        "fallback_to_cpu": True
    }
}

# XTerm window settings
XTERM_TITLE = "TCCC Audio Capture"
XTERM_GEOMETRY = "120x40+0+0"  # Width x Height + X + Y (full screen)
XTERM_FONT = "-fa 'Monospace' -fs 20"  # Larger font for visibility
XTERM_BG = "black"
XTERM_FG = "green"  # More visible on Jetson display

# Terminal window specifications for other terminals
TERM_FONTSIZE = 20
TERM_WIDTH = 120
TERM_HEIGHT = 40

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TCCC-MicCapture")

def apply_noise_reduction(audio_data, noise_profile, reduction_strength=0.7):
    """
    Apply spectral subtraction noise reduction.
    
    Args:
        audio_data: Input audio data (numpy array)
        noise_profile: Noise profile spectrum
        reduction_strength: Strength of noise reduction (0.0-1.0)
        
    Returns:
        Noise-reduced audio data
    """
    # Convert to frequency domain
    fft_data = np.fft.rfft(audio_data)
    magnitude = np.abs(fft_data)
    phase = np.angle(fft_data)
    
    # Apply spectral subtraction
    # Subtract noise profile from magnitude, ensuring it doesn't go below 0
    threshold = 0.01  # Minimum spectral floor
    magnitude = np.maximum(magnitude - reduction_strength * noise_profile, threshold * np.max(magnitude))
    
    # Reconstruct signal with reduced noise magnitude but original phase
    fft_reduced = magnitude * np.exp(1j * phase)
    
    # Convert back to time domain
    reduced_audio = np.fft.irfft(fft_reduced)
    
    # Ensure same length as input
    reduced_audio = reduced_audio[:len(audio_data)]
    
    return reduced_audio

def normalize_audio(audio_data, target_level=-18):
    """
    Normalize audio levels to target RMS level in dB.
    
    Args:
        audio_data: Input audio data (numpy array)
        target_level: Target RMS level in dB
        
    Returns:
        Normalized audio data
    """
    # Convert to float for processing
    if audio_data.dtype != np.float32:
        float_audio = audio_data.astype(np.float32) / 32767.0
    else:
        float_audio = audio_data.copy()
    
    # Calculate current RMS level
    rms = np.sqrt(np.mean(float_audio**2))
    if rms < 1e-8:  # Avoid log of zero
        return audio_data
    
    # Convert to dB
    current_db = 20 * np.log10(rms)
    
    # Calculate gain needed
    gain_db = target_level - current_db
    gain_linear = 10 ** (gain_db/20)
    
    # Apply gain
    normalized = float_audio * gain_linear
    
    # Prevent clipping
    max_val = np.max(np.abs(normalized))
    if max_val > 0.95:
        normalized = normalized * 0.95 / max_val
    
    # Convert back to original format
    if audio_data.dtype != np.float32:
        return (normalized * 32767.0).astype(audio_data.dtype)
    return normalized

def resample_audio(audio_data, src_rate, dst_rate):
    """
    Resample audio to a different sample rate.
    
    Args:
        audio_data: Input audio data (numpy array)
        src_rate: Source sample rate in Hz
        dst_rate: Destination sample rate in Hz
        
    Returns:
        Resampled audio data
    """
    # Calculate duration and create time arrays
    duration = len(audio_data) / src_rate
    src_time = np.arange(0, len(audio_data)) / src_rate
    dst_time = np.arange(0, duration, 1/dst_rate)
    
    # Resample using scipy's resample function
    num_samples = int(len(audio_data) * dst_rate / src_rate)
    resampled = signal.resample(audio_data, num_samples)
    
    return resampled

def launch_visible_terminal():
    """
    Launch a visible terminal window on the Jetson's monitor.
    Specifically handles SSH sessions to ensure display appears on Jetson's physical screen.
    
    Returns:
        - True if terminal was launched successfully
        - False if unable to launch terminal
    """
    # Get Jetson's display
    jetson_display = None
    
    # Try different methods to detect the Jetson's physical display
    try:
        # Method 1: Try to find the main display
        result = subprocess.run(['xrandr', '--listactivemonitors'], capture_output=True, text=True)
        if result.returncode == 0 and result.stdout:
            monitors = result.stdout.splitlines()
            if len(monitors) > 1:  # At least one monitor besides the heading
                logger.info(f"Detected monitors: {monitors}")
                jetson_display = ":0"  # Default X display
    except Exception as e:
        logger.warning(f"Error detecting displays with xrandr: {e}")
    
    # Method 2: Check if we have an active desktop session
    if not jetson_display:
        try:
            # Try to find active X sessions
            result = subprocess.run(['w', '-h'], capture_output=True, text=True)
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    if ':0' in line:
                        jetson_display = ":0"
                        logger.info(f"Found active X session: {line}")
                        break
        except Exception as e:
            logger.warning(f"Error detecting X sessions: {e}")
    
    # Method 3: Check who owns the X server
    if not jetson_display:
        try:
            result = subprocess.run(['ls', '-la', '/tmp/.X11-unix/'], capture_output=True, text=True)
            if result.returncode == 0 and 'X0' in result.stdout:
                jetson_display = ":0"
                logger.info("X server socket found in /tmp/.X11-unix/")
        except Exception as e:
            logger.warning(f"Error checking X server sockets: {e}")
    
    # If still no display found, try default
    if not jetson_display:
        jetson_display = ":0"
        logger.warning("No display detected, falling back to default :0")
    
    # Set up terminal commands with the detected display
    terminals = [
        f"DISPLAY={jetson_display} xterm -T '{XTERM_TITLE}' -fs 20 -geometry {XTERM_GEOMETRY} -bg {XTERM_BG} -fg {XTERM_FG} -e",
        f"DISPLAY={jetson_display} gnome-terminal --title='{XTERM_TITLE}' --geometry={XTERM_GEOMETRY} --"
    ]
    
    # Add specific Jetson-compatible terminals
    try:
        # Check if lxterminal is available (common on lightweight systems)
        lxt = subprocess.run(['which', 'lxterminal'], capture_output=True)
        if lxt.returncode == 0:
            terminals.insert(0, f"DISPLAY={jetson_display} lxterminal -t '{XTERM_TITLE}' -e")
            
        # Check if sakura is available
        sakura = subprocess.run(['which', 'sakura'], capture_output=True)
        if sakura.returncode == 0:
            terminals.insert(0, f"DISPLAY={jetson_display} sakura -t '{XTERM_TITLE}' -e")
    except Exception:
        pass
    
    # Build the command to re-run this script with a flag indicating we're in the visible terminal
    script_path = os.path.abspath(__file__)
    args = sys.argv[1:] + ["--in-terminal"]
    cmd_args = ' '.join([shlex.quote(arg) for arg in args])
    
    # Log that we're trying to launch on the Jetson's display
    logger.info(f"Attempting to launch terminal on display {jetson_display}")
    
    # Try each terminal until one works
    for terminal_cmd in terminals:
        try:
            full_cmd = f"{terminal_cmd} python3 {script_path} {cmd_args}"
            logger.info(f"Launching terminal with: {full_cmd}")
            
            # Use different methods to ensure the window appears on Jetson's screen
            # First try with setsid to create a new session
            launch_cmd = f"setsid -w {full_cmd}"
            result = subprocess.run(launch_cmd, shell=True, stderr=subprocess.PIPE, text=True)
            
            # If there's an error, try without setsid
            if "command not found" in result.stderr or "No such file" in result.stderr:
                logger.info("Setsid not available, trying direct command")
                subprocess.Popen(full_cmd, shell=True)
            
            # Wait a bit to see if terminal launches
            time.sleep(1)
            return True
            
        except Exception as e:
            logger.warning(f"Failed to launch terminal with {terminal_cmd}: {e}")
    
    # Last resort: try launching directly with DISPLAY environment variable
    try:
        os.environ["DISPLAY"] = jetson_display
        subprocess.Popen(f"xterm -T 'TCCC Audio Capture' -fs 20 -bg black -fg green -e python3 {script_path} --in-terminal", shell=True)
        time.sleep(1)
        return True
    except Exception as e:
        logger.warning(f"Failed all terminal launch attempts: {e}")
        
    # Final fallback: Try using Tkinter if available to create a direct window
    if HAS_TKINTER:
        try:
            logger.info("Attempting to launch Tkinter window as fallback")
            # Create subprocess that will run tkinter window
            tk_cmd = f"DISPLAY={jetson_display} python3 -c \"import tkinter as tk; import os; import subprocess; root = tk.Tk(); root.title('TCCC AUDIO CAPTURE'); root.attributes('-fullscreen', True); root.configure(bg='black'); label = tk.Label(root, text='LAUNCHING AUDIO CAPTURE...\\n\\nPlease wait while the system initializes', fg='green', bg='black', font=('Courier', 36, 'bold')); label.pack(expand=True); root.after(1000, lambda: subprocess.Popen(['python3', '{script_path}', '--in-terminal'])); root.mainloop()\""
            
            subprocess.Popen(tk_cmd, shell=True)
            time.sleep(2)
            return True
        except Exception as e:
            logger.error(f"Failed to launch Tkinter window: {e}")
    
    return False

def display_audio_level(level, width=50, symbol='▓', empty='░', use_color=True):
    """
    Create a visual audio level meter with improved rendering and color.
    
    Args:
        level: Audio level (0.0-1.0)
        width: Width of the meter bar
        symbol: Character to use for the filled portion
        empty: Character to use for the empty portion
        use_color: Whether to use ANSI color codes
        
    Returns:
        String containing the visual meter
    """
    filled = int(level * width)
    meter = symbol * filled + empty * (width - filled)
    db_level = 20 * np.log10(level + 1e-10)  # Convert to dB for better display
    
    # Color coding based on levels (green-yellow-red)
    if use_color:
        if level < 0.3:  # Low level - green
            color_code = "\033[92m"  # Bright green
        elif level < 0.7:  # Medium level - yellow
            color_code = "\033[93m"  # Bright yellow
        else:  # High level - red
            color_code = "\033[91m"  # Bright red
        
        reset_code = "\033[0m"
        meter = f"{color_code}|{meter}|{reset_code}"
    else:
        meter = f"|{meter}|"
    
    # Add a visual indicator for voice detection
    voice_indicator = ""
    if level > 0.02:  # Voice detection threshold
        if use_color:
            voice_indicator = "\033[1;97;42m SPEAKING \033[0m"  # White on green background
        else:
            voice_indicator = "SPEAKING"
    
    return f"{meter} {db_level:.1f} dB ({level*100:.1f}%) {voice_indicator}"

def print_big_message(message, style="info"):
    """
    Print a VERY prominent message with extreme visibility for the Jetson's monitor.
    
    Args:
        message: The message to display
        style: Style of message - "info", "alert", "success", "warning", "start", "stop"
    """
    # Define ANSI color codes for different styles - using high contrast combinations
    styles = {
        "info": "\033[1;97;44m",     # White on blue
        "alert": "\033[1;97;41m",    # White on red
        "success": "\033[1;97;42m",  # White on green
        "warning": "\033[1;30;43m",  # Black on yellow
        "start": "\033[1;30;102m",   # Black on bright green
        "stop": "\033[1;97;101m"     # White on bright red
    }
    
    # Default to info if style not found
    style_code = styles.get(style, styles["info"])
    reset = "\033[0m"
    
    # Create a more attention-grabbing box
    width = len(message) + 12  # Extra padding for visibility
    box_top = "╔" + "═" * width + "╗"
    box_middle = "║" + " " * width + "║"
    box_bottom = "╚" + "═" * width + "╝"
    
    # For critical messages like START/STOP, make them SUPER obvious with multiple lines
    is_critical = style in ["start", "stop"]
    
    # Print the emphasized message
    print("\n" * (3 if is_critical else 1))  # More space for critical messages
    print(f"{style_code}{box_top}{reset}")
    print(f"{style_code}{box_middle}{reset}")
    
    # For critical messages, add extra emphasis
    if is_critical:
        # Add stars around critical messages
        stars = "*" * (width - len(message) - 6)
        half_stars = stars[:len(stars)//2]
        print(f"{style_code}║  {half_stars} {message} {half_stars}  ║{reset}")
    else:
        print(f"{style_code}║     {message}     ║{reset}")
    
    print(f"{style_code}{box_middle}{reset}")
    print(f"{style_code}{box_bottom}{reset}")
    print("\n" * (3 if is_critical else 1))
    
    # Also log the message
    logger.info(message)

def print_progress_bar(current, total, width=60, prefix='Progress:', suffix='Complete', 
                      fill='█', empty='░', use_color=True):
    """
    Displays a customizable progress bar with color and percentage.
    
    Args:
        current: Current progress value
        total: Total value (100%)
        width: Width of the progress bar in characters
        prefix: Text to display before the progress bar
        suffix: Text to display after the progress bar
        fill: Character to use for filled portion
        empty: Character to use for empty portion
        use_color: Whether to use ANSI color codes
    """
    percent = min(100, int(current / total * 100))
    filled_length = int(width * current // total)
    bar = fill * filled_length + empty * (width - filled_length)
    
    # Color based on progress
    if use_color:
        if percent < 33:
            color = "\033[92m"  # Green
        elif percent < 66:
            color = "\033[93m"  # Yellow
        else:
            color = "\033[91m"  # Red
        reset = "\033[0m"
    else:
        color = ""
        reset = ""
    
    # Print the progress bar
    print(f"\r{prefix} {color}|{bar}|{reset} {percent}% {suffix}", end='', flush=True)

def main():
    """Capture and process real microphone input through STT with enhanced audio quality."""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Enhanced microphone to STT with FullSubNet")
    parser.add_argument("--enhancement", type=str, choices=["battlefield", "fullsubnet", "both", "none", "auto"],
                      default="auto", help="Audio enhancement mode to use")
    parser.add_argument("--in-terminal", action="store_true", help="Running in terminal mode")
    args = parser.parse_args()
    
    # Override enhancement mode if specified
    global ENHANCEMENT_MODE
    if args.enhancement != "auto":
        ENHANCEMENT_MODE = args.enhancement
        print(f"Using {ENHANCEMENT_MODE} enhancement mode")
    
    # Check if running inside the launched terminal
    in_terminal = args.in_terminal or "--in-terminal" in sys.argv
    
    # Check for X server and display availability - even if DISPLAY variable is not set
    has_display = False
    
    # Check for display using multiple methods
    if os.path.exists('/tmp/.X11-unix/X0'):
        has_display = True
        logger.info("X11 socket detected at /tmp/.X11-unix/X0")
    
    # If we're not already in a terminal window and a display is available, try launching one
    if not in_terminal and (has_display or os.environ.get('DISPLAY')):
        logger.info("Launching visible terminal window on Jetson's physical monitor...")
        
        # Make sure the script can find the X server
        if not os.environ.get('DISPLAY') and has_display:
            os.environ['DISPLAY'] = ':0'
            logger.info(f"Setting DISPLAY environment variable to {os.environ['DISPLAY']}")
        
        # Also ensure XAUTHORITY is set if needed
        if not os.environ.get('XAUTHORITY') and os.path.exists('/run/user/1000/gdm/Xauthority'):
            os.environ['XAUTHORITY'] = '/run/user/1000/gdm/Xauthority'
            logger.info(f"Setting XAUTHORITY to {os.environ['XAUTHORITY']}")
            
        # Try launching a visible window
        launch_success = launch_visible_terminal()
        
        # If launch was successful
        if launch_success:
            logger.info("Visible window launched successfully! Monitor the Jetson's physical screen.")
            logger.info("This SSH session will now exit.")
            
            # Exit this instance as the visible window will run its own instance
            return
        else:
            # Direct fallback for truly desperate situations - create an EXTREMELY visible window
            # with big text announcing start/stop speaking commands
            try:
                if HAS_TKINTER and os.environ.get('DISPLAY'):
                    logger.info("Attempting emergency direct Tkinter window")
                    
                    # Start a separate process for the Tkinter window
                    direct_display_cmd = f"DISPLAY={os.environ.get('DISPLAY', ':0')} python3 -c \"import tkinter as tk; import subprocess; import threading; import time; import os; def start_audio_process(): subprocess.run(['python3', '{os.path.abspath(__file__)}', '--in-terminal']); root = tk.Tk(); root.title('TCCC SPEECH CAPTURE'); root.attributes('-fullscreen', True); root.configure(background='black'); title = tk.Label(root, text='TCCC SPEECH CAPTURE', font=('Arial', 48, 'bold'), fg='white', bg='black'); title.pack(pady=30); status_text = tk.StringVar(); status_text.set('STANDBY - CALIBRATING MICROPHONE'); status = tk.Label(root, textvariable=status_text, font=('Arial', 36), fg='yellow', bg='black'); status.pack(pady=20); instructions = tk.Label(root, text='Watch this window for speech capture status\\nSpeak clearly when prompted', font=('Arial', 24), fg='white', bg='black'); instructions.pack(pady=20); thread = threading.Thread(target=start_audio_process); thread.daemon = True; thread.start(); def update_status(): states = [('CALIBRATING - Please remain silent', 'yellow'), ('START SPEAKING NOW', 'green'), ('RECORDING YOUR SPEECH', 'green'), ('PLEASE CONTINUE SPEAKING', 'green'), ('STOP SPEAKING', 'red'), ('PROCESSING AUDIO', 'yellow')]; i = 0; while True: text, color = states[i % len(states)]; status_text.set(text); status.config(fg=color); if 'START' in text: root.configure(background='darkgreen'); title.config(bg='darkgreen'); instructions.config(bg='darkgreen'); status.config(bg='darkgreen'); elif 'STOP' in text: root.configure(background='darkred'); title.config(bg='darkred'); instructions.config(bg='darkred'); status.config(bg='darkred'); else: root.configure(background='black'); title.config(bg='black'); instructions.config(bg='black'); status.config(bg='black'); time.sleep(5); i += 1; status_thread = threading.Thread(target=update_status); status_thread.daemon = True; status_thread.start(); root.mainloop()\""
                    
                    # Launch window process
                    subprocess.Popen(direct_display_cmd, shell=True)
                    time.sleep(2)
                    return
            except Exception as e:
                logger.warning(f"Failed to launch direct Tkinter window: {e}")
                    
            logger.warning("Failed to launch all visible window options. Running in current terminal instead.")
    
    # Clear screen for better visibility (works in terminal window)
    if in_terminal:
        os.system('clear')
        # Also maximize terminal and set larger font if possible in terminal mode
        if in_terminal:
            try:
                # Set larger font size with escape sequences (works in many terminals)
                print("\033]50;xft:Monospace:pixelsize=18\007")
                # Attempt to maximize the window
                print("\033[9;1t")
            except:
                pass
        
    # Use big, colored messages in terminal for visibility
    if in_terminal:
        print_big_message("TCCC ENHANCED MICROPHONE CAPTURE", "info")
        print("\033[1m\033[97mThis program will record and transcribe your speech with enhanced quality\033[0m")
        print("\033[1m- Higher sample rate (44.1kHz)\n- Noise reduction\n- Level normalization\n- Real-time visual monitoring\033[0m\n")
    else:
        # Simpler output for SSH session
        print("\n===== ENHANCED Microphone to Text Transcription =====")
        print("This program will record and transcribe ACTUAL speech from your microphone")
        print("Audio quality improvements: Higher sample rate, noise reduction, normalization\n")
        logger.info("Running in SSH session - check the Jetson's screen if available")
    
    # Initialize STT engine
    print("Initializing STT engine... (this will take a moment)")
    stt_engine = create_stt_engine("faster-whisper")
    
    stt_config = {
        "model": {
            "size": "tiny",
            "compute_type": "int8",
            "vad_filter": False  # We'll use our enhanced VAD instead
        }
    }
    
    stt_engine.initialize(stt_config)
    print("STT engine initialized!")
    
    # Initialize audio enhancers based on selected mode
    battlefield_enhancer = None
    fullsubnet_enhancer = None
    
    # Determine which enhancers to use based on mode
    use_battlefield = ENHANCEMENT_MODE in ["battlefield", "both", "auto"]
    use_fullsubnet = ENHANCEMENT_MODE in ["fullsubnet", "both", "auto"]
    
    # Auto mode: use available enhancers with preference for FullSubNet
    if ENHANCEMENT_MODE == "auto":
        if HAS_FULLSUBNET_ENHANCER:
            use_battlefield = False  # Prefer FullSubNet in auto mode
        elif not HAS_BATTLEFIELD_ENHANCER:
            print("\nWarning: No audio enhancers available. Using raw audio.")
    
    # Initialize battlefield enhancer if needed
    if use_battlefield and HAS_BATTLEFIELD_ENHANCER:
        try:
            print("\nInitializing battlefield audio enhancer...")
            battlefield_enhancer = BattlefieldAudioEnhancer(BATTLEFIELD_CONFIG)
            print("Battlefield audio enhancer initialized with enhanced outdoor capabilities")
            print("- Adaptive gain control for varying distances")
            print("- Outdoor noise reduction optimized for clarity")
            print("- Enhanced voice activity detection for battlefield conditions")
        except Exception as e:
            logger.error(f"Failed to initialize battlefield audio enhancer: {e}")
            battlefield_enhancer = None
    
    # Initialize FullSubNet enhancer if needed
    if use_fullsubnet and HAS_FULLSUBNET_ENHANCER:
        try:
            print("\nInitializing FullSubNet speech enhancer...")
            fullsubnet_enhancer = FullSubNetEnhancer(FULLSUBNET_CONFIG)
            print("FullSubNet enhancer initialized with GPU acceleration")
            print("- Deep learning based speech enhancement")
            print("- Optimized for Nvidia Jetson hardware")
            
            # Check for CUDA availability
            cuda_status = "using CUDA" if fullsubnet_enhancer.use_gpu else "using CPU (slower)"
            print(f"- Hardware acceleration: {cuda_status}")
        except Exception as e:
            logger.error(f"Failed to initialize FullSubNet enhancer: {e}")
            fullsubnet_enhancer = None
    
    # Initialize PyAudio directly for better control over audio quality
    print("\nSetting up enhanced audio capture...")
    audio = pyaudio.PyAudio()
    
    # List available audio devices for better debugging
    print("\nAvailable audio input devices:")
    for i in range(audio.get_device_count()):
        dev_info = audio.get_device_info_by_index(i)
        if dev_info['maxInputChannels'] > 0:  # Only show input devices
            print(f"  Device {i}: {dev_info['name']}")
    
    # Using device 0 (first microphone) as detected earlier
    device_id = 0
    
    # Prepare output file
    with open(OUTPUT_FILE, 'w') as f:
        f.write("=== Enhanced Speech Transcription ===\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Add audio enhancement details
        f.write("Audio Quality Improvements:\n")
        f.write("- 44.1kHz Sample Rate for higher fidelity\n")
        f.write("- Multi-band noise reduction\n")
        f.write("- Adaptive level normalization\n")
        
        # Add enhancement details
        if battlefield_enhancer is not None:
            f.write("\nBattlefield Audio Enhancements:\n")
            f.write("- Adaptive gain control for varying distances\n")
            f.write("- Outdoor noise filtering optimized for clarity\n") 
            f.write("- Enhanced voice activity detection\n")
            f.write("- Wind noise reduction\n")
        
        if fullsubnet_enhancer is not None:
            f.write("\nFullSubNet Speech Enhancements:\n")
            f.write("- Deep learning based speech enhancement\n")
            f.write("- GPU-accelerated Nvidia processing\n")
            f.write("- Advanced noise suppression\n")
            f.write("- Trained on diverse speech and noise conditions\n")
        
        f.write("\nActual words spoken (no simulation):\n\n")
    
    # Create ring buffer for stable processing
    # This helps prevent choppy audio by maintaining a consistent buffer
    ring_buffer = collections.deque(maxlen=10)
    
    # Create arrays to store all audio for saving to file
    all_audio_data = np.array([], dtype=np.int16)
    highquality_audio = np.array([], dtype=np.int16)  # Store original high-quality audio
    
    # Buffer to accumulate audio for STT processing
    stt_buffer = np.array([], dtype=np.int16)
    stt_buffer_size = 16000  # 1 second at 16kHz for STT
    
    # Noise profile for noise reduction
    noise_profile = None
    
    # Instructions to user for calibration
    if in_terminal:
        print_big_message("AMBIENT NOISE CALIBRATION", "warning")
        print(f"\033[1mPlease remain SILENT for {CALIBRATION_TIME} seconds to measure background noise\033[0m")
    else:
        print("\n" + "=" * 60)
        print("AMBIENT NOISE CALIBRATION")
        print(f"Please remain silent for {CALIBRATION_TIME} seconds to measure background noise")
        print("=" * 60)
    
    # Start audio stream for calibration
    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        input_device_index=device_id,
        frames_per_buffer=CHUNK_SIZE
    )
    
    # Collect noise samples for calibration
    if in_terminal:
        print("\n\033[1;33mCALIBRATING...\033[0m")
    else:
        print("\nCalibrating...")
        
    noise_samples = []
    calib_start = time.time()
    
    while time.time() - calib_start < CALIBRATION_TIME:
        try:
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            audio_chunk = np.frombuffer(data, dtype=np.int16)
            noise_samples.append(audio_chunk)
            
            # Show noise level for user feedback
            level = np.max(np.abs(audio_chunk)) / 32767.0
            noise_meter = display_audio_level(level, use_color=in_terminal)
            sys.stdout.write(f"\rNoise Level: {noise_meter}")
            sys.stdout.flush()
            
        except Exception as e:
            error_msg = f"\nError during calibration: {e}"
            logger.error(error_msg)
            print(error_msg)
    
    # Calculate noise profile for spectral subtraction
    avg_noise_level = 0
    if noise_samples:
        noise_concat = np.concatenate(noise_samples)
        noise_fft = np.abs(np.fft.rfft(noise_concat))
        # Average the noise spectrum
        noise_profile = noise_fft / len(noise_fft)
        # Scale noise profile to match future FFT sizes
        target_size = CHUNK_SIZE // 2 + 1
        if len(noise_profile) > target_size:
            noise_profile = noise_profile[:target_size]
        elif len(noise_profile) < target_size:
            noise_profile = np.pad(noise_profile, (0, target_size - len(noise_profile)))
        
        # Calculate average noise level for adaptive thresholding
        avg_noise_level = np.sqrt(np.mean(noise_concat**2)) / 32767.0
        db_level = 20 * np.log10(avg_noise_level + 1e-10)
        
        if in_terminal:
            print(f"\n\033[1mAverage noise level: \033[93m{db_level:.1f} dB\033[0m")
        else:
            print(f"\nAverage noise level: {db_level:.1f} dB")
    
    # Clear screen in terminal mode before recording
    if in_terminal:
        time.sleep(1)
        os.system('clear')
    
    # Instructions to user for recording
    if in_terminal:
        print_big_message("GET READY TO SPEAK", "alert")
        print(f"\033[1;97mAudio will be recorded for \033[93m{RECORDING_DURATION} seconds\033[97m with enhanced quality\033[0m")
        print(f"\033[1;97mYour ACTUAL words will be transcribed in real-time\033[0m")
    else:
        print("\n" + "=" * 60)
        print("GET READY TO SPEAK")
        print(f"Audio will be recorded for {RECORDING_DURATION} seconds with enhanced quality")
        print("Your ACTUAL words will be transcribed (not simulated)")
        print("=" * 60)
    
    # Add visual separation for better visibility
    if in_terminal:
        print("\n\n" + "█" * 100 + "\n")
    
    # Countdown - make it VERY big and visible in terminal mode
    for i in range(5, 0, -1):
        if in_terminal:
            # Make countdown extra large and visible
            countdown_msg = f"STARTING IN {i}"
            print("\033[2J\033[H")  # Clear screen for maximum visibility
            
            # Print progress as blocks
            blocks = "█" * (5-i) + "░" * i
            
            # Use different colors based on how close we are to starting
            if i > 3:
                color = "\033[1;93m"  # Yellow
            elif i > 1:
                color = "\033[1;93;101m"  # Yellow on red
            else:
                color = "\033[1;97;101m"  # White on red
                
            # Extra large countdown display
            print("\n" * 5)
            print(f"{color}{countdown_msg.center(100)}\033[0m")
            print("\n" * 2)
            print(f"{color}{blocks.center(100)}\033[0m")
            print("\n" * 5)
            
            # Also print countdown at bottom of screen for visibility
            print("\n" * 10)
            print(f"{color}PREPARE TO SPEAK IN {i} SECONDS\033[0m".center(100))
            
        else:
            print(f"Starting in {i}...")
        time.sleep(1)
    
    # Start recording message - ULTRA visible for in_terminal mode
    if in_terminal:
        print("\033[2J\033[H")  # Clear screen
        print_big_message("START SPEAKING NOW", "start")
        
        # Add flashing effect for maximum visibility
        for _ in range(3):
            print("\033[?5h")  # Enable blinking
            time.sleep(0.3)
            print("\033[?5l")  # Disable blinking
            time.sleep(0.3)
        
        print("\033[1;32mRecording with ENHANCED audio quality\033[0m\n")
    else:
        print("\n>>> START SPEAKING NOW - Recording with ENHANCED audio quality <<<\n")
    
    # Main processing loop
    start_time = time.time()
    last_text = ""
    threshold_level = max(0.02, avg_noise_level * 2.5)  # Adaptive threshold based on noise
    
    # Keep track of quality metrics
    peak_level = 0.0
    avg_level = 0.0
    sample_count = 0
    
    # Create arrays to store all audio for saving to file
    all_audio_data = np.array([], dtype=np.int16)  # Processed audio for STT at 16kHz
    highquality_audio = np.array([], dtype=np.int16)  # Original unprocessed audio at 44.1kHz
    
    # Buffer for STT processing
    stt_buffer = np.array([], dtype=np.int16)
    stt_buffer_size = 16000  # 1 second at 16kHz for STT
    
    # Determine remaining recording time (for countdown display)
    remaining_time = RECORDING_DURATION
    
    # For terminal visibility - create a dedicated area for display
    if in_terminal:
        print("\033[2J\033[H")  # Clear screen for better visibility
        # Add space for meter display and progress bar
        for _ in range(12):
            print("")
        
        # Add dedicated area for transcription
        print("\033[1;97m" + "=" * 100 + "\033[0m")
        print("\033[1;96mTRANSCRIPTION RESULTS:\033[0m".center(100))
        print("\033[1;97m" + "=" * 100 + "\033[0m")
        
        # Add more blank space for transcription output
        for _ in range(8):
            print("")
    
    # Variables to track speaking state for UI feedback
    was_speaking = False
    speaking_duration = 0
    silence_duration = 0
    
    # Timing for visual warnings
    five_sec_warning_shown = False
    three_sec_warning_shown = False
    one_sec_warning_shown = False
    
    try:
        while time.time() - start_time < RECORDING_DURATION:
            try:
                # Read chunk directly from audio stream for better quality
                data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                audio_chunk = np.frombuffer(data, dtype=np.int16)
                
                # Store original high-quality audio
                highquality_audio = np.append(highquality_audio, audio_chunk)
                
                # Calculate audio level for display
                raw_level = np.max(np.abs(audio_chunk)) / 32767.0  # Normalized 0-1
                
                # Add to ring buffer for smoother level display
                ring_buffer.append(raw_level)
                smoothed_level = sum(ring_buffer) / len(ring_buffer)
                
                # Update peak and average levels for quality metrics
                peak_level = max(peak_level, raw_level)
                avg_level = (avg_level * sample_count + raw_level) / (sample_count + 1)
                sample_count += 1
                
                # Process audio through enhancement pipeline
                # Convert to float32 for processing
                float_audio = audio_chunk.astype(np.float32) / 32767.0
                is_speaking = False
                
                # Process with FullSubNet if available
                if fullsubnet_enhancer is not None:
                    try:
                        # Process with FullSubNet enhancer
                        enhanced_audio, fs_speech = fullsubnet_enhancer.process_audio(float_audio, SAMPLE_RATE)
                        is_speaking = fs_speech
                        
                        # Note: FullSubNet enhancer already returns float32 in range [-1, 1]
                        processed_audio = enhanced_audio
                    except Exception as e:
                        logger.error(f"Error in FullSubNet processing: {e}")
                        processed_audio = float_audio  # Fall back to original
                        # Continue with battlefield enhancer as fallback
                elif battlefield_enhancer is not None:
                    try:
                        # Process with battlefield enhancer
                        enhanced_audio, bf_speech = battlefield_enhancer.process_audio(float_audio, SAMPLE_RATE)
                        is_speaking = bf_speech
                        
                        # Note: Battlefield enhancer already returns float32 in range [-1, 1]
                        processed_audio = enhanced_audio
                    except Exception as e:
                        logger.error(f"Error in Battlefield processing: {e}")
                        processed_audio = float_audio  # Fall back to original
                else:
                    # No enhancers available, use basic noise reduction
                    if noise_profile is not None:
                        # Convert audio chunk to appropriate shape for FFT if needed
                        if len(audio_chunk) != CHUNK_SIZE:
                            # Pad or truncate to match chunk size for FFT
                            if len(audio_chunk) < CHUNK_SIZE:
                                padded = np.zeros(CHUNK_SIZE, dtype=np.int16)
                                padded[:len(audio_chunk)] = audio_chunk
                                audio_chunk = padded
                            else:
                                audio_chunk = audio_chunk[:CHUNK_SIZE]
                        
                        processed_audio = apply_noise_reduction(audio_chunk, noise_profile, 0.7) / 32767.0
                    else:
                        processed_audio = float_audio
                    
                    # Use simple energy-based VAD for speech detection
                    energy = np.sqrt(np.mean(processed_audio ** 2))
                    is_speaking = energy > threshold_level
                
                # Apply level normalization to all audio
                processed_audio = normalize_audio(processed_audio, target_level=-18)
                
                # Convert back to int16
                processed_chunk = (processed_audio * 32767).astype(np.int16)
                
                # Resample to STT engine's required rate (16kHz)
                if SAMPLE_RATE != 16000:
                    resampled_chunk = resample_audio(processed_chunk, SAMPLE_RATE, 16000)
                else:
                    resampled_chunk = processed_chunk
                
                # Convert to int16 if needed
                if resampled_chunk.dtype != np.int16:
                    resampled_chunk = (resampled_chunk * 32767).astype(np.int16)
                
                # Add processed audio to our processing buffer
                all_audio_data = np.append(all_audio_data, resampled_chunk)
                
                # Add to STT buffer 
                stt_buffer = np.append(stt_buffer, resampled_chunk)
                
                # Update remaining time
                elapsed = time.time() - start_time
                remaining_time = max(0, RECORDING_DURATION - elapsed)
                
                # Check for time warnings to show countdown
                if remaining_time <= 5 and not five_sec_warning_shown and in_terminal:
                    print("\033[2J\033[H")  # Clear screen
                    print_big_message("FINISHING IN 5 SECONDS", "alert")
                    five_sec_warning_shown = True
                    time.sleep(0.2)  # Brief pause for visibility
                    
                elif remaining_time <= 3 and not three_sec_warning_shown and in_terminal:
                    print("\033[2J\033[H")  # Clear screen
                    print_big_message("FINISHING IN 3 SECONDS", "alert")
                    three_sec_warning_shown = True
                    time.sleep(0.2)  # Brief pause for visibility
                    
                elif remaining_time <= 1 and not one_sec_warning_shown and in_terminal:
                    print("\033[2J\033[H")  # Clear screen
                    print_big_message("STOP SPEAKING NOW", "stop")
                    one_sec_warning_shown = True
                    time.sleep(0.2)  # Brief pause for visibility
                
                # Display audio level with improved meter
                is_speaking = smoothed_level > threshold_level
                
                # Track speaking state changes for UI feedback
                if is_speaking and not was_speaking:
                    if in_terminal:
                        print("\033[2J\033[H")  # Clear screen
                        print_big_message("SPEECH DETECTED", "success")
                        time.sleep(0.1)  # Brief pause for visibility
                    speaking_duration = 0
                    
                elif not is_speaking and was_speaking:
                    # Only show silence message if we've been speaking for a while
                    if speaking_duration > 1.0 and in_terminal:
                        print("\033[2J\033[H")  # Clear screen
                        print_big_message("SILENCE DETECTED", "warning")
                        time.sleep(0.1)  # Brief pause for visibility
                    silence_duration = 0
                
                # Update speaking duration tracking
                if is_speaking:
                    speaking_duration += CHUNK_SIZE / SAMPLE_RATE
                    silence_duration = 0
                else:
                    silence_duration += CHUNK_SIZE / SAMPLE_RATE
                    
                # Remember current speaking state
                was_speaking = is_speaking
                
                # Create an extra large level meter with enhanced visuals
                level_meter = display_audio_level(smoothed_level, width=80, use_color=in_terminal)
                
                # Build status line based on terminal mode
                if in_terminal:
                    # Move cursor to top of screen for clean display
                    print("\033[H")  # Move to home position
                    
                    # Display header with recording info
                    print("\033[1;97;44m" + "TCCC SPEECH CAPTURE".center(100) + "\033[0m")
                    
                    # Display remaining time prominently with progress bar
                    time_display = f"\033[1mRemaining: \033[93m{int(remaining_time):02d}\033[0m of {RECORDING_DURATION} seconds"
                    print(f"\n{time_display.center(100)}")
                    
                    # Add progress bar for recording time
                    progress_pct = (RECORDING_DURATION - remaining_time) / RECORDING_DURATION
                    print_progress_bar(progress_pct, 1.0, width=80, 
                                      prefix="Recording Progress:", suffix="",
                                      use_color=True)
                    
                    # Space for visual separation
                    print("\n")
                    
                    # Display large, colored level meter with status
                    if is_speaking:
                        # Highlight speaking with green background
                        status = "\033[1;97;42m SPEAKING DETECTED \033[0m"
                        border = "\033[92m" + "★" * 90 + "\033[0m"
                    else:
                        # Show silence with dimmer display
                        status = "\033[1;90m Silence \033[0m"
                        border = "\033[90m" + "·" * 90 + "\033[0m"
                    
                    # Show speaking/silence borders
                    print(f"{border}")
                    print(f"\r{level_meter}")
                    print(f"{status.center(100)}")
                    print(f"{border}")
                    
                    # Add real-time guidance for the user
                    if is_speaking:
                        print(f"\n\033[1;92mContinue speaking clearly - Audio quality: {peak_level*100:.1f}% peak\033[0m".center(100))
                    else:
                        print(f"\n\033[1;93mPlease speak clearly into the microphone\033[0m".center(100))
                        
                    # Remind of countdown if near the end
                    if remaining_time < 10:
                        print(f"\n\033[1;91mALMOST FINISHED - {int(remaining_time)} SECONDS LEFT\033[0m".center(100))
                    
                else:
                    # Simpler display for SSH session
                    voice_indicator = "VOICE DETECTED" if is_speaking else "silence"
                    sys.stdout.write(f"\rLevel: {level_meter} {voice_indicator} | Time left: {int(remaining_time):02d}s")
                
                sys.stdout.flush()
                
                # Process through STT when we have enough data and significant audio
                # Using adaptive threshold that's slightly above the noise floor
                if len(stt_buffer) >= stt_buffer_size and is_speaking:
                    # Process through STT
                    result = stt_engine.transcribe_segment(stt_buffer)
                    
                    # Reset buffer for next chunk but keep some overlap for continuity
                    overlap = min(4000, len(stt_buffer) // 4)  # 250ms overlap
                    stt_buffer = stt_buffer[-overlap:] if overlap > 0 else np.array([], dtype=np.int16)
                    
                    # Process result
                    if result and 'text' in result and result['text'].strip():
                        text = result['text'].strip()
                        
                        # Only show/save if different from the last result and not empty
                        if text != last_text and text:
                            last_text = text
                            
                            # Get current timestamp
                            elapsed = time.time() - start_time
                            timestamp = f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
                            
                            # Display differently based on terminal mode
                            if in_terminal:
                                # Clear the line and display with color
                                print(f"\r\033[K\033[1;96m[{timestamp}] \033[97m{text}\033[0m")
                                # Add extra lines to maintain display area
                                for _ in range(2):
                                    print("")
                            else:
                                # Simple display for SSH
                                print(f"\r[{timestamp}] {text}")
                            
                            # Save to file
                            with open(OUTPUT_FILE, 'a') as f:
                                f.write(f"[{timestamp}] {text}\n")
                
            except Exception as e:
                error_msg = f"Error during recording: {e}"
                logger.error(error_msg)
                print(f"\n{error_msg}")
                time.sleep(0.1)  # Prevent busy-waiting in case of repeated errors
        
            # Small delay to prevent CPU overuse
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("\n\nTranscription stopped by user.")
    
    # Final processing of any remaining audio
    if in_terminal:
        # Add some spacing
        for _ in range(5):
            print("")
    
    # Process any remaining audio in the STT buffer
    if len(stt_buffer) > 0:
        result = stt_engine.transcribe_segment(stt_buffer)
        if result and 'text' in result and result['text'].strip():
            text = result['text'].strip()
            elapsed = time.time() - start_time
            timestamp = f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
            
            if in_terminal:
                print(f"\r\033[K\033[1;96m[{timestamp}] \033[97m{text} (final)\033[0m")
            else:
                print(f"\r[{timestamp}] {text}")
                
            with open(OUTPUT_FILE, 'a') as f:
                f.write(f"[{timestamp}] {text}\n")
                
    # Clean up the audio stream
    stream.stop_stream()
    stream.close()
    audio.terminate()
    
    # Show recording complete message
    if in_terminal:
        print_big_message("RECORDING COMPLETE", "success")
    else:
        print("\n\n==== RECORDING COMPLETE ====")
    
    # Print audio quality metrics with color for terminal
    if in_terminal:
        print(f"\n\033[1;97mAudio Quality Metrics:\033[0m")
        peak_db = 20*np.log10(peak_level+1e-10)
        avg_db = 20*np.log10(avg_level+1e-10)
        
        # Color-code the peak level
        peak_color = "\033[92m"  # Green
        if peak_db > -10:
            peak_color = "\033[91m"  # Red if potentially clipping
        elif peak_db > -18:
            peak_color = "\033[93m"  # Yellow if high but not clipping
            
        print(f"Peak level: {peak_color}{peak_level*100:.1f}% ({peak_db:.1f} dB)\033[0m")
        print(f"Average level: \033[96m{avg_level*100:.1f}% ({avg_db:.1f} dB)\033[0m")
        
        # Show enhancer performance stats
        if battlefield_enhancer is not None:
            print(f"\n\033[1;97mBattlefield Audio Enhancement:\033[0m")
            stats = battlefield_enhancer.get_performance_stats()
            
            # Display processing performance
            proc_time = stats.get('average_processing_time_ms', 0)
            print(f"Processing time: \033[96m{proc_time:.2f} ms/chunk\033[0m")
            
            # Show speech detection stats
            speech_ratio = stats.get('speech_to_noise_ratio', 0) * 100
            print(f"Speech detection ratio: \033[93m{speech_ratio:.1f}%\033[0m")
            
            # Show adaptive settings
            adaptive = stats.get('adaptive_settings', {})
            distance = adaptive.get('distance_factor', 1.0)
            env_type = adaptive.get('environment_type', 'unknown')
            print(f"Environment type: \033[93m{env_type.upper()}\033[0m, Distance factor: \033[93m{distance:.1f}x\033[0m")
            
            # Show estimated SNR
            est_snr = stats.get('estimated_snr_db', 0)
            print(f"Estimated SNR: \033[92m{est_snr:.1f} dB\033[0m")
            
        if fullsubnet_enhancer is not None:
            print(f"\n\033[1;97mFullSubNet Speech Enhancement:\033[0m")
            stats = fullsubnet_enhancer.get_performance_stats()
            
            # Display processing performance
            proc_time = stats.get('average_processing_time_ms', 0)
            print(f"Processing time: \033[96m{proc_time:.2f} ms/chunk\033[0m")
            
            # Show GPU utilization if available
            if stats.get('using_gpu', False):
                if 'gpu_memory_allocated_mb' in stats:
                    print(f"GPU memory used: \033[93m{stats['gpu_memory_allocated_mb']:.1f} MB\033[0m")
                print(f"Acceleration: \033[92m{stats.get('using_mixed_precision', False) and 'Mixed Precision' or 'FP32'}\033[0m")
            
            # Show estimated SNR improvement
            snr_improv = stats.get('estimated_snr_improvement_db', 0)
            print(f"Estimated SNR improvement: \033[92m{snr_improv:.1f} dB\033[0m")
            
            # Show processing rate
            rate = stats.get('processing_rate', 0)
            print(f"Processing rate: \033[93m{rate:.1f} chunks/second\033[0m")
            
    else:
        print(f"\nAudio Quality Metrics:")
        print(f"Peak level: {peak_level*100:.1f}% ({20*np.log10(peak_level+1e-10):.1f} dB)")
        print(f"Average level: {avg_level*100:.1f}% ({20*np.log10(avg_level+1e-10):.1f} dB)")
        
        # Show enhancer stats in simple format for non-terminal
        if battlefield_enhancer is not None:
            print("\nBattlefield Audio Enhancement:")
            stats = battlefield_enhancer.get_performance_stats()
            print(f"Processing time: {stats.get('average_processing_time_ms', 0):.2f} ms/chunk")
            print(f"Environment type: {stats.get('adaptive_settings', {}).get('environment_type', 'unknown').upper()}")
            print(f"Estimated SNR: {stats.get('estimated_snr_db', 0):.1f} dB")
            
        if fullsubnet_enhancer is not None:
            print("\nFullSubNet Speech Enhancement:")
            stats = fullsubnet_enhancer.get_performance_stats()
            print(f"Processing time: {stats.get('average_processing_time_ms', 0):.2f} ms/chunk")
            print(f"Using GPU: {stats.get('using_gpu', False)}")
            print(f"Estimated SNR improvement: {stats.get('estimated_snr_improvement_db', 0):.1f} dB")
                
    # Save the high-quality original audio recording
    if len(highquality_audio) > 0:
        try:
            save_msg = f"Saving high quality audio to {HIGHQUALITY_AUDIO}..."
            logger.info(save_msg)
            if in_terminal:
                print(f"\n\033[1;93m{save_msg}\033[0m")
            else:
                print(f"\n{save_msg}")
                
            # Save the raw high-quality audio data to a WAV file at original sample rate
            with wave.open(HIGHQUALITY_AUDIO, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(highquality_audio.tobytes())
                
            duration_sec = len(highquality_audio)/SAMPLE_RATE
            result_msg = f"High-quality audio saved: {HIGHQUALITY_AUDIO} ({duration_sec:.1f} seconds, {SAMPLE_RATE/1000:.1f}kHz)"
            logger.info(result_msg)
            
            if in_terminal:
                print(f"\033[92m{result_msg}\033[0m")
            else:
                print(result_msg)
                
        except Exception as e:
            error_msg = f"Error saving high-quality audio file: {e}"
            logger.error(error_msg)
            print(error_msg)
    
    # Save the processed audio used for STT
    if len(all_audio_data) > 0:
        try:
            save_msg = f"Saving processed audio to {AUDIO_FILE}..."
            logger.info(save_msg)
            if in_terminal:
                print(f"\n\033[1;93m{save_msg}\033[0m")
            else:
                print(f"{save_msg}")
                
            # Save the processed audio data to a WAV file
            with wave.open(AUDIO_FILE, 'wb') as wf:
                wf.setnchannels(1)  # Mono
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(16000)  # 16kHz
                wf.writeframes(all_audio_data.tobytes())
                
            duration_sec = len(all_audio_data)/16000
            result_msg = f"Processed audio saved: {AUDIO_FILE} ({duration_sec:.1f} seconds, noise-reduced & normalized)"
            logger.info(result_msg)
            
            if in_terminal:
                print(f"\033[92m{result_msg}\033[0m")
            else:
                print(result_msg)
                
        except Exception as e:
            error_msg = f"Error saving processed audio file: {e}"
            logger.error(error_msg)
            print(error_msg)
    
    # Clean up
    shutdown_msg = "Finishing transcription..."
    logger.info(shutdown_msg)
    print(f"\n{shutdown_msg}")
    stt_engine.shutdown()
    
    # Show results
    try:
        transcription_content = ""
        with open(OUTPUT_FILE, 'r') as f:
            transcription_content = f.read()
            
        # Get word count for the transcription
        word_count = sum(len(line.split()) for line in transcription_content.splitlines() if ']' in line)
        
        if in_terminal:
            print(f"\n\033[1;97mTranscription saved to: \033[93m{OUTPUT_FILE}\033[0m")
            print(f"\033[1;92mTranscription complete with enhanced audio quality!\033[0m")
            print(f"\033[1;97mWords detected: \033[96m{word_count}\033[0m")
        else:
            print(f"\nTranscription saved to: {OUTPUT_FILE}")
            print(f"Transcription complete with enhanced audio quality!")
            print(f"Words detected: {word_count}")
            
        # Print transcription content in terminal mode
        if in_terminal and transcription_content:
            print("\n\033[1;97m========== TRANSCRIBED TEXT ==========\033[0m")
            print(transcription_content)
            print("\033[1;97m======================================\033[0m")
    except Exception as e:
        error_msg = f"Error displaying transcription: {e}"
        logger.error(error_msg)
    
    # Print file size comparison to show quality difference
    try:
        highquality_size = os.path.getsize(HIGHQUALITY_AUDIO)
        processed_size = os.path.getsize(AUDIO_FILE)
        
        if in_terminal:
            print(f"\n\033[1;97mFile size comparison:\033[0m")
            print(f"\033[1;97mHigh-quality audio: \033[93m{highquality_size/1024:.1f} KB\033[0m")
            print(f"\033[1;97mProcessed audio: \033[93m{processed_size/1024:.1f} KB\033[0m")
            print(f"\033[1;97mQuality ratio: \033[93m{highquality_size/processed_size:.1f}x\033[0m larger high-quality file")
        else:
            print(f"\nFile size comparison:")
            print(f"High-quality audio: {highquality_size/1024:.1f} KB")
            print(f"Processed audio: {processed_size/1024:.1f} KB")
            print(f"Quality ratio: {highquality_size/processed_size:.1f}x larger high-quality file")
    except Exception as e:
        error_msg = f"Error comparing file sizes: {e}"
        logger.error(error_msg)
    
    # Keep terminal window open if in terminal mode
    if in_terminal:
        print_big_message("TERMINAL WILL STAY OPEN", "info")
        print("\033[1;97mPress Ctrl+C to close this window when finished viewing results\033[0m")
        
        # Keep the terminal window open
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\033[1;93mTerminal closed by user\033[0m")
    else:
        print("\nTranscription process complete. Check the output files for results.")

if __name__ == "__main__":
    main()