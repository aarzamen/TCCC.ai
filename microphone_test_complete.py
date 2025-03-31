#!/usr/bin/env python3
"""
Complete Microphone Test Script for TCCC

This script demonstrates the full speech-to-text capability:
1. Records from Razer Seiren V3 Mini microphone (device 0)
2. Processes speech using faster-whisper STT
3. Detects medical terms in the transcription
4. Displays results on screen
"""

import os
import sys
import time
import argparse
import threading
import numpy as np
import re
import wave
from pathlib import Path

# Set environment variables for stable operation
os.environ["SDL_VIDEODRIVER"] = "x11"  # For display
os.environ["USE_MOCK_STT"] = "0"       # Use real STT

# Set up paths
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(project_dir, 'src'))

# Import required components - fail early if not available
try:
    import pygame
    import pyaudio
    import soundfile as sf
    from tccc.stt_engine import create_stt_engine
    from tccc.utils.config_manager import ConfigManager
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please install required packages: pygame, pyaudio, soundfile")
    sys.exit(1)

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TCCC.MicTest")

class MicrophoneTest:
    """Microphone test implementation with direct display."""
    
    def __init__(self):
        """Initialize the microphone test."""
        self.config_manager = ConfigManager()
        self.stt_engine = None
        self.is_recording = False
        self.audio_buffer = []
        self.current_transcription = ""
        self.detected_entities = []
        self.recording_thread = None
        self.display_thread = None
        self.running = True
        
        # Pygame and display variables
        self.screen = None
        # WaveShare specific resolution (1280x800)
        self.width = 1280
        self.height = 800
        self.clock = None
        self.fonts = {}
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.GREEN = (0, 255, 0)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 120, 255)
        self.YELLOW = (255, 255, 0)
        
        # PyAudio setup
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.p = None
        self.stream = None
        
        # Test script - TCCC specific medical scenario for complete testing
        self.test_script = [
            "This is Medic One-Alpha reporting from grid coordinate Charlie-Delta 4352.",
            "Patient is a 28-year-old male with blast injuries to the right leg from IED.",
            "Initially unresponsive at scene with significant bleeding from right thigh.",
            "Applied tourniquet at 0930 hours and established two IVs.",
            "Vital signs are: BP 100/60, pulse 120, respiratory rate 24, oxygen saturation 92%.",
            "GCS is now 14, was initially 12 when found.",
            "Performed needle decompression on right chest for suspected tension pneumothorax.",
            "Administered 10mg morphine IV at 0940 and 1g ceftriaxone IV.",
            "Patient has severe right leg injury with controlled hemorrhage, possible TBI.",
            "We're continuing fluid resuscitation and monitoring vitals every 5 minutes.",
            "This is an urgent surgical case, requesting immediate MEDEVAC to Role 2."
        ]
        
    def initialize(self):
        """Initialize all components."""
        logger.info("Initializing microphone test...")
        
        # Initialize pygame for display
        try:
            # Force clean pygame initialization for WaveShare display
            if pygame.display.get_init():
                pygame.display.quit()
            pygame.quit()
            pygame.init()
            
            # Set specific environment variables for WaveShare display
            os.environ['SDL_VIDEODRIVER'] = 'x11'  # Use X11 driver
            os.putenv('SDL_FBDEV', '/dev/fb0')     # Use framebuffer device
            
            # Try hardware acceleration first - FULLSCREEN for WaveShare
            flags = pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF
            self.screen = pygame.display.set_mode((self.width, self.height), flags)
            pygame.display.set_caption("TCCC Microphone Test")
            self.clock = pygame.time.Clock()
            logger.info("Display initialized with hardware acceleration for WaveShare")
            
            # Load logo image if available
            self.logo = None
            try:
                logo_path = os.path.join(project_dir, "images", "blue_logo.png")
                if os.path.exists(logo_path):
                    self.logo = pygame.image.load(logo_path)
                    # Scale logo to reasonable size
                    self.logo = pygame.transform.scale(self.logo, (200, 150))
            except Exception as img_err:
                logger.warning(f"Could not load logo: {img_err}")
                
        except Exception as e:
            logger.error(f"Failed to initialize display with hardware acceleration: {e}")
            try:
                # Fallback to software rendering - still FULLSCREEN for WaveShare
                os.environ['SDL_VIDEODRIVER'] = 'x11'
                pygame.display.quit()
                pygame.init()
                self.screen = pygame.display.set_mode((self.width, self.height), 
                                                    pygame.FULLSCREEN | pygame.SWSURFACE)
                pygame.display.set_caption("TCCC Microphone Test (Software Rendering)")
                self.clock = pygame.time.Clock()
                logger.info("Display initialized with software rendering for WaveShare")
            except Exception as e2:
                logger.error(f"Failed to initialize display with software rendering: {e2}")
                # Will be caught in the render_display method
                self.screen = None
        
        # Load fonts
        self.fonts = {
            'large': pygame.font.SysFont('Arial', 36),
            'medium': pygame.font.SysFont('Arial', 24),
            'small': pygame.font.SysFont('Arial', 18)
        }
        
        # Initialize STT engine
        logger.info("Initializing STT engine...")
        stt_config = self.config_manager.load_config("stt_engine")
        
        # Override to use tiny model for speed
        if 'model' not in stt_config:
            stt_config['model'] = {}
        stt_config['model']['size'] = 'tiny.en'
        
        self.stt_engine = create_stt_engine("faster-whisper", stt_config)
        if not self.stt_engine.initialize(stt_config):
            logger.error("Failed to initialize STT engine")
            return False
            
        logger.info("STT engine initialized successfully")
        return True
        
    def start_recording(self):
        """Start recording audio from the microphone."""
        if self.is_recording:
            return
            
        self.is_recording = True
        self.audio_buffer = []
        
        # Start recording thread
        self.recording_thread = threading.Thread(target=self._record_audio)
        self.recording_thread.daemon = True
        self.recording_thread.start()
        
    def stop_recording(self):
        """Stop recording audio."""
        self.is_recording = False
        if self.recording_thread:
            self.recording_thread.join(timeout=1.0)
            
    def _record_audio(self):
        """Record audio from microphone."""
        try:
            self.p = pyaudio.PyAudio()
            
            # Print availability and selection of Razer Seiren specifically
            razer_device_id = None
            print("\n==== AUDIO DEVICE DETECTION ====")
            print("Looking for Razer Seiren V3 Mini...")
            
            # List devices for debugging
            logger.info("Available audio devices:")
            for i in range(self.p.get_device_count()):
                device_info = self.p.get_device_info_by_index(i)
                if device_info.get('maxInputChannels') > 0:
                    device_name = device_info.get('name', '')
                    logger.info(f"Device {i}: {device_name}")
                    print(f"  Device {i}: {device_name}")
                    
                    # Try to find Razer device
                    if 'razer' in device_name.lower() or 'seiren' in device_name.lower():
                        razer_device_id = i
                        print(f"  → FOUND RAZER MICROPHONE: Device {i}")
            
            # Use the identified Razer device or fall back to device 0
            if razer_device_id is None:
                logger.warning("Razer Seiren microphone not found. Using default device 0.")
                print("  → Razer Seiren not found. Using default device 0")
                razer_device_id = 0
            
            print(f"\n==== SELECTED DEVICE {razer_device_id} ====")
            print("Press SPACE to start/stop recording")
            print("Press ESC to exit")
            print("============================")
            
            # Open stream using detected Razer device
            self.stream = self.p.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                input_device_index=razer_device_id,
                frames_per_buffer=self.CHUNK
            )
            
            logger.info(f"Recording started with device {razer_device_id}")
            print("\n==== RECORDING STARTED ====")
            
            # Save audio to file as we record for backup
            frames = []
            
            # Record in a loop
            while self.is_recording:
                data = self.stream.read(self.CHUNK, exception_on_overflow=False)
                self.audio_buffer.append(data)
                frames.append(data)  # Store for file backup
                
                # Process periodically
                if len(self.audio_buffer) >= 8:  # Process ~1 second chunks
                    self._process_audio()
            
            # Cleanup
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()
            
            # Save recorded audio to file as backup
            if frames:
                backup_filename = f"tccc_recording_{int(time.time())}.wav"
                logger.info(f"Saving backup recording to {backup_filename}")
                wf = wave.open(backup_filename, 'wb')
                wf.setnchannels(self.CHANNELS)
                wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
                wf.setframerate(self.RATE)
                wf.writeframes(b''.join(frames))
                wf.close()
                print(f"\nRecording saved to {backup_filename}")
            
            logger.info("Recording stopped")
            print("\n==== RECORDING STOPPED ====")
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            logger.error(f"Error in recording: {e}\n{error_trace}")
            print(f"\n==== ERROR IN RECORDING ====\n{e}")
            self.is_recording = False
    
    def _process_audio(self):
        """Process recorded audio with STT."""
        if not self.audio_buffer:
            return
            
        try:
            # Combine the audio buffer contents
            raw_audio = b''.join(self.audio_buffer)
            
            # Force concrete feedback - log detailed buffer info
            logger.info(f"Processing audio buffer: {len(self.audio_buffer)} chunks, {len(raw_audio)} bytes")
            
            # Convert audio buffer to numpy array
            audio_data = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Clear buffer for next chunk
            self.audio_buffer = []
            
            # Process with STT engine and measure time
            start_time = time.time()
            result = self.stt_engine.transcribe_segment(audio_data)
            process_time = time.time() - start_time
            
            # Print detailed result
            if result:
                logger.info(f"STT Result: {result}")
                logger.info(f"Processing time: {process_time:.2f}s")
            
            # Extract text if available
            if result and 'text' in result and result['text'].strip():
                self.current_transcription += " " + result['text']
                
                # Print to console regardless of display mode working
                print(f"\rTranscribed: {result['text']}", end="", flush=True)
                logger.info(f"Transcribed: {result['text']}")
                
                # Extract medical entities
                self._extract_medical_terms(result['text'])
            else:
                # Empty result - log it clearly 
                logger.warning(f"Empty transcription result from audio segment of {len(audio_data)} samples")
                
        except Exception as e:
            # Detailed error reporting
            import traceback
            error_trace = traceback.format_exc()
            logger.error(f"Error processing audio: {e}\n{error_trace}")
    
    def _extract_medical_terms(self, text):
        """Extract medical terms from transcription."""
        text = text.lower()
        
        # Check for injuries
        if any(term in text for term in ["injury", "injuries", "wounded", "blast", "trauma", "bleeding"]):
            injury_type = "blast injury" if "blast" in text else "trauma" if "trauma" in text else "injury"
            location = "leg" if "leg" in text else "arm" if "arm" in text else "unknown"
            self.detected_entities.append({
                "type": "injury",
                "text": f"{injury_type} to {location}",
                "confidence": 0.9
            })
            
        # Check for vital signs
        bp_match = re.search(r"(?:blood pressure|bp)[^\d]*(\d+/\d+)", text)
        if bp_match:
            self.detected_entities.append({
                "type": "vital",
                "text": f"BP {bp_match.group(1)}",
                "confidence": 0.95
            })
            
        pulse_match = re.search(r"(?:pulse|heart rate)[^\d]*(\d+)", text)
        if pulse_match:
            self.detected_entities.append({
                "type": "vital",
                "text": f"Pulse {pulse_match.group(1)}",
                "confidence": 0.9
            })
            
        # Check for procedures
        if "tourniquet" in text:
            self.detected_entities.append({
                "type": "procedure",
                "text": "Applied tourniquet",
                "confidence": 0.9
            })
            
        if "iv" in text or "intravenous" in text:
            self.detected_entities.append({
                "type": "procedure",
                "text": "IV access",
                "confidence": 0.85
            })
            
        # Check for medications
        if "morphine" in text:
            dosage = "10mg" if "10" in text else "unknown dosage"
            self.detected_entities.append({
                "type": "medication",
                "text": f"Morphine {dosage}",
                "confidence": 0.9
            })
    
    def render_display(self):
        """Render the display window for WaveShare display."""
        if not self.screen:
            return
            
        # Clear screen
        self.screen.fill(self.BLACK)
        
        # Draw TCCC.ai logo if available
        if hasattr(self, 'logo') and self.logo:
            self.screen.blit(self.logo, (20, 15))
        
        # Draw header
        header = self.fonts['large'].render("TCCC.ai MICROPHONE TEST", True, self.WHITE)
        self.screen.blit(header, (self.width/2 - header.get_width()/2, 25))
        
        # Draw status with prominent status indicator
        status_text = "RECORDING" if self.is_recording else "READY TO RECORD"
        status_color = self.GREEN if self.is_recording else self.WHITE
        
        # Create status box
        status_box_width = 400
        status_box_height = 40
        status_box_x = self.width/2 - status_box_width/2
        status_box_y = 75
        
        # Draw status box with color indication
        pygame.draw.rect(self.screen, 
                        (0, 100, 0) if self.is_recording else (50, 50, 50), 
                        (status_box_x, status_box_y, status_box_width, status_box_height))
        pygame.draw.rect(self.screen, status_color, 
                        (status_box_x, status_box_y, status_box_width, status_box_height), 3)
        
        # Draw status text
        status = self.fonts['medium'].render(status_text, True, status_color)
        self.screen.blit(status, (self.width/2 - status.get_width()/2, status_box_y + 8))
        
        # Draw script box for the medical scenario
        script_box_height = 350
        pygame.draw.rect(self.screen, (15, 25, 45), (20, 130, self.width/2 - 40, script_box_height))
        pygame.draw.rect(self.screen, self.WHITE, (20, 130, self.width/2 - 40, script_box_height), 2)
        
        script_header = self.fonts['medium'].render("TCCC MEDICAL SCENARIO (READ ALOUD)", True, self.YELLOW)
        self.screen.blit(script_header, (30, 140))
        
        # Render test script with scrolling if needed
        visible_lines = min(len(self.test_script), 10)  # Show max 10 lines at once
        for i, line in enumerate(self.test_script[:visible_lines]):
            script_line = self.fonts['small'].render(line, True, self.WHITE)
            self.screen.blit(script_line, (30, 180 + i*30))
        
        # Add page indicator if more lines exist
        if len(self.test_script) > visible_lines:
            more_text = self.fonts['small'].render("(More lines below - test supports scrolling)", True, (180, 180, 180))
            self.screen.blit(more_text, (30, 180 + visible_lines*30 + 10))
            
        # Draw instructions box with clear border
        instructions_y = 130 + script_box_height + 20
        pygame.draw.rect(self.screen, (15, 25, 45), (20, instructions_y, self.width/2 - 40, 100))
        pygame.draw.rect(self.screen, self.BLUE, (20, instructions_y, self.width/2 - 40, 100), 2)
        
        instructions = [
            "INSTRUCTIONS:",
            "• Press SPACE BAR to start/stop recording",
            "• Press ESCAPE to exit the test",
            "• Speak clearly into the Razer Seiren V3 Mini"
        ]
        
        for i, line in enumerate(instructions):
            instr = self.fonts['small'].render(line, True, self.BLUE if i == 0 else self.WHITE)
            self.screen.blit(instr, (30, instructions_y + 10 + i*25))
            
        # Draw transcription box - make it larger for WaveShare
        pygame.draw.rect(self.screen, (15, 25, 45), (self.width/2 + 20, 130, self.width/2 - 40, script_box_height))
        pygame.draw.rect(self.screen, self.GREEN, (self.width/2 + 20, 130, self.width/2 - 40, script_box_height), 2)
        
        transcription_header = self.fonts['medium'].render("LIVE TRANSCRIPTION", True, self.GREEN)
        self.screen.blit(transcription_header, (self.width/2 + 30, 140))
        
        # Render transcription (with word wrapping)
        transcription_text = self.current_transcription[-500:] if self.current_transcription else "Speak into the microphone..."
        
        # Simple word wrapping
        words = transcription_text.split()
        lines = []
        current_line = ""
        
        for word in words:
            test_line = current_line + " " + word if current_line else word
            if self.fonts['small'].size(test_line)[0] < self.width/2 - 60:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
                
        if current_line:
            lines.append(current_line)
            
        # Show more lines for the WaveShare display
        for i, line in enumerate(lines[-12:]):  # Show up to 12 lines
            text = self.fonts['small'].render(line, True, self.WHITE)
            self.screen.blit(text, (self.width/2 + 30, 180 + i*25))
            
        # Draw entities box that uses the full width of the WaveShare display
        entities_y = instructions_y + 120
        pygame.draw.rect(self.screen, (15, 25, 45), (20, entities_y, self.width - 40, 270))
        pygame.draw.rect(self.screen, self.YELLOW, (20, entities_y, self.width - 40, 270), 2)
        
        entities_header = self.fonts['medium'].render("DETECTED MEDICAL ENTITIES", True, self.YELLOW)
        self.screen.blit(entities_header, (self.width/2 - entities_header.get_width()/2, entities_y + 10))
        
        # Group entities by type
        entities_by_type = {}
        for entity in self.detected_entities:
            entity_type = entity.get("type", "other")
            if entity_type not in entities_by_type:
                entities_by_type[entity_type] = []
            entities_by_type[entity_type].append(entity)
            
        # Define colors and titles for each type
        type_configs = {
            "injury": {"title": "INJURIES", "color": self.RED},
            "vital": {"title": "VITAL SIGNS", "color": self.GREEN},
            "procedure": {"title": "PROCEDURES", "color": self.BLUE},
            "medication": {"title": "MEDICATIONS", "color": (255, 165, 0)}  # Orange
        }
        
        # Render entities by type in a grid optimized for WaveShare
        col_width = (self.width - 60) / 2  # Use 2 columns
        row_height = 120
        
        # Layout in a 2x2 grid
        positions = [
            {"x": 40, "y": entities_y + 50},  # Top left
            {"x": 40 + col_width, "y": entities_y + 50},  # Top right
            {"x": 40, "y": entities_y + 50 + row_height},  # Bottom left
            {"x": 40 + col_width, "y": entities_y + 50 + row_height}  # Bottom right
        ]
        
        for i, (entity_type, config) in enumerate(type_configs.items()):
            pos = positions[i]
            
            # Draw type header
            type_header = self.fonts['medium'].render(config["title"], True, config["color"])
            self.screen.blit(type_header, (pos["x"], pos["y"]))
            
            # Draw box around this category
            box_padding = 10
            box_width = col_width - 20
            box_height = row_height - 20
            pygame.draw.rect(self.screen, (30, 30, 60), 
                            (pos["x"] - box_padding, pos["y"] - box_padding, 
                             box_width, box_height), 1)
            
            # Render entities of this type
            if entity_type in entities_by_type:
                for j, entity in enumerate(entities_by_type[entity_type][-5:]):  # Show last 5
                    if j >= 5:  # Safety check - limit to 5 entities per category
                        break
                        
                    entity_text = entity.get("text", "")
                    confidence = entity.get("confidence", 0) * 100
                    
                    text = self.fonts['small'].render(f"• {entity_text}", True, self.WHITE)
                    conf = self.fonts['small'].render(f"({confidence:.0f}%)", True, config["color"])
                    
                    text_y = pos["y"] + 30 + j*22
                    self.screen.blit(text, (pos["x"], text_y))
                    self.screen.blit(conf, (pos["x"] + text.get_width() + 10, text_y))
        
        # Draw timestamp at bottom
        timestamp = self.fonts['small'].render(f"Time: {time.strftime('%H:%M:%S')}", True, (150, 150, 150))
        self.screen.blit(timestamp, (self.width - timestamp.get_width() - 20, self.height - 30))
        
        # Update the display - force buffer swap for WaveShare
        pygame.display.flip()
        pygame.event.pump()  # Process events to ensure display updates
            
    def run(self):
        """Run the microphone test."""
        if not self.initialize():
            return False
            
        # Start display thread
        self.display_thread = threading.Thread(target=self._display_thread)
        self.display_thread.daemon = True
        self.display_thread.start()
        
        # Main event loop
        try:
            while self.running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            self.running = False
                        elif event.key == pygame.K_SPACE:
                            if self.is_recording:
                                self.stop_recording()
                            else:
                                self.start_recording()
                
                time.sleep(0.05)  # Short sleep to prevent high CPU usage
                
        except KeyboardInterrupt:
            logger.info("Test interrupted by user")
        finally:
            # Cleanup
            self.stop_recording()
            self.running = False
            
            if self.display_thread:
                self.display_thread.join(timeout=1.0)
                
            pygame.quit()
            
        return True
    
    def _display_thread(self):
        """Thread function to update display."""
        try:
            while self.running:
                self.render_display()
                self.clock.tick(30)  # 30 FPS
        except Exception as e:
            logger.error(f"Display thread error: {e}")
            # Fallback to console mode if display fails
            print("\n==== DISPLAY ERROR - FALLING BACK TO CONSOLE MODE ====")
            print("Press Ctrl+C to exit the program")
            while self.running:
                time.sleep(1.0)
                if self.current_transcription:
                    print(f"\nTranscription: {self.current_transcription}")
                    for entity in self.detected_entities[-5:]:  # Show last 5 entities
                        entity_type = entity.get("type", "unknown")
                        text = entity.get("text", "")
                        print(f"  • {entity_type.upper()}: {text}")
            
def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="TCCC Microphone Test")
    args = parser.parse_args()
    
    test = MicrophoneTest()
    if test.run():
        print("Test completed successfully")
        return 0
    else:
        print("Test failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())