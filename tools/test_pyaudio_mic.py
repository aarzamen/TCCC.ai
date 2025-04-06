import pyaudio
import time
import sys
import traceback

# --- Configuration ---
# !!! IMPORTANT: Replace None with the actual device index from check_pyaudio_devices.py !!!
# If you want to test the default device, leave it as None.
DEVICE_INDEX = None  # e.g., 0, 1, 2, ... or None

CHUNK = 1024             # Samples per frame
FORMAT = pyaudio.paInt16  # Audio format (bytes per sample)
CHANNELS = 1             # Single channel for microphone
RATE = 16000             # Sample rate (samples per second)
RECORD_SECONDS = 5       # How long to record for

# --- Script ---
pa = None
stream = None

print("-" * 30)
print("Minimal PyAudio Microphone Test")
print("-" * 30)
print(f"Configuration:")
print(f"  Device Index: {DEVICE_INDEX if DEVICE_INDEX is not None else 'Default (None)'}")
print(f"  Channels: {CHANNELS}")
print(f"  Rate: {RATE} Hz")
print(f"  Format: paInt16")
print(f"  Chunk Size: {CHUNK} frames")
print(f"  Record Duration: {RECORD_SECONDS} seconds")
print("-" * 30)

try:
    # 1. Initialize PyAudio
    print("Initializing PyAudio...")
    pa = pyaudio.PyAudio()
    print("PyAudio Initialized Successfully.")

    # 2. Open Audio Stream
    print(f"Attempting to open stream for device index: {DEVICE_INDEX}...")
    stream = pa.open(format=FORMAT,
                     channels=CHANNELS,
                     rate=RATE,
                     input=True,
                     frames_per_buffer=CHUNK,
                     input_device_index=DEVICE_INDEX) # Use the specified device index
    print("Audio Stream Opened Successfully.")
    print("-" * 30)

    # 3. Record Audio
    print(f"Recording for {RECORD_SECONDS} seconds...")
    frames_recorded = 0
    total_chunks = int(RATE / CHUNK * RECORD_SECONDS)
    data_received = False

    for i in range(0, total_chunks):
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames_recorded += len(data) // (CHANNELS * pa.get_sample_size(FORMAT)) # Calculate frames based on bytes read
            if len(data) > 0:
                data_received = True
                # Print progress indicator without newline
                print(f"\r  Reading chunk {i+1}/{total_chunks}... Read {len(data)} bytes.", end="")
            else:
                print(f"\r  Reading chunk {i+1}/{total_chunks}... Read 0 bytes (Potential Issue!).", end="")
                time.sleep(0.01) # Small sleep if no data

        except IOError as e:
            print(f"\nIOError during stream read: {e}")
            print("This might indicate an input overflow or device issue.")
            # Optionally break or continue depending on desired behavior
            break
        except Exception as e:
            print(f"\nUnexpected error during stream read: {e}")
            traceback.print_exc()
            break

    print("\nRecording Finished.")
    print("-" * 30)
    print("Summary:")
    print(f"  Total chunks expected: {total_chunks}")
    print(f"  Actual frames recorded (calculated): {frames_recorded}")
    print(f"  Data received at least once: {data_received}")
    if not data_received:
        print("  WARNING: No audio data was received from the stream!")


except pyaudio.PyAudioError as pa_err:
    print(f"\nPyAudio Error: {pa_err}", file=sys.stderr)
    if "Invalid device index" in str(pa_err):
        print(">>> The specified DEVICE_INDEX is likely incorrect. Please verify with check_pyaudio_devices.py.", file=sys.stderr)
    elif "Invalid sample rate" in str(pa_err):
         print(f">>> The sample rate ({RATE}Hz) might not be supported by this device.", file=sys.stderr)
    elif "Device unavailable" in str(pa_err):
         print(">>> The audio device might be in use by another application or disconnected.", file=sys.stderr)
    else:
        traceback.print_exc()

except Exception as e:
    print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)
    traceback.print_exc()

finally:
    # 4. Clean Up
    print("-" * 30)
    print("Cleaning up...")
    if stream is not None:
        try:
            if stream.is_active():
                stream.stop_stream()
                print("  Stream stopped.")
            stream.close()
            print("  Stream closed.")
        except Exception as e:
            print(f"  Error closing stream: {e}", file=sys.stderr)

    if pa is not None:
        pa.terminate()
        print("  PyAudio terminated.")
    print("Cleanup Complete.")
    print("-" * 30)
