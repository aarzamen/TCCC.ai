import pyaudio
import sys

def list_audio_devices():
    """Lists available audio devices using PyAudio."""
    pa = None
    try:
        pa = pyaudio.PyAudio()
        print("PyAudio Initialized Successfully.")
        print("-" * 30)
        print("Available Audio Devices:")
        print("-" * 30)

        info = pa.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')
        found_input_device = False

        if numdevices is None:
            print("Could not get device count from host API info.")
            print("Host API Info:", info)
            return

        print(f"Found {numdevices} devices using host API {info.get('name')}:")

        for i in range(0, numdevices):
            device_info = pa.get_device_info_by_index(i)
            print(f"\nDevice Index: {i}")
            print(f"  Name: {device_info.get('name')}")
            print(f"  Host API: {device_info.get('hostApi')} ({pa.get_host_api_info_by_index(device_info.get('hostApi')).get('name')})")
            print(f"  Max Input Channels: {device_info.get('maxInputChannels')}")
            print(f"  Max Output Channels: {device_info.get('maxOutputChannels')}")
            print(f"  Default Sample Rate: {device_info.get('defaultSampleRate')}")

            if device_info.get('maxInputChannels', 0) > 0:
                found_input_device = True
                print("  *** This is an Input Device ***")

        print("-" * 30)
        if not found_input_device:
            print("!!! No input devices found by PyAudio !!!")
        else:
            print("Input devices were found.")
        print("-" * 30)

    except Exception as e:
        print(f"Error initializing PyAudio or getting device info: {e}", file=sys.stderr)
        if "No Default Input Device Available" in str(e):
            print("\nThis specific error often means the underlying sound system (like ALSA on Linux)", file=sys.stderr)
            print("cannot find or access the audio device hardware.", file=sys.stderr)
            print("Check connections, system audio settings (e.g., using 'arecord -l' on Linux), and permissions.", file=sys.stderr)

    finally:
        if pa:
            pa.terminate()
            print("\nPyAudio Terminated.")

if __name__ == "__main__":
    list_audio_devices()
