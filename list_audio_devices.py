import pyaudio

p = pyaudio.PyAudio()
print('Available PyAudio Input Devices:')
try:
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info.get('maxInputChannels') > 0:
            print(f"  Index {i}: {info.get('name')} (Input Channels: {info.get('maxInputChannels')})")
except Exception as e:
    print(f"Error querying devices: {e}")
finally:
    p.terminate()
