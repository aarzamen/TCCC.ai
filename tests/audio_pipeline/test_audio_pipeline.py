"""
Tests for the Audio Pipeline module.
"""

import os
import time
import unittest
import tempfile
from unittest.mock import MagicMock, patch
import numpy as np

from tccc.audio_pipeline import AudioPipeline, AudioProcessor, AudioSource, StreamBuffer


class TestStreamBuffer(unittest.TestCase):
    """Test the StreamBuffer class."""
    
    def setUp(self):
        """Set up test environment."""
        self.buffer = StreamBuffer(buffer_size=5, timeout_ms=50)
        
    def test_write_read(self):
        """Test writing to and reading from buffer."""
        # Create test data
        test_data = np.ones(1024, dtype=np.int16)
        
        # Write to buffer
        bytes_written = self.buffer.write(test_data)
        self.assertEqual(bytes_written, len(test_data.tobytes()))
        
        # Read from buffer
        read_data = self.buffer.read()
        
        # Verify data
        self.assertTrue(np.array_equal(test_data, read_data))
    
    def test_buffer_full(self):
        """Test buffer when full."""
        # Fill the buffer
        test_data = np.ones(1024, dtype=np.int16)
        for _ in range(5):  # Buffer size is 5
            self.buffer.write(test_data)
        
        # Try to write when buffer is full
        self.buffer.timeout = 0.01  # Set timeout very small for faster test
        result = self.buffer.write(test_data)
        
        # Write should fail with 0 bytes written
        self.assertEqual(result, 0)
    
    def test_buffer_empty(self):
        """Test reading from empty buffer."""
        # Try to read from empty buffer
        self.buffer.timeout = 0.01  # Set timeout very small for faster test
        result = self.buffer.read()
        
        # Should return empty array
        self.assertEqual(len(result), 0)
    
    def test_close(self):
        """Test closing the buffer."""
        # Write some data
        test_data = np.ones(1024, dtype=np.int16)
        self.buffer.write(test_data)
        
        # Close buffer
        self.buffer.close()
        
        # Try to write after close
        result = self.buffer.write(test_data)
        self.assertEqual(result, 0)
        
        # Try to read after close
        result = self.buffer.read()
        self.assertEqual(len(result), 0)


class TestAudioProcessor(unittest.TestCase):
    """Test the AudioProcessor class."""
    
    def setUp(self):
        """Set up test environment."""
        self.config = {
            'audio': {
                'sample_rate': 16000,
                'channels': 1,
                'format': 'int16',
                'chunk_size': 1024
            },
            'noise_reduction': {
                'enabled': True,
                'strength': 0.7
            },
            'enhancement': {
                'enabled': True
            },
            'vad': {
                'enabled': False  # Disable for testing without dependencies
            }
        }
        self.processor = AudioProcessor(self.config)
    
    def test_initialization(self):
        """Test initialization of audio processor."""
        self.assertEqual(self.processor.sample_rate, 16000)
        self.assertEqual(self.processor.channels, 1)
        self.assertEqual(self.processor.format, 'int16')
        self.assertTrue(self.processor.nr_enabled)
        self.assertTrue(self.processor.enh_enabled)
        self.assertFalse(self.processor.vad_enabled)
    
    def test_process_audio(self):
        """Test processing audio data."""
        # Create test audio data
        test_audio = np.random.randint(-32768, 32767, 1024, dtype=np.int16)
        
        # Process audio
        processed_audio, is_speech = self.processor.process(test_audio)
        
        # Verify output
        self.assertEqual(processed_audio.dtype, np.int16)
        self.assertEqual(len(processed_audio), len(test_audio))
        self.assertFalse(is_speech)  # VAD is disabled
    
    def test_noise_reduction(self):
        """Test noise reduction processing."""
        # Create test audio with "noise"
        test_audio = np.random.normal(0, 0.01, 1024).astype(np.float32)
        
        # Apply noise reduction
        reduced_audio = self.processor.apply_noise_reduction(test_audio)
        
        # Verify output
        self.assertEqual(len(reduced_audio), len(test_audio))
        # Noise reduction should generally reduce amplitude
        self.assertLessEqual(np.mean(np.abs(reduced_audio)), np.mean(np.abs(test_audio)))
    
    def test_enhancement(self):
        """Test audio enhancement processing."""
        # Create test audio with low amplitude
        test_audio = np.random.normal(0, 0.01, 1024).astype(np.float32)
        
        # Set target level high
        self.processor.enh_target_level_db = -20
        
        # Apply enhancement
        enhanced_audio = self.processor.apply_enhancement(test_audio)
        
        # Verify output
        self.assertEqual(len(enhanced_audio), len(test_audio))
        
        # Enhancement should generally increase amplitude for low-level signals
        self.assertGreaterEqual(np.mean(np.abs(enhanced_audio)), np.mean(np.abs(test_audio)))


class TestAudioPipeline(unittest.TestCase):
    """Test the AudioPipeline class."""
    
    def setUp(self):
        """Set up test environment."""
        self.config = {
            'audio': {
                'sample_rate': 16000,
                'channels': 1,
                'format': 'int16',
                'chunk_size': 1024
            },
            'noise_reduction': {
                'enabled': True
            },
            'enhancement': {
                'enabled': True
            },
            'vad': {
                'enabled': False  # Disable for testing
            },
            'io': {
                'input_sources': [
                    {
                        'name': 'test_source',
                        'type': 'file',
                        'path': 'dummy_path.wav',
                        'loop': False
                    }
                ],
                'default_input': 'test_source',
                'stream_output': {
                    'buffer_size': 5,
                    'timeout_ms': 50
                }
            }
        }
        
        # Create pipeline
        self.pipeline = AudioPipeline()
        
        # Mock file existence check in FileSource initialization
        self.os_path_exists_patch = patch('os.path.exists')
        self.mock_path_exists = self.os_path_exists_patch.start()
        self.mock_path_exists.return_value = True
    
    def tearDown(self):
        """Clean up resources."""
        self.os_path_exists_patch.stop()
    
    def test_initialization(self):
        """Test initialization of audio pipeline."""
        # Initialize with config
        result = self.pipeline.initialize(self.config)
        
        # Verify
        self.assertTrue(result)
        self.assertTrue(self.pipeline.initialized)
        self.assertIsNotNone(self.pipeline.audio_processor)
        self.assertIsNotNone(self.pipeline.output_buffer)
        self.assertEqual(len(self.pipeline.sources), 1)
        self.assertIsNotNone(self.pipeline.active_source)
        self.assertEqual(self.pipeline.active_source.name, 'test_source')
    
    @patch('wave.open')
    def test_start_stop_capture(self, mock_wave_open):
        """Test starting and stopping audio capture."""
        # Setup mock wave file
        mock_wave = MagicMock()
        mock_wave.__enter__.return_value = mock_wave
        mock_wave.getframerate.return_value = 16000
        mock_wave.getnchannels.return_value = 1
        mock_wave.getsampwidth.return_value = 2
        mock_wave.getnframes.return_value = 16000
        
        # First call returns data, second call returns empty to simulate end of file
        mock_wave.readframes.side_effect = [
            np.ones(1024, dtype=np.int16).tobytes(),
            b''
        ]
        
        mock_wave_open.return_value = mock_wave
        
        # Initialize pipeline
        self.pipeline.initialize(self.config)
        
        # Start capture
        result = self.pipeline.start_capture()
        self.assertTrue(result)
        self.assertTrue(self.pipeline.is_running)
        
        # Allow some time for processing
        time.sleep(0.1)
        
        # Stop capture
        result = self.pipeline.stop_capture()
        self.assertTrue(result)
        self.assertFalse(self.pipeline.is_running)
    
    def test_get_status(self):
        """Test getting status."""
        # Initialize pipeline
        self.pipeline.initialize(self.config)
        
        # Get status
        status = self.pipeline.get_status()
        
        # Verify status structure
        self.assertIn('initialized', status)
        self.assertTrue(status['initialized'])
        self.assertIn('running', status)
        self.assertFalse(status['running'])
        self.assertIn('active_source', status)
        self.assertEqual(status['active_source'], 'test_source')
        self.assertIn('stats', status)
        self.assertIn('processor', status)
        self.assertIn('sources', status)
        self.assertEqual(status['sources'], 1)
    
    def test_set_quality_parameters(self):
        """Test setting quality parameters."""
        # Initialize pipeline
        self.pipeline.initialize(self.config)
        
        # Define new parameters
        new_params = {
            'noise_reduction': {
                'enabled': False,
                'strength': 0.5
            },
            'enhancement': {
                'enabled': False,
                'target_level_db': -12
            },
            'vad': {
                'enabled': True,
                'sensitivity': 1
            }
        }
        
        # Set parameters
        result = self.pipeline.set_quality_parameters(new_params)
        self.assertTrue(result)
        
        # Verify parameters were updated
        self.assertFalse(self.pipeline.audio_processor.nr_enabled)
        self.assertEqual(self.pipeline.audio_processor.nr_strength, 0.5)
        self.assertFalse(self.pipeline.audio_processor.enh_enabled)
        self.assertEqual(self.pipeline.audio_processor.enh_target_level_db, -12)
        # VAD might not actually change if webrtcvad isn't available


if __name__ == '__main__':
    unittest.main()