"""
Unit tests for the LLM Analysis module.
"""

import os
import json
import pytest
from unittest.mock import patch, MagicMock, Mock
import tempfile
from datetime import datetime

from tccc.llm_analysis.llm_analysis import (
    LLMAnalysis,
    LLMEngine,
    MedicalEntityExtractor,
    TemporalEventSequencer,
    ReportGenerator,
    ContextIntegrator
)

# Test configuration
@pytest.fixture
def test_config():
    """Create a test configuration."""
    config = {
        "model": {
            "primary": {
                "provider": "local",
                "name": "test-model",
                "path": "/tmp/test-model",
                "max_context_length": 2048,
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 512
            },
            "fallback": {
                "provider": "openai",
                "name": "gpt-3.5-turbo",
                "api_key": "test-key",
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 512
            }
        },
        "hardware": {
            "enable_acceleration": False,
            "cuda_device": -1,
            "use_tensorrt": False,
            "quantization": "4-bit",
            "cuda_streams": 1,
            "memory_limit_mb": 2048
        },
        "recommendations": {
            "enabled": True,
            "max_recommendations": 3,
            "confidence_threshold": 0.7,
            "ranking_method": "relevance",
            "categories": ["compliance", "process_efficiency"],
            "template_path": "templates/recommendations/"
        },
        "compliance": {
            "enabled": True,
            "frameworks": [
                {"name": "hipaa", "enabled": True}
            ],
            "rules_path": "config/compliance/rules/",
            "alert_threshold": 0.8,
            "detailed_reports": True
        },
        "policy_qa": {
            "enabled": True,
            "knowledge_base_paths": ["knowledge/policies/"],
            "retrieval_method": "embedding",
            "num_documents": 3,
            "embedding_model": "all-MiniLM-L6-v2"
        },
        "caching": {
            "enabled": True,
            "type": "memory",
            "ttl_seconds": 300,
            "max_size_mb": 128
        },
        "monitoring": {
            "log_prompts": False,
            "log_token_usage": True,
            "log_latency": True,
            "log_path": "logs/llm_analysis/"
        }
    }
    return config


@pytest.fixture
def mock_document_library():
    """Create a mock DocumentLibrary instance."""
    mock = MagicMock()
    
    # Mock query method
    mock.query.return_value = {
        "results": [
            {
                "text": "Sample context for testing",
                "score": 0.95,
                "metadata": {"file_name": "test_doc.txt"},
                "document_id": "test_doc_123"
            }
        ]
    }
    
    # Mock get_status method
    mock.get_status.return_value = {
        "status": "initialized",
        "documents": {"count": 5, "chunks": 100}
    }
    
    return mock


@pytest.fixture
def mock_llm_engine():
    """Create a mock LLMEngine instance."""
    mock = MagicMock()
    
    # Mock generate_text method
    def mock_generate_text(prompt, max_tokens=None, temperature=None, top_p=None, use_fallback=False):
        # Return different responses based on the prompt
        if "entity_extraction" in prompt:
            return {
                "text": """[
                    {
                        "type": "procedure",
                        "value": "intubation",
                        "time": "10:30",
                        "context": "emergency procedure"
                    },
                    {
                        "type": "measurement",
                        "value": "blood pressure",
                        "time": "10:35",
                        "context": "120/80"
                    }
                ]""",
                "model": {
                    "type": "primary",
                    "name": "test-model",
                    "provider": "local"
                },
                "metrics": {
                    "latency": 0.5
                }
            }
        elif "temporal_extraction" in prompt:
            return {
                "text": """[
                    {
                        "event_id": "1",
                        "event": "intubation",
                        "timestamp": "2023-01-01T10:30:00",
                        "relative_time": null,
                        "sequence": "first procedure",
                        "confidence": "high"
                    },
                    {
                        "event_id": "2",
                        "event": "blood pressure check",
                        "timestamp": "2023-01-01T10:35:00",
                        "relative_time": "5 minutes after intubation",
                        "sequence": "after intubation",
                        "confidence": "high"
                    }
                ]""",
                "model": {
                    "type": "primary",
                    "name": "test-model",
                    "provider": "local"
                },
                "metrics": {
                    "latency": 0.5
                }
            }
        elif "medevac" in prompt:
            return {
                "text": """MEDEVAC REQUEST

Line 1: Pickup location: Grid 123456
Line 2: Radio frequency: 68.0, Call sign: DUSTOFF
Line 3: Number of patients by precedence: 1 Urgent
Line 4: Special equipment required: None
Line 5: Number of patients by type: 1 Litter
Line 6: Security at pickup site: Secure
Line 7: Method of marking pickup site: Green smoke
Line 8: Patient nationality and status: US Military
Line 9: NBC contamination: None""",
                "model": {
                    "type": "primary",
                    "name": "test-model",
                    "provider": "local"
                },
                "metrics": {
                    "latency": 0.5
                }
            }
        elif "zmist" in prompt:
            return {
                "text": """ZMIST REPORT

Z - Mechanism of injury: Blast injury from IED
M - Injuries sustained: Penetrating trauma to right leg
I - Signs: BP 110/70, HR 110, RR 22, SpO2 95%
S - Treatment given: Tourniquet applied, morphine administered
T - Trends: Patient stabilized, vitals improving""",
                "model": {
                    "type": "primary",
                    "name": "test-model",
                    "provider": "local"
                },
                "metrics": {
                    "latency": 0.5
                }
            }
        else:
            return {
                "text": "Mock LLM response",
                "model": {
                    "type": "primary",
                    "name": "test-model",
                    "provider": "local"
                },
                "metrics": {
                    "latency": 0.5
                }
            }
            
    mock.generate_text.side_effect = mock_generate_text
    
    # Mock get_status method
    mock.get_status.return_value = {
        "models": {
            "primary": {
                "loaded": True,
                "provider": "local",
                "name": "test-model"
            },
            "fallback": {
                "loaded": True,
                "provider": "openai",
                "name": "gpt-3.5-turbo"
            }
        },
        "hardware": {
            "acceleration": False,
            "cuda_device": -1,
            "cuda_available": False,
            "quantization": "4-bit"
        }
    }
    
    return mock


@pytest.fixture
def sample_transcription():
    """Create a sample transcription for testing."""
    return {
        "text": """
        Doctor: Let's check the patient's vitals. What's the blood pressure?
        Nurse: Blood pressure is 120/80, taken at 10:35.
        Doctor: Heart rate?
        Nurse: Heart rate is 85 beats per minute.
        Doctor: The patient needs to be intubated. I performed the intubation at 10:30.
        Nurse: Confirmed. Oxygen saturation is now at 98%.
        Doctor: Let's give 5mg of morphine IV for pain management.
        Nurse: Morphine 5mg IV administered at 10:40.
        Doctor: We need to monitor for any changes over the next hour.
        """
    }


class TestLLMEngine:
    """Test the LLMEngine class."""
    
    @patch('torch.cuda.is_available', return_value=False)
    @patch('os.path.exists', return_value=True)
    def test_initialization(self, mock_path_exists, mock_cuda_available, test_config):
        """Test LLM engine initialization."""
        with patch('torch.cuda.get_device_properties') as mock_props:
            mock_props.return_value = MagicMock(total_memory=8 * 1024**3, name="Mock GPU")
            
            engine = LLMEngine(test_config)
            
            # Verify model info is set correctly
            assert engine.model_info["primary"]["name"] == test_config["model"]["primary"]["name"]
            assert engine.model_info["fallback"]["name"] == test_config["model"]["fallback"]["name"]
            
            # Verify hardware settings
            assert engine.hardware_config["cuda_device"] == -1  # Set to -1 when CUDA not available
    
    @patch('torch.cuda.is_available', return_value=False)
    @patch('os.path.exists', return_value=True)
    def test_generate_text(self, mock_path_exists, mock_cuda_available, test_config):
        """Test text generation."""
        engine = LLMEngine(test_config)
        
        # Mock the primary model's generate method
        engine.primary_model.generate = MagicMock(return_value={
            "id": "test-id",
            "choices": [{"text": "Generated text response"}]
        })
        
        # Test generation
        result = engine.generate_text("Test prompt")
        
        # Verify result
        assert "text" in result
        assert "model" in result
        assert "metrics" in result
        assert result["text"] == "Generated text response"
        assert result["model"]["type"] == "primary"
        
        # Verify primary model was called
        engine.primary_model.generate.assert_called_once()
    
    @patch('torch.cuda.is_available', return_value=False)
    @patch('os.path.exists', return_value=True)
    def test_fallback_model(self, mock_path_exists, mock_cuda_available, test_config):
        """Test fallback to secondary model."""
        engine = LLMEngine(test_config)
        
        # Mock the primary model to raise an exception
        engine.primary_model.generate = MagicMock(side_effect=RuntimeError("Primary model failed"))
        
        # Mock the fallback model
        engine.fallback_model.generate = MagicMock(return_value={
            "id": "test-id",
            "choices": [{"text": "Fallback model response"}]
        })
        
        # Test generation
        result = engine.generate_text("Test prompt")
        
        # Verify result
        assert result["text"] == "Fallback model response"
        assert result["model"]["type"] == "fallback"
        
        # Verify both models were called
        engine.primary_model.generate.assert_called_once()
        engine.fallback_model.generate.assert_called_once()


class TestMedicalEntityExtractor:
    """Test the MedicalEntityExtractor class."""
    
    def test_initialization(self, mock_llm_engine, test_config):
        """Test extractor initialization."""
        extractor = MedicalEntityExtractor(mock_llm_engine, test_config)
        
        # Verify prompt templates were loaded
        assert "entity_extraction" in extractor.prompt_templates
        assert "temporal_extraction" in extractor.prompt_templates
        assert "vital_signs" in extractor.prompt_templates
        assert "medication" in extractor.prompt_templates
        assert "procedures" in extractor.prompt_templates
    
    def test_extract_entities(self, mock_llm_engine, test_config, sample_transcription):
        """Test entity extraction."""
        extractor = MedicalEntityExtractor(mock_llm_engine, test_config)
        
        entities = extractor.extract_entities(sample_transcription["text"])
        
        # Verify correct entities were extracted
        assert len(entities) == 2
        assert entities[0]["type"] == "procedure"
        assert entities[0]["value"] == "intubation"
        assert entities[1]["type"] == "measurement"
        assert entities[1]["value"] == "blood pressure"
        
        # Verify LLM was called with the correct prompt
        mock_llm_engine.generate_text.assert_called_with(
            extractor._render_prompt("entity_extraction", transcription=sample_transcription["text"])
        )
    
    def test_extract_all(self, mock_llm_engine, test_config, sample_transcription):
        """Test extracting all medical information."""
        extractor = MedicalEntityExtractor(mock_llm_engine, test_config)
        
        all_info = extractor.extract_all(sample_transcription["text"])
        
        # Verify all categories were extracted
        assert "entities" in all_info
        assert "temporal" in all_info
        assert "vitals" in all_info
        assert "medications" in all_info
        assert "procedures" in all_info
        
        # Verify LLM was called for each category
        assert mock_llm_engine.generate_text.call_count == 5


class TestTemporalEventSequencer:
    """Test the TemporalEventSequencer class."""
    
    def test_sequence_events_with_timestamps(self):
        """Test sequencing events with explicit timestamps."""
        sequencer = TemporalEventSequencer()
        
        events = [
            {
                "event": "blood pressure check",
                "timestamp": "2023-01-01T10:35:00"
            },
            {
                "event": "intubation",
                "timestamp": "2023-01-01T10:30:00"
            }
        ]
        
        sequenced = sequencer.sequence_events(events)
        
        # Verify correct sequencing by timestamp
        assert sequenced[0]["event"] == "intubation"
        assert sequenced[1]["event"] == "blood pressure check"
        
        # Verify sequence metadata was added
        assert "sequence_metadata" in sequenced[0]
        assert sequenced[0]["sequence_metadata"]["position"] == 0
        assert sequenced[1]["sequence_metadata"]["position"] == 1
    
    def test_sequence_events_with_relative_time(self):
        """Test sequencing events with relative time references."""
        sequencer = TemporalEventSequencer()
        
        events = [
            {
                "event": "administer medication",
                "relative_time": "after intubation"
            },
            {
                "event": "intubation",
                "relative_time": "upon arrival"
            },
            {
                "event": "patient transport",
                "relative_time": "20 minutes later"
            }
        ]
        
        sequenced = sequencer.sequence_events(events)
        
        # Verify sequencing based on relative references
        assert sequenced[0]["event"] == "intubation"
        assert sequenced[1]["event"] == "administer medication"
        assert sequenced[2]["event"] == "patient transport"


class TestReportGenerator:
    """Test the ReportGenerator class."""
    
    def test_initialization(self, mock_llm_engine, test_config):
        """Test report generator initialization."""
        generator = ReportGenerator(mock_llm_engine, test_config)
        
        # Verify report templates were loaded
        assert "medevac" in generator.report_templates
        assert "zmist" in generator.report_templates
        assert "soap" in generator.report_templates
        assert "tccc" in generator.report_templates
    
    def test_generate_medevac_report(self, mock_llm_engine, test_config):
        """Test generating a MEDEVAC report."""
        generator = ReportGenerator(mock_llm_engine, test_config)
        
        events = [
            {
                "type": "procedure",
                "value": "intubation",
                "time": "10:30"
            },
            {
                "type": "measurement",
                "value": "blood pressure",
                "time": "10:35",
                "context": "120/80"
            }
        ]
        
        report = generator.generate_report("medevac", events)
        
        # Verify report structure
        assert report["report_type"] == "medevac"
        assert "content" in report
        assert "Line 1:" in report["content"]
        assert "Line 9:" in report["content"]
        assert "generated_at" in report
        assert report["events_count"] == 2
    
    def test_generate_zmist_report(self, mock_llm_engine, test_config):
        """Test generating a ZMIST report."""
        generator = ReportGenerator(mock_llm_engine, test_config)
        
        events = [
            {
                "type": "injury",
                "value": "penetrating trauma",
                "location": "right leg"
            },
            {
                "type": "vital",
                "value": "blood pressure",
                "reading": "110/70"
            }
        ]
        
        report = generator.generate_report("zmist", events)
        
        # Verify report structure
        assert report["report_type"] == "zmist"
        assert "content" in report
        assert "Z -" in report["content"]
        assert "T -" in report["content"]
        assert "generated_at" in report
        assert report["events_count"] == 2


class TestContextIntegrator:
    """Test the ContextIntegrator class."""
    
    def test_get_relevant_context(self, mock_document_library, test_config):
        """Test getting relevant context."""
        integrator = ContextIntegrator(mock_document_library, test_config)
        
        context = integrator.get_relevant_context("intubation procedure", n_results=1)
        
        # Verify context was retrieved
        assert len(context) == 1
        assert "text" in context[0]
        assert "score" in context[0]
        assert "source" in context[0]
        assert "document_id" in context[0]
        
        # Verify document library was queried
        mock_document_library.query.assert_called_with("intubation procedure", n_results=1)
    
    def test_enhance_with_context(self, mock_document_library, test_config):
        """Test enhancing events with context."""
        integrator = ContextIntegrator(mock_document_library, test_config)
        
        events = [
            {
                "type": "procedure",
                "value": "intubation"
            },
            {
                "type": "measurement",
                "value": "blood pressure"
            }
        ]
        
        enhanced = integrator.enhance_with_context(events)
        
        # Verify events were enhanced with context
        assert len(enhanced) == 2
        assert "context_reference" in enhanced[0]
        assert enhanced[0]["context_reference"]["text"] == "Sample context for testing"
        assert enhanced[0]["context_reference"]["score"] == 0.95


class TestLLMAnalysis:
    """Test the LLMAnalysis class."""
    
    @patch('tccc.llm_analysis.llm_analysis.LLMEngine')
    @patch('tccc.llm_analysis.llm_analysis.MedicalEntityExtractor')
    @patch('tccc.llm_analysis.llm_analysis.TemporalEventSequencer')
    @patch('tccc.llm_analysis.llm_analysis.ReportGenerator')
    @patch('tccc.document_library.DocumentLibrary')
    @patch('tccc.llm_analysis.llm_analysis.Config')
    def test_initialization(self, mock_config, mock_doc_lib, mock_report_gen, 
                            mock_sequencer, mock_extractor, mock_engine, test_config):
        """Test LLM analysis initialization."""
        # Set up mock config
        mock_config.return_value.get.return_value = {}
        
        # Initialize module
        llm_analysis = LLMAnalysis()
        success = llm_analysis.initialize(test_config)
        
        # Verify initialization
        assert success is True
        assert llm_analysis.initialized is True
        
        # Verify components were initialized
        mock_engine.assert_called_once_with(test_config)
        mock_extractor.assert_called_once()
        mock_sequencer.assert_called_once()
        mock_report_gen.assert_called_once()
        
        # Verify document library was initialized (since policy_qa is enabled)
        mock_doc_lib.assert_called_once()
        mock_doc_lib.return_value.initialize.assert_called_once()
    
    @patch('tccc.llm_analysis.llm_analysis.LLMEngine')
    @patch('tccc.llm_analysis.llm_analysis.MedicalEntityExtractor')
    @patch('tccc.llm_analysis.llm_analysis.TemporalEventSequencer')
    @patch('tccc.llm_analysis.llm_analysis.ReportGenerator')
    def test_process_transcription(self, mock_report_gen, mock_sequencer, 
                                  mock_extractor, mock_engine, test_config, sample_transcription):
        """Test processing a transcription."""
        # Set up mocks
        mock_engine.return_value = MagicMock()
        mock_extractor.return_value = MagicMock()
        mock_extractor.return_value.extract_all.return_value = {
            "entities": [{"type": "procedure", "value": "intubation"}],
            "temporal": [{"event": "intubation", "timestamp": "2023-01-01T10:30:00"}],
            "vitals": [{"type": "blood_pressure", "value": "120/80"}],
            "medications": [{"name": "morphine", "dosage": "5mg"}],
            "procedures": [{"name": "intubation", "status": "completed"}]
        }
        
        mock_sequencer.return_value = MagicMock()
        mock_sequencer.return_value.sequence_events.return_value = [
            {"type": "procedure", "value": "intubation", "category": "entities"},
            {"type": "blood_pressure", "value": "120/80", "category": "vitals"},
            {"name": "morphine", "dosage": "5mg", "category": "medications"}
        ]
        
        # Initialize and process transcription
        llm_analysis = LLMAnalysis()
        llm_analysis.initialize(test_config)
        
        # Disable context integration for this test
        llm_analysis.context_integrator = None
        
        results = llm_analysis.process_transcription(sample_transcription)
        
        # Verify results
        assert len(results) == 3
        
        # Verify extract_all was called
        mock_extractor.return_value.extract_all.assert_called_with(sample_transcription["text"])
        
        # Verify sequence_events was called
        mock_sequencer.return_value.sequence_events.assert_called_once()
    
    @patch('tccc.llm_analysis.llm_analysis.LLMEngine')
    @patch('tccc.llm_analysis.llm_analysis.MedicalEntityExtractor')
    @patch('tccc.llm_analysis.llm_analysis.TemporalEventSequencer')
    @patch('tccc.llm_analysis.llm_analysis.ReportGenerator')
    def test_generate_report(self, mock_report_gen, mock_sequencer, 
                           mock_extractor, mock_engine, test_config):
        """Test generating a report."""
        # Set up mocks
        mock_engine.return_value = MagicMock()
        mock_report_gen.return_value = MagicMock()
        mock_report_gen.return_value.generate_report.return_value = {
            "report_type": "medevac",
            "content": "MEDEVAC REPORT...",
            "generated_at": datetime.now().isoformat(),
            "events_count": 3,
            "model": {"type": "primary", "name": "test-model"}
        }
        
        # Initialize and generate report
        llm_analysis = LLMAnalysis()
        llm_analysis.initialize(test_config)
        
        events = [
            {"type": "procedure", "value": "intubation"},
            {"type": "measurement", "value": "blood pressure"}
        ]
        
        report = llm_analysis.generate_report("medevac", events)
        
        # Verify report
        assert report["report_type"] == "medevac"
        assert "content" in report
        assert "generated_at" in report
        assert report["events_count"] == 3
        
        # Verify generate_report was called
        mock_report_gen.return_value.generate_report.assert_called_with("medevac", events)
    
    @patch('tccc.llm_analysis.llm_analysis.LLMEngine')
    @patch('tccc.llm_analysis.llm_analysis.MedicalEntityExtractor')
    @patch('tccc.llm_analysis.llm_analysis.TemporalEventSequencer')
    @patch('tccc.llm_analysis.llm_analysis.ReportGenerator')
    def test_get_status(self, mock_report_gen, mock_sequencer, 
                      mock_extractor, mock_engine, test_config):
        """Test getting module status."""
        # Set up mocks
        mock_engine.return_value = MagicMock()
        mock_engine.return_value.get_status.return_value = {
            "models": {
                "primary": {"loaded": True},
                "fallback": {"loaded": True}
            },
            "hardware": {"acceleration": False}
        }
        
        # Initialize and get status
        llm_analysis = LLMAnalysis()
        llm_analysis.initialize(test_config)
        
        status = llm_analysis.get_status()
        
        # Verify status
        assert status["initialized"] is True
        assert status["cache_enabled"] is True
        assert "llm_engine" in status
        assert status["llm_engine"]["models"]["primary"]["loaded"] is True