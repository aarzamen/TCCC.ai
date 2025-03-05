"""
Unit tests for the module registration system.
"""

import pytest
import time
import asyncio
from typing import Dict, Any, List, Optional
from unittest.mock import MagicMock, patch

from tccc.processing_core.processing_core import ProcessingCore, ModuleState, ModuleInfo


class MockModule:
    """Mock module for testing."""
    
    def __init__(self, name, config=None):
        self.name = name
        self.config = config or {}
        self.initialized = False
        self.shutdown_called = False
    
    def initialize(self, config=None):
        """Initialize the mock module."""
        self.initialized = True
        return True
    
    def shutdown(self):
        """Shutdown the mock module."""
        self.shutdown_called = True


@pytest.fixture
def core():
    """Create a ProcessingCore instance without initialization."""
    return ProcessingCore()


class TestModuleRegistration:
    """Tests for the module registration system."""
    
    def test_register_module(self, core):
        """Test registering a module."""
        # Create mock module
        module = MockModule("test_module")
        
        # Register the module
        success = core.register_module(
            name="test_module",
            module_type="test",
            instance=module
        )
        
        # Verify registration was successful
        assert success
        assert "test_module" in core.modules
        assert core.modules["test_module"].instance == module
        assert core.modules["test_module"].state == ModuleState.UNINITIALIZED
    
    def test_register_module_with_dependencies(self, core):
        """Test registering a module with dependencies."""
        # Create and register dependency module
        dep_module = MockModule("dependency")
        core.register_module(
            name="dependency",
            module_type="dep",
            instance=dep_module
        )
        
        # Create and register module with dependency
        module = MockModule("dependent")
        success = core.register_module(
            name="dependent",
            module_type="test",
            instance=module,
            dependencies=["dependency"]
        )
        
        # Verify registration was successful
        assert success
        assert "dependent" in core.modules
        
        # Verify dependency graph was updated
        assert core.dependency_graph.has_edge("dependency", "dependent")
    
    def test_register_module_with_missing_dependency(self, core):
        """Test registering a module with a missing dependency."""
        # Create module
        module = MockModule("test_module")
        
        # Register module with nonexistent dependency
        success = core.register_module(
            name="test_module",
            module_type="test",
            instance=module,
            dependencies=["nonexistent"]
        )
        
        # Verify registration was successful but warning generated
        assert success
        assert "test_module" in core.modules
        
        # Verify dependency graph has node but no edge
        assert core.dependency_graph.has_node("test_module")
        assert not core.dependency_graph.has_edge("nonexistent", "test_module")
    
    def test_get_module(self, core):
        """Test getting a module by name."""
        # Create and register module
        module = MockModule("test_module")
        core.register_module(
            name="test_module",
            module_type="test",
            instance=module
        )
        
        # Get module
        retrieved = core.get_module("test_module")
        
        # Verify module was retrieved correctly
        assert retrieved == module
        
        # Try to get nonexistent module
        nonexistent = core.get_module("nonexistent")
        assert nonexistent is None
    
    def test_get_module_info(self, core):
        """Test getting module info by name."""
        # Create and register module
        module = MockModule("test_module")
        core.register_module(
            name="test_module",
            module_type="test",
            instance=module
        )
        
        # Get module info
        info = core.get_module_info("test_module")
        
        # Verify module info was retrieved correctly
        assert info.name == "test_module"
        assert info.module_type == "test"
        assert info.instance == module
        
        # Try to get nonexistent module info
        nonexistent = core.get_module_info("nonexistent")
        assert nonexistent is None
    
    def test_module_state_transitions(self, core):
        """Test module state transitions."""
        # Create and register module
        module = MockModule("test_module")
        core.register_module(
            name="test_module",
            module_type="test",
            instance=module
        )
        
        # Verify initial state
        assert core.modules["test_module"].state == ModuleState.UNINITIALIZED
        
        # Update state to INITIALIZING
        core.modules["test_module"].update_state(ModuleState.INITIALIZING)
        assert core.modules["test_module"].state == ModuleState.INITIALIZING
        
        # Update state to READY
        core.modules["test_module"].update_state(ModuleState.READY)
        assert core.modules["test_module"].state == ModuleState.READY
        
        # Update state to ACTIVE
        core.modules["test_module"].update_state(ModuleState.ACTIVE)
        assert core.modules["test_module"].state == ModuleState.ACTIVE
        
        # Update state to ERROR with message
        core.modules["test_module"].update_state(ModuleState.ERROR, "Error message")
        assert core.modules["test_module"].state == ModuleState.ERROR
        assert core.modules["test_module"].state_message == "Error message"
    
    def test_get_module_status(self, core):
        """Test getting status for all modules."""
        # Create and register modules
        for i in range(3):
            module = MockModule(f"module_{i}")
            core.register_module(
                name=f"module_{i}",
                module_type="test",
                instance=module
            )
            
            # Update state to different values
            if i == 0:
                core.modules[f"module_{i}"].update_state(ModuleState.READY)
            elif i == 1:
                core.modules[f"module_{i}"].update_state(ModuleState.ACTIVE)
            else:
                core.modules[f"module_{i}"].update_state(ModuleState.ERROR, "Test error")
        
        # Get module status
        status = core.get_module_status()
        
        # Verify status contains all modules
        assert len(status) == 3
        assert "module_0" in status
        assert "module_1" in status
        assert "module_2" in status
        
        # Verify state values
        assert status["module_0"]["state"] == "READY"
        assert status["module_1"]["state"] == "ACTIVE"
        assert status["module_2"]["state"] == "ERROR"
        assert status["module_2"]["state_message"] == "Test error"
    
    def test_unregister_module(self, core):
        """Test unregistering a module."""
        # Create and register module
        module = MockModule("test_module")
        core.register_module(
            name="test_module",
            module_type="test",
            instance=module
        )
        
        # Unregister module
        success = core.unregister_module("test_module")
        
        # Verify unregistration was successful
        assert success
        assert "test_module" not in core.modules
        assert not core.dependency_graph.has_node("test_module")
    
    def test_unregister_dependent_module(self, core):
        """Test attempt to unregister a module with dependents."""
        # Create and register dependency module
        dep_module = MockModule("dependency")
        core.register_module(
            name="dependency",
            module_type="dep",
            instance=dep_module
        )
        
        # Create and register module with dependency
        module = MockModule("dependent")
        core.register_module(
            name="dependent",
            module_type="test",
            instance=module,
            dependencies=["dependency"]
        )
        
        # Try to unregister dependency module
        success = core.unregister_module("dependency")
        
        # Verify unregistration failed
        assert not success
        assert "dependency" in core.modules
        assert core.dependency_graph.has_node("dependency")
    
    def test_get_initialization_order(self, core):
        """Test getting initialization order based on dependencies."""
        # Create dependency structure:
        # A -> B -> D
        # A -> C -> D
        # E (independent)
        
        for name in ["A", "B", "C", "D", "E"]:
            module = MockModule(name)
            core.register_module(
                name=name,
                module_type="test",
                instance=module
            )
        
        # Set dependencies
        core.modules["B"].dependencies = ["A"]
        core.dependency_graph.add_edge("A", "B")
        
        core.modules["C"].dependencies = ["A"]
        core.dependency_graph.add_edge("A", "C")
        
        core.modules["D"].dependencies = ["B", "C"]
        core.dependency_graph.add_edge("B", "D")
        core.dependency_graph.add_edge("C", "D")
        
        # Get initialization order
        order = core.get_initialization_order()
        
        # Verify order respects dependencies
        # Either A -> B -> C -> D -> E or A -> C -> B -> D -> E or other valid topological order
        a_index = order.index("A")
        b_index = order.index("B")
        c_index = order.index("C")
        d_index = order.index("D")
        
        # A must come before B and C
        assert a_index < b_index and a_index < c_index
        
        # B and C must come before D
        assert b_index < d_index and c_index < d_index
    
    def test_circular_dependency_handling(self, core):
        """Test handling of circular dependencies."""
        # Create modules
        for name in ["A", "B", "C"]:
            module = MockModule(name)
            core.register_module(
                name=name,
                module_type="test",
                instance=module
            )
        
        # Create circular dependency: A -> B -> C -> A
        core.modules["B"].dependencies = ["A"]
        core.dependency_graph.add_edge("A", "B")
        
        core.modules["C"].dependencies = ["B"]
        core.dependency_graph.add_edge("B", "C")
        
        core.modules["A"].dependencies = ["C"]
        core.dependency_graph.add_edge("C", "A")
        
        # Get initialization order - should return a list of all modules since
        # no valid topological sort exists
        order = core.get_initialization_order()
        
        # Verify all modules are included
        assert len(order) == 3
        assert "A" in order
        assert "B" in order
        assert "C" in order
    
    @pytest.mark.asyncio
    async def test_module_shutdown_order(self, core):
        """Test proper shutdown order based on dependencies."""
        # Create modules with dependency chain: A -> B -> C
        modules = {}
        
        for name in ["A", "B", "C"]:
            module = MockModule(name)
            modules[name] = module
            core.register_module(
                name=name,
                module_type="test",
                instance=module
            )
        
        # Set dependencies
        core.modules["B"].dependencies = ["A"]
        core.dependency_graph.add_edge("A", "B")
        
        core.modules["C"].dependencies = ["B"]
        core.dependency_graph.add_edge("B", "C")
        
        # Simulate initialization
        for name in ["A", "B", "C"]:
            core.modules[name].update_state(ModuleState.READY)
        
        # Set up a list to track shutdown order
        shutdown_order = []
        
        # Mock the shutdown method of each module to track order
        def make_shutdown_mock(name):
            def mock_shutdown():
                shutdown_order.append(name)
            return mock_shutdown
        
        for name, module in modules.items():
            module.shutdown = make_shutdown_mock(name)
        
        # Shut down the core
        core.shutdown()
        
        # Verify shutdown order is reverse of dependency order (C -> B -> A)
        assert shutdown_order[0] == "C"
        assert shutdown_order[1] == "B"
        assert shutdown_order[2] == "A"
        
    def test_module_info_serialization(self):
        """Test serialization of ModuleInfo objects."""
        # Create ModuleInfo
        module = MockModule("test")
        info = ModuleInfo(
            name="test_module",
            module_type="test",
            instance=module,
            dependencies=["dep1", "dep2"],
            state=ModuleState.READY,
            state_message="Test message"
        )
        
        # Convert to dictionary
        info_dict = info.to_dict()
        
        # Verify dictionary contains expected fields
        assert info_dict["name"] == "test_module"
        assert info_dict["type"] == "test"
        assert info_dict["state"] == "READY"
        assert info_dict["state_message"] == "Test message"
        assert "dep1" in info_dict["dependencies"]
        assert "dep2" in info_dict["dependencies"]
        
        # Verify instance is not serialized
        assert "instance" not in info_dict