"""
Unit tests for the Visualization module.
"""

import pytest
import numpy as np
from pathlib import Path
import sys
import tempfile
import os

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from bayesian_network import BayesianNetwork, Node, CPT
from visualization import NetworkVisualizer
from inference_engine import InferenceEngine, InferenceResult


class TestNetworkVisualizer:
    """Test cases for NetworkVisualizer class."""
    
    def setup_method(self):
        """Setup method to create a test network."""
        self.network = BayesianNetwork("Test Network")
        
        # Create nodes
        node_a = Node("A", ["True", "False"])
        node_b = Node("B", ["True", "False"])
        node_c = Node("C", ["True", "False"])
        
        self.network.add_node(node_a)
        self.network.add_node(node_b)
        self.network.add_node(node_c)
        
        # Create edges: A -> B -> C
        self.network.add_edge("A", "B")
        self.network.add_edge("B", "C")
        
        # Create CPTs
        cpt_a = CPT("A", [], {"True": 0.6, "False": 0.4})
        cpt_b = CPT("B", ["A"], {
            "True|True": 0.8, "False|True": 0.2,
            "True|False": 0.3, "False|False": 0.7
        })
        cpt_c = CPT("C", ["B"], {
            "True|True": 0.7, "False|True": 0.3,
            "True|False": 0.2, "False|False": 0.8
        })
        
        self.network.set_cpt(cpt_a)
        self.network.set_cpt(cpt_b)
        self.network.set_cpt(cpt_c)
        
        # Create visualizer
        self.visualizer = NetworkVisualizer(self.network)
    
    def test_visualizer_creation(self):
        """Test NetworkVisualizer creation."""
        assert self.visualizer.network == self.network
    
    def test_setup_plotting_style(self):
        """Test plotting style setup."""
        # This should not raise any errors
        self.visualizer.setup_plotting_style()
    
    def test_plot_network_topology(self):
        """Test network topology plotting."""
        fig = self.visualizer.plot_network_topology()
        
        assert fig is not None
        assert hasattr(fig, 'data')
        assert len(fig.data) > 0
        
        # Test with different layouts
        layouts = ["spring", "circular", "kamada_kawai", "shell"]
        for layout in layouts:
            fig = self.visualizer.plot_network_topology(layout=layout)
            assert fig is not None
    
    def test_plot_network_topology_with_colors(self):
        """Test network topology plotting with custom colors."""
        node_colors = {"A": "#ff0000", "B": "#00ff00", "C": "#0000ff"}
        edge_colors = {("A", "B"): "#ff00ff", ("B", "C"): "#ffff00"}
        
        fig = self.visualizer.plot_network_topology(
            node_colors=node_colors,
            edge_colors=edge_colors
        )
        
        assert fig is not None
    
    def test_plot_network_topology_save(self):
        """Test network topology plotting with save."""
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as tmp_file:
            save_path = tmp_file.name
        
        try:
            fig = self.visualizer.plot_network_topology(save_path=save_path)
            assert fig is not None
            assert os.path.exists(save_path)
        finally:
            if os.path.exists(save_path):
                os.remove(save_path)
    
    def test_plot_cpt_heatmap(self):
        """Test CPT heatmap plotting."""
        # Test for root node
        fig = self.visualizer.plot_cpt_heatmap("A")
        assert fig is not None
        assert hasattr(fig, 'data')
        
        # Test for node with parents
        fig = self.visualizer.plot_cpt_heatmap("B")
        assert fig is not None
        assert hasattr(fig, 'data')
    
    def test_plot_cpt_heatmap_nonexistent_node(self):
        """Test CPT heatmap plotting for non-existent node."""
        with pytest.raises(ValueError, match="No CPT found"):
            self.visualizer.plot_cpt_heatmap("NonExistent")
    
    def test_plot_cpt_heatmap_save(self):
        """Test CPT heatmap plotting with save."""
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as tmp_file:
            save_path = tmp_file.name
        
        try:
            fig = self.visualizer.plot_cpt_heatmap("A", save_path=save_path)
            assert fig is not None
            assert os.path.exists(save_path)
        finally:
            if os.path.exists(save_path):
                os.remove(save_path)
    
    def test_plot_inference_results(self):
        """Test inference results plotting."""
        # Create a mock inference result
        result = InferenceResult(
            query_variables=["A", "B"],
            evidence={"C": "True"},
            probabilities={
                "A": {"True": 0.6, "False": 0.4},
                "B": {"True": 0.7, "False": 0.3}
            },
            inference_time=0.1,
            method="VariableElimination"
        )
        
        # Test different plot types
        plot_types = ["bar", "heatmap", "radar"]
        for plot_type in plot_types:
            fig = self.visualizer.plot_inference_results(result, plot_type=plot_type)
            assert fig is not None
            assert hasattr(fig, 'data')
    
    def test_plot_inference_results_invalid_type(self):
        """Test inference results plotting with invalid plot type."""
        result = InferenceResult(
            query_variables=["A"],
            evidence={},
            probabilities={"A": {"True": 0.6, "False": 0.4}},
            inference_time=0.1,
            method="VariableElimination"
        )
        
        with pytest.raises(ValueError, match="Unsupported plot type"):
            self.visualizer.plot_inference_results(result, plot_type="invalid")
    
    def test_plot_inference_results_save(self):
        """Test inference results plotting with save."""
        result = InferenceResult(
            query_variables=["A"],
            evidence={},
            probabilities={"A": {"True": 0.6, "False": 0.4}},
            inference_time=0.1,
            method="VariableElimination"
        )
        
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as tmp_file:
            save_path = tmp_file.name
        
        try:
            fig = self.visualizer.plot_inference_results(result, save_path=save_path)
            assert fig is not None
            assert os.path.exists(save_path)
        finally:
            if os.path.exists(save_path):
                os.remove(save_path)
    
    def test_plot_belief_propagation(self):
        """Test belief propagation plotting."""
        try:
            # Create inference engine
            engine = InferenceEngine(self.network)
            
            # Test belief propagation plotting
            evidence = {"A": "True"}
            fig = self.visualizer.plot_belief_propagation(engine, evidence, steps=3)
            
            assert fig is not None
            assert hasattr(fig, 'data')
            assert hasattr(fig, 'frames')  # Should have animation frames
            
        except ImportError:
            pytest.skip("pgmpy not available")
    
    def test_plot_sensitivity_analysis(self):
        """Test sensitivity analysis plotting."""
        sensitivity_results = {
            "True": 0.8,
            "False": 0.3
        }
        
        fig = self.visualizer.plot_sensitivity_analysis(
            sensitivity_results, "C", "True"
        )
        
        assert fig is not None
        assert hasattr(fig, 'data')
    
    def test_plot_sensitivity_analysis_save(self):
        """Test sensitivity analysis plotting with save."""
        sensitivity_results = {
            "True": 0.8,
            "False": 0.3
        }
        
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as tmp_file:
            save_path = tmp_file.name
        
        try:
            fig = self.visualizer.plot_sensitivity_analysis(
                sensitivity_results, "C", "True", save_path=save_path
            )
            assert fig is not None
            assert os.path.exists(save_path)
        finally:
            if os.path.exists(save_path):
                os.remove(save_path)
    
    def test_create_dashboard(self):
        """Test dashboard creation."""
        # Create mock inference results
        results = [
            InferenceResult(
                query_variables=["A", "B"],
                evidence={},
                probabilities={
                    "A": {"True": 0.6, "False": 0.4},
                    "B": {"True": 0.7, "False": 0.3}
                },
                inference_time=0.1,
                method="VariableElimination"
            ),
            InferenceResult(
                query_variables=["A", "B"],
                evidence={},
                probabilities={
                    "A": {"True": 0.5, "False": 0.5},
                    "B": {"True": 0.6, "False": 0.4}
                },
                inference_time=0.2,
                method="BeliefPropagation"
            )
        ]
        
        fig = self.visualizer.create_dashboard(results)
        
        assert fig is not None
        assert hasattr(fig, 'data')
    
    def test_create_dashboard_save(self):
        """Test dashboard creation with save."""
        results = [
            InferenceResult(
                query_variables=["A"],
                evidence={},
                probabilities={"A": {"True": 0.6, "False": 0.4}},
                inference_time=0.1,
                method="VariableElimination"
            )
        ]
        
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as tmp_file:
            save_path = tmp_file.name
        
        try:
            fig = self.visualizer.create_dashboard(results, save_path=save_path)
            assert fig is not None
            assert os.path.exists(save_path)
        finally:
            if os.path.exists(save_path):
                os.remove(save_path)
    
    def test_save_all_visualizations(self):
        """Test saving all visualizations."""
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock inference result
            result = InferenceResult(
                query_variables=["A"],
                evidence={},
                probabilities={"A": {"True": 0.6, "False": 0.4}},
                inference_time=0.1,
                method="VariableElimination"
            )
            
            # Save all visualizations
            saved_files = self.visualizer.save_all_visualizations(
                inference_result=result,
                output_dir=temp_dir
            )
            
            # Check that files were saved
            assert len(saved_files) > 0
            
            # Check that files exist
            for file_path in saved_files.values():
                assert os.path.exists(file_path)
            
            # Check specific file types
            assert "network_topology" in saved_files
            assert "cpt_A" in saved_files
            assert "cpt_B" in saved_files
            assert "cpt_C" in saved_files
            assert "inference_bar" in saved_files
            assert "inference_heatmap" in saved_files
    
    def test_save_all_visualizations_no_inference(self):
        """Test saving all visualizations without inference result."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save all visualizations without inference result
            saved_files = self.visualizer.save_all_visualizations(
                output_dir=temp_dir
            )
            
            # Check that files were saved
            assert len(saved_files) > 0
            
            # Check that files exist
            for file_path in saved_files.values():
                assert os.path.exists(file_path)
            
            # Should not have inference result files
            assert "inference_bar" not in saved_files
            assert "inference_heatmap" not in saved_files


class TestVisualizationWithComplexNetwork:
    """Test visualization with a more complex network."""
    
    def setup_method(self):
        """Setup method to create a complex test network."""
        self.network = BayesianNetwork("Complex Network")
        
        # Create nodes with multiple states
        node_a = Node("A", ["Low", "Medium", "High"])
        node_b = Node("B", ["True", "False"])
        node_c = Node("C", ["X", "Y", "Z"])
        
        self.network.add_node(node_a)
        self.network.add_node(node_b)
        self.network.add_node(node_c)
        
        # Create edges
        self.network.add_edge("A", "B")
        self.network.add_edge("B", "C")
        
        # Create complex CPTs
        cpt_a = CPT("A", [], {"Low": 0.3, "Medium": 0.5, "High": 0.2})
        cpt_b = CPT("B", ["A"], {
            "True|Low": 0.2, "False|Low": 0.8,
            "True|Medium": 0.5, "False|Medium": 0.5,
            "True|High": 0.8, "False|High": 0.2
        })
        cpt_c = CPT("C", ["B"], {
            "X|True": 0.4, "Y|True": 0.4, "Z|True": 0.2,
            "X|False": 0.2, "Y|False": 0.3, "Z|False": 0.5
        })
        
        self.network.set_cpt(cpt_a)
        self.network.set_cpt(cpt_b)
        self.network.set_cpt(cpt_c)
        
        # Create visualizer
        self.visualizer = NetworkVisualizer(self.network)
    
    def test_complex_network_topology(self):
        """Test topology plotting with complex network."""
        fig = self.visualizer.plot_network_topology()
        assert fig is not None
        assert hasattr(fig, 'data')
    
    def test_complex_cpt_heatmap(self):
        """Test CPT heatmap with complex network."""
        # Test for node with multiple states
        fig = self.visualizer.plot_cpt_heatmap("A")
        assert fig is not None
        
        # Test for node with complex conditional probabilities
        fig = self.visualizer.plot_cpt_heatmap("C")
        assert fig is not None
    
    def test_complex_inference_results(self):
        """Test inference results with complex network."""
        result = InferenceResult(
            query_variables=["A", "B", "C"],
            evidence={},
            probabilities={
                "A": {"Low": 0.3, "Medium": 0.5, "High": 0.2},
                "B": {"True": 0.6, "False": 0.4},
                "C": {"X": 0.3, "Y": 0.4, "Z": 0.3}
            },
            inference_time=0.1,
            method="VariableElimination"
        )
        
        # Test different plot types
        for plot_type in ["bar", "heatmap", "radar"]:
            fig = self.visualizer.plot_inference_results(result, plot_type=plot_type)
            assert fig is not None


class TestVisualizationErrorHandling:
    """Test error handling in visualization."""
    
    def setup_method(self):
        """Setup method to create a test network."""
        self.network = BayesianNetwork("Test Network")
        node = Node("A", ["True", "False"])
        self.network.add_node(node)
        self.visualizer = NetworkVisualizer(self.network)
    
    def test_plot_cpt_heatmap_no_cpt(self):
        """Test CPT heatmap plotting when CPT doesn't exist."""
        with pytest.raises(ValueError, match="No CPT found"):
            self.visualizer.plot_cpt_heatmap("A")
    
    def test_plot_inference_results_empty(self):
        """Test inference results plotting with empty result."""
        result = InferenceResult(
            query_variables=[],
            evidence={},
            probabilities={},
            inference_time=0.0,
            method="VariableElimination"
        )
        
        # Should handle empty results gracefully
        fig = self.visualizer.plot_inference_results(result)
        assert fig is not None


if __name__ == "__main__":
    pytest.main([__file__]) 