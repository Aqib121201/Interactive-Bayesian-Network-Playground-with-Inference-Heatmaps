"""
Unit tests for the Inference Engine module.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from bayesian_network import BayesianNetwork, Node, CPT
from inference_engine import InferenceEngine, InferenceResult


class TestInferenceResult:
    """Test cases for InferenceResult class."""
    
    def test_inference_result_creation(self):
        """Test basic InferenceResult creation."""
        query_variables = ["A", "B"]
        evidence = {"C": "True"}
        probabilities = {
            "A": {"True": 0.6, "False": 0.4},
            "B": {"True": 0.7, "False": 0.3}
        }
        
        result = InferenceResult(
            query_variables=query_variables,
            evidence=evidence,
            probabilities=probabilities,
            inference_time=0.1,
            method="VariableElimination"
        )
        
        assert result.query_variables == query_variables
        assert result.evidence == evidence
        assert result.probabilities == probabilities
        assert result.inference_time == 0.1
        assert result.method == "VariableElimination"
    
    def test_to_dataframe(self):
        """Test conversion to pandas DataFrame."""
        probabilities = {
            "A": {"True": 0.6, "False": 0.4},
            "B": {"True": 0.7, "False": 0.3}
        }
        
        result = InferenceResult(
            query_variables=["A", "B"],
            evidence={},
            probabilities=probabilities,
            inference_time=0.1,
            method="VariableElimination"
        )
        
        df = result.to_dataframe()
        
        assert len(df) == 4  # 2 variables * 2 states each
        assert "variable" in df.columns
        assert "state" in df.columns
        assert "probability" in df.columns
        
        # Check specific values
        a_true = df[(df["variable"] == "A") & (df["state"] == "True")]["probability"].iloc[0]
        assert a_true == 0.6
    
    def test_get_marginal(self):
        """Test getting marginal distribution for a variable."""
        probabilities = {
            "A": {"True": 0.6, "False": 0.4},
            "B": {"True": 0.7, "False": 0.3}
        }
        
        result = InferenceResult(
            query_variables=["A", "B"],
            evidence={},
            probabilities=probabilities,
            inference_time=0.1,
            method="VariableElimination"
        )
        
        marginal_a = result.get_marginal("A")
        assert marginal_a == {"True": 0.6, "False": 0.4}
        
        marginal_b = result.get_marginal("B")
        assert marginal_b == {"True": 0.7, "False": 0.3}
        
        # Test non-existent variable
        marginal_c = result.get_marginal("C")
        assert marginal_c == {}
    
    def test_get_max_probability_state(self):
        """Test getting most probable state for a variable."""
        probabilities = {
            "A": {"True": 0.6, "False": 0.4},
            "B": {"True": 0.3, "False": 0.7}
        }
        
        result = InferenceResult(
            query_variables=["A", "B"],
            evidence={},
            probabilities=probabilities,
            inference_time=0.1,
            method="VariableElimination"
        )
        
        max_state_a, max_prob_a = result.get_max_probability_state("A")
        assert max_state_a == "True"
        assert max_prob_a == 0.6
        
        max_state_b, max_prob_b = result.get_max_probability_state("B")
        assert max_state_b == "False"
        assert max_prob_b == 0.7
        
        # Test non-existent variable
        max_state_c, max_prob_c = result.get_max_probability_state("C")
        assert max_state_c is None
        assert max_prob_c == 0.0


class TestInferenceEngine:
    """Test cases for InferenceEngine class."""
    
    def setup_method(self):
        """Setup method to create a simple test network."""
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
    
    def test_inference_engine_creation(self):
        """Test InferenceEngine creation."""
        # This test will fail if pgmpy is not available
        try:
            engine = InferenceEngine(self.network)
            assert engine.network == self.network
            assert engine.pgmpy_model is not None
        except ImportError:
            pytest.skip("pgmpy not available")
    
    def test_validate_inference_setup(self):
        """Test inference setup validation."""
        try:
            engine = InferenceEngine(self.network)
            is_valid, errors = engine.validate_inference_setup()
            assert is_valid is True
            assert len(errors) == 0
        except ImportError:
            pytest.skip("pgmpy not available")
    
    def test_validate_inference_setup_no_pgmpy(self):
        """Test inference setup validation without pgmpy."""
        # Mock the case where pgmpy is not available
        import inference_engine
        original_pgmpy_available = inference_engine.PGMPY_AVAILABLE
        inference_engine.PGMPY_AVAILABLE = False
        
        try:
            engine = InferenceEngine.__new__(InferenceEngine)
            engine.network = self.network
            engine.pgmpy_model = None
            
            is_valid, errors = engine.validate_inference_setup()
            assert is_valid is False
            assert "pgmpy is not available" in errors[0]
        finally:
            inference_engine.PGMPY_AVAILABLE = original_pgmpy_available
    
    def test_query_validation(self):
        """Test query input validation."""
        try:
            engine = InferenceEngine(self.network)
            
            # Test empty query variables
            with pytest.raises(ValueError, match="at least one variable"):
                engine.query([], {})
            
            # Test non-existent variable
            with pytest.raises(ValueError, match="not found in network"):
                engine.query(["NonExistent"], {})
            
            # Test invalid evidence
            with pytest.raises(ValueError, match="not found in network"):
                engine.query(["A"], {"NonExistent": "True"})
            
            # Test invalid state
            with pytest.raises(ValueError, match="not valid for variable"):
                engine.query(["A"], {"A": "InvalidState"})
            
            # Test unsupported method
            with pytest.raises(ValueError, match="Unsupported inference method"):
                engine.query(["A"], {}, method="UnsupportedMethod")
                
        except ImportError:
            pytest.skip("pgmpy not available")
    
    def test_variable_elimination_query(self):
        """Test variable elimination inference."""
        try:
            engine = InferenceEngine(self.network)
            
            # Query single variable
            result = engine.query(["A"])
            
            assert result.query_variables == ["A"]
            assert result.evidence == {}
            assert result.method == "VariableElimination"
            assert "A" in result.probabilities
            assert len(result.probabilities["A"]) == 2
            assert abs(sum(result.probabilities["A"].values()) - 1.0) < 1e-6
            
        except ImportError:
            pytest.skip("pgmpy not available")
    
    def test_query_with_evidence(self):
        """Test query with evidence."""
        try:
            engine = InferenceEngine(self.network)
            
            # Query with evidence
            result = engine.query(["C"], {"A": "True"})
            
            assert result.query_variables == ["C"]
            assert result.evidence == {"A": "True"}
            assert result.method == "VariableElimination"
            assert "C" in result.probabilities
            assert len(result.probabilities["C"]) == 2
            assert abs(sum(result.probabilities["C"].values()) - 1.0) < 1e-6
            
        except ImportError:
            pytest.skip("pgmpy not available")
    
    def test_multiple_variable_query(self):
        """Test querying multiple variables."""
        try:
            engine = InferenceEngine(self.network)
            
            # Query multiple variables
            result = engine.query(["A", "B", "C"])
            
            assert set(result.query_variables) == {"A", "B", "C"}
            assert result.evidence == {}
            assert result.method == "VariableElimination"
            
            for var in ["A", "B", "C"]:
                assert var in result.probabilities
                assert len(result.probabilities[var]) == 2
                assert abs(sum(result.probabilities[var].values()) - 1.0) < 1e-6
            
        except ImportError:
            pytest.skip("pgmpy not available")
    
    def test_get_marginal_probabilities(self):
        """Test getting marginal probabilities."""
        try:
            engine = InferenceEngine(self.network)
            
            # Get marginals for all variables
            marginals = engine.get_marginal_probabilities()
            
            assert "A" in marginals
            assert "B" in marginals
            assert "C" in marginals
            
            for var, probs in marginals.items():
                assert len(probs) == 2
                assert abs(sum(probs.values()) - 1.0) < 1e-6
            
            # Get marginals for specific variables
            specific_marginals = engine.get_marginal_probabilities(["A", "B"])
            assert set(specific_marginals.keys()) == {"A", "B"}
            
        except ImportError:
            pytest.skip("pgmpy not available")
    
    def test_get_conditional_probability(self):
        """Test getting conditional probability."""
        try:
            engine = InferenceEngine(self.network)
            
            # Test conditional probability
            prob = engine.get_conditional_probability("C", "True", {"A": "True"})
            
            assert 0.0 <= prob <= 1.0
            
        except ImportError:
            pytest.skip("pgmpy not available")
    
    def test_sensitivity_analysis(self):
        """Test sensitivity analysis."""
        try:
            engine = InferenceEngine(self.network)
            
            # Perform sensitivity analysis
            results = engine.sensitivity_analysis(
                target_var="C",
                target_state="True",
                parameter_var="A",
                parameter_states=["True", "False"]
            )
            
            assert "True" in results
            assert "False" in results
            
            for prob in results.values():
                assert 0.0 <= prob <= 1.0
            
        except ImportError:
            pytest.skip("pgmpy not available")
    
    def test_most_probable_explanation(self):
        """Test most probable explanation."""
        try:
            engine = InferenceEngine(self.network)
            
            # Find MPE
            evidence = {"C": "True"}
            mpe = engine.most_probable_explanation(evidence)
            
            assert "A" in mpe
            assert "B" in mpe
            assert mpe["A"] in ["True", "False"]
            assert mpe["B"] in ["True", "False"]
            
        except ImportError:
            pytest.skip("pgmpy not available")
    
    def test_get_network_statistics(self):
        """Test getting network statistics."""
        try:
            engine = InferenceEngine(self.network)
            
            stats = engine.get_network_statistics()
            
            assert "network_info" in stats
            assert "inference_methods" in stats
            assert "pgmpy_available" in stats
            assert "max_query_variables" in stats
            assert "supported_evidence_types" in stats
            
            assert stats["pgmpy_available"] is True
            assert stats["max_query_variables"] == 3
            assert "VariableElimination" in stats["inference_methods"]
            
        except ImportError:
            pytest.skip("pgmpy not available")


class TestInferenceMethods:
    """Test different inference methods."""
    
    def setup_method(self):
        """Setup method to create a test network."""
        self.network = BayesianNetwork("Test Network")
        
        # Create a simple network: A -> B
        node_a = Node("A", ["True", "False"])
        node_b = Node("B", ["True", "False"])
        
        self.network.add_node(node_a)
        self.network.add_node(node_b)
        self.network.add_edge("A", "B")
        
        # Create CPTs
        cpt_a = CPT("A", [], {"True": 0.6, "False": 0.4})
        cpt_b = CPT("B", ["A"], {
            "True|True": 0.8, "False|True": 0.2,
            "True|False": 0.3, "False|False": 0.7
        })
        
        self.network.set_cpt(cpt_a)
        self.network.set_cpt(cpt_b)
    
    def test_belief_propagation(self):
        """Test belief propagation inference."""
        try:
            engine = InferenceEngine(self.network)
            
            result = engine.query(["B"], method="BeliefPropagation")
            
            assert result.method == "BeliefPropagation"
            assert "B" in result.probabilities
            assert len(result.probabilities["B"]) == 2
            assert abs(sum(result.probabilities["B"].values()) - 1.0) < 1e-6
            
        except ImportError:
            pytest.skip("pgmpy not available")
    
    def test_junction_tree(self):
        """Test junction tree inference."""
        try:
            engine = InferenceEngine(self.network)
            
            result = engine.query(["B"], method="JunctionTree")
            
            assert result.method == "JunctionTree"
            assert "B" in result.probabilities
            assert len(result.probabilities["B"]) == 2
            assert abs(sum(result.probabilities["B"].values()) - 1.0) < 1e-6
            
        except ImportError:
            pytest.skip("pgmpy not available")
    
    def test_gibbs_sampling(self):
        """Test Gibbs sampling inference."""
        try:
            engine = InferenceEngine(self.network)
            
            result = engine.query(["B"], method="GibbsSampling", samples=1000)
            
            assert result.method == "GibbsSampling"
            assert result.samples == 1000
            assert "B" in result.probabilities
            assert len(result.probabilities["B"]) == 2
            assert abs(sum(result.probabilities["B"].values()) - 1.0) < 1e-6
            
        except ImportError:
            pytest.skip("pgmpy not available")


if __name__ == "__main__":
    pytest.main([__file__]) 