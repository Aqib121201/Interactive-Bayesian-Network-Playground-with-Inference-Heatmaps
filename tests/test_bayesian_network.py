"""
Unit tests for the Bayesian Network module.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from bayesian_network import BayesianNetwork, Node, CPT


class TestNode:
    """Test cases for Node class."""
    
    def test_node_creation(self):
        """Test basic node creation."""
        node = Node("TestNode", ["True", "False"], "Test description")
        
        assert node.name == "TestNode"
        assert node.states == ["True", "False"]
        assert node.description == "Test description"
        assert node.position == (0.0, 0.0)
    
    def test_node_default_states(self):
        """Test node creation with default states."""
        node = Node("TestNode")
        
        assert node.name == "TestNode"
        assert node.states == ["True", "False"]
        assert node.description == ""
    
    def test_node_to_dict(self):
        """Test node serialization to dictionary."""
        node = Node("TestNode", ["A", "B", "C"], "Test", (1.0, 2.0))
        node_dict = node.to_dict()
        
        expected = {
            "name": "TestNode",
            "states": ["A", "B", "C"],
            "description": "Test",
            "position": (1.0, 2.0)
        }
        
        assert node_dict == expected
    
    def test_node_from_dict(self):
        """Test node creation from dictionary."""
        data = {
            "name": "TestNode",
            "states": ["A", "B", "C"],
            "description": "Test",
            "position": (1.0, 2.0)
        }
        
        node = Node.from_dict(data)
        
        assert node.name == "TestNode"
        assert node.states == ["A", "B", "C"]
        assert node.description == "Test"
        assert node.position == (1.0, 2.0)


class TestCPT:
    """Test cases for CPT class."""
    
    def test_cpt_creation(self):
        """Test basic CPT creation."""
        probabilities = {"True": 0.6, "False": 0.4}
        cpt = CPT("TestNode", [], probabilities)
        
        assert cpt.node_name == "TestNode"
        assert cpt.parent_names == []
        assert cpt.probabilities == probabilities
    
    def test_cpt_validation_root_node(self):
        """Test CPT validation for root node."""
        # Valid CPT
        valid_cpt = CPT("TestNode", [], {"True": 0.6, "False": 0.4})
        assert valid_cpt.validate() is True
        
        # Invalid CPT (probabilities don't sum to 1.0)
        invalid_cpt = CPT("TestNode", [], {"True": 0.6, "False": 0.3})
        assert invalid_cpt.validate() is False
    
    def test_cpt_validation_conditional(self):
        """Test CPT validation for conditional node."""
        # Valid conditional CPT
        valid_cpt = CPT("Child", ["Parent"], {
            "True|True": 0.8, "False|True": 0.2,
            "True|False": 0.3, "False|False": 0.7
        })
        assert valid_cpt.validate() is True
        
        # Invalid conditional CPT
        invalid_cpt = CPT("Child", ["Parent"], {
            "True|True": 0.8, "False|True": 0.1,  # Doesn't sum to 1.0
            "True|False": 0.3, "False|False": 0.7
        })
        assert invalid_cpt.validate() is False
    
    def test_cpt_to_dict(self):
        """Test CPT serialization to dictionary."""
        probabilities = {"True": 0.6, "False": 0.4}
        cpt = CPT("TestNode", ["Parent"], probabilities)
        cpt_dict = cpt.to_dict()
        
        expected = {
            "node_name": "TestNode",
            "parent_names": ["Parent"],
            "probabilities": probabilities
        }
        
        assert cpt_dict == expected
    
    def test_cpt_from_dict(self):
        """Test CPT creation from dictionary."""
        data = {
            "node_name": "TestNode",
            "parent_names": ["Parent"],
            "probabilities": {"True": 0.6, "False": 0.4}
        }
        
        cpt = CPT.from_dict(data)
        
        assert cpt.node_name == "TestNode"
        assert cpt.parent_names == ["Parent"]
        assert cpt.probabilities == {"True": 0.6, "False": 0.4}


class TestBayesianNetwork:
    """Test cases for BayesianNetwork class."""
    
    def test_network_creation(self):
        """Test basic network creation."""
        network = BayesianNetwork("Test Network")
        
        assert network.name == "Test Network"
        assert len(network.nodes) == 0
        assert len(network.edges) == 0
        assert len(network.cpts) == 0
    
    def test_add_node(self):
        """Test adding nodes to network."""
        network = BayesianNetwork("Test Network")
        node = Node("TestNode", ["True", "False"])
        
        # Add node
        result = network.add_node(node)
        assert result is True
        assert "TestNode" in network.nodes
        assert network.nodes["TestNode"] == node
    
    def test_add_duplicate_node(self):
        """Test adding duplicate node raises error."""
        network = BayesianNetwork("Test Network")
        node1 = Node("TestNode", ["True", "False"])
        node2 = Node("TestNode", ["A", "B"])
        
        network.add_node(node1)
        
        with pytest.raises(ValueError, match="already exists"):
            network.add_node(node2)
    
    def test_remove_node(self):
        """Test removing nodes from network."""
        network = BayesianNetwork("Test Network")
        node = Node("TestNode", ["True", "False"])
        
        network.add_node(node)
        assert "TestNode" in network.nodes
        
        # Remove node
        result = network.remove_node("TestNode")
        assert result is True
        assert "TestNode" not in network.nodes
    
    def test_remove_nonexistent_node(self):
        """Test removing non-existent node returns False."""
        network = BayesianNetwork("Test Network")
        
        result = network.remove_node("Nonexistent")
        assert result is False
    
    def test_add_edge(self):
        """Test adding edges to network."""
        network = BayesianNetwork("Test Network")
        
        # Add nodes first
        node1 = Node("Node1", ["True", "False"])
        node2 = Node("Node2", ["True", "False"])
        network.add_node(node1)
        network.add_node(node2)
        
        # Add edge
        result = network.add_edge("Node1", "Node2")
        assert result is True
        assert ("Node1", "Node2") in network.edges
    
    def test_add_edge_nonexistent_nodes(self):
        """Test adding edge with non-existent nodes raises error."""
        network = BayesianNetwork("Test Network")
        
        with pytest.raises(ValueError, match="must exist"):
            network.add_edge("Node1", "Node2")
    
    def test_add_self_loop(self):
        """Test adding self-loop raises error."""
        network = BayesianNetwork("Test Network")
        node = Node("Node1", ["True", "False"])
        network.add_node(node)
        
        with pytest.raises(ValueError, match="Self-loops"):
            network.add_edge("Node1", "Node1")
    
    def test_add_edge_creates_cycle(self):
        """Test adding edge that creates cycle raises error."""
        network = BayesianNetwork("Test Network")
        
        # Create A -> B -> C
        for i in range(3):
            network.add_node(Node(f"Node{i}", ["True", "False"]))
        
        network.add_edge("Node0", "Node1")
        network.add_edge("Node1", "Node2")
        
        # Adding C -> A would create a cycle
        with pytest.raises(ValueError, match="create a cycle"):
            network.add_edge("Node2", "Node0")
    
    def test_remove_edge(self):
        """Test removing edges from network."""
        network = BayesianNetwork("Test Network")
        
        # Add nodes and edge
        node1 = Node("Node1", ["True", "False"])
        node2 = Node("Node2", ["True", "False"])
        network.add_node(node1)
        network.add_node(node2)
        network.add_edge("Node1", "Node2")
        
        # Remove edge
        result = network.remove_edge("Node1", "Node2")
        assert result is True
        assert ("Node1", "Node2") not in network.edges
    
    def test_remove_nonexistent_edge(self):
        """Test removing non-existent edge returns False."""
        network = BayesianNetwork("Test Network")
        
        result = network.remove_edge("Node1", "Node2")
        assert result is False
    
    def test_get_parents(self):
        """Test getting parent nodes."""
        network = BayesianNetwork("Test Network")
        
        # Create A -> B -> C
        for i in range(3):
            network.add_node(Node(f"Node{i}", ["True", "False"]))
        
        network.add_edge("Node0", "Node1")
        network.add_edge("Node1", "Node2")
        
        assert network.get_parents("Node0") == []
        assert network.get_parents("Node1") == ["Node0"]
        assert network.get_parents("Node2") == ["Node1"]
    
    def test_get_children(self):
        """Test getting child nodes."""
        network = BayesianNetwork("Test Network")
        
        # Create A -> B -> C
        for i in range(3):
            network.add_node(Node(f"Node{i}", ["True", "False"]))
        
        network.add_edge("Node0", "Node1")
        network.add_edge("Node1", "Node2")
        
        assert network.get_children("Node0") == ["Node1"]
        assert network.get_children("Node1") == ["Node2"]
        assert network.get_children("Node2") == []
    
    def test_is_acyclic(self):
        """Test cycle detection."""
        network = BayesianNetwork("Test Network")
        
        # Create A -> B -> C (acyclic)
        for i in range(3):
            network.add_node(Node(f"Node{i}", ["True", "False"]))
        
        network.add_edge("Node0", "Node1")
        network.add_edge("Node1", "Node2")
        
        assert network.is_acyclic() is True
        
        # Create A -> B -> C -> A (cyclic)
        network.add_edge("Node2", "Node0")
        assert network.is_acyclic() is False
    
    def test_get_topological_order(self):
        """Test topological ordering."""
        network = BayesianNetwork("Test Network")
        
        # Create A -> B -> C
        for i in range(3):
            network.add_node(Node(f"Node{i}", ["True", "False"]))
        
        network.add_edge("Node0", "Node1")
        network.add_edge("Node1", "Node2")
        
        order = network.get_topological_order()
        assert order == ["Node0", "Node1", "Node2"]
    
    def test_get_topological_order_cyclic(self):
        """Test topological ordering with cycle raises error."""
        network = BayesianNetwork("Test Network")
        
        # Create A -> B -> C -> A (cyclic)
        for i in range(3):
            network.add_node(Node(f"Node{i}", ["True", "False"]))
        
        network.add_edge("Node0", "Node1")
        network.add_edge("Node1", "Node2")
        network.add_edge("Node2", "Node0")
        
        with pytest.raises(ValueError, match="contains cycles"):
            network.get_topological_order()
    
    def test_set_cpt(self):
        """Test setting CPT for node."""
        network = BayesianNetwork("Test Network")
        node = Node("TestNode", ["True", "False"])
        network.add_node(node)
        
        cpt = CPT("TestNode", [], {"True": 0.6, "False": 0.4})
        result = network.set_cpt(cpt)
        
        assert result is True
        assert "TestNode" in network.cpts
        assert network.cpts["TestNode"] == cpt
    
    def test_set_cpt_nonexistent_node(self):
        """Test setting CPT for non-existent node raises error."""
        network = BayesianNetwork("Test Network")
        cpt = CPT("TestNode", [], {"True": 0.6, "False": 0.4})
        
        with pytest.raises(ValueError, match="does not exist"):
            network.set_cpt(cpt)
    
    def test_set_invalid_cpt(self):
        """Test setting invalid CPT raises error."""
        network = BayesianNetwork("Test Network")
        node = Node("TestNode", ["True", "False"])
        network.add_node(node)
        
        # Invalid CPT (probabilities don't sum to 1.0)
        invalid_cpt = CPT("TestNode", [], {"True": 0.6, "False": 0.3})
        
        with pytest.raises(ValueError, match="must sum to 1.0"):
            network.set_cpt(invalid_cpt)
    
    def test_validate_network(self):
        """Test network validation."""
        network = BayesianNetwork("Test Network")
        
        # Add nodes and edges
        node1 = Node("Node1", ["True", "False"])
        node2 = Node("Node2", ["True", "False"])
        network.add_node(node1)
        network.add_node(node2)
        network.add_edge("Node1", "Node2")
        
        # Add CPTs
        cpt1 = CPT("Node1", [], {"True": 0.6, "False": 0.4})
        cpt2 = CPT("Node2", ["Node1"], {
            "True|True": 0.8, "False|True": 0.2,
            "True|False": 0.3, "False|False": 0.7
        })
        network.set_cpt(cpt1)
        network.set_cpt(cpt2)
        
        is_valid, errors = network.validate_network()
        assert is_valid is True
        assert len(errors) == 0
    
    def test_validate_network_missing_cpt(self):
        """Test network validation with missing CPT."""
        network = BayesianNetwork("Test Network")
        node = Node("TestNode", ["True", "False"])
        network.add_node(node)
        
        is_valid, errors = network.validate_network()
        assert is_valid is False
        assert any("missing CPT" in error for error in errors)
    
    def test_validate_network_cyclic(self):
        """Test network validation with cycle."""
        network = BayesianNetwork("Test Network")
        
        # Create A -> B -> C -> A (cyclic)
        for i in range(3):
            network.add_node(Node(f"Node{i}", ["True", "False"]))
        
        network.add_edge("Node0", "Node1")
        network.add_edge("Node1", "Node2")
        network.add_edge("Node2", "Node0")
        
        is_valid, errors = network.validate_network()
        assert is_valid is False
        assert any("contains cycles" in error for error in errors)
    
    def test_get_network_info(self):
        """Test getting network information."""
        network = BayesianNetwork("Test Network")
        
        # Add nodes and edges
        node1 = Node("Node1", ["True", "False"])
        node2 = Node("Node2", ["True", "False"])
        network.add_node(node1)
        network.add_node(node2)
        network.add_edge("Node1", "Node2")
        
        # Add CPTs
        cpt1 = CPT("Node1", [], {"True": 0.6, "False": 0.4})
        cpt2 = CPT("Node2", ["Node1"], {
            "True|True": 0.8, "False|True": 0.2,
            "True|False": 0.3, "False|False": 0.7
        })
        network.set_cpt(cpt1)
        network.set_cpt(cpt2)
        
        info = network.get_network_info()
        
        assert info["name"] == "Test Network"
        assert info["node_count"] == 2
        assert info["edge_count"] == 1
        assert info["is_acyclic"] is True
        assert info["is_valid"] is True
        assert len(info["validation_errors"]) == 0
        assert info["topological_order"] == ["Node1", "Node2"]
    
    def test_save_load_network_json(self):
        """Test saving and loading network in JSON format."""
        network = BayesianNetwork("Test Network")
        
        # Add nodes and edges
        node1 = Node("Node1", ["True", "False"])
        node2 = Node("Node2", ["True", "False"])
        network.add_node(node1)
        network.add_node(node2)
        network.add_edge("Node1", "Node2")
        
        # Add CPTs
        cpt1 = CPT("Node1", [], {"True": 0.6, "False": 0.4})
        cpt2 = CPT("Node2", ["Node1"], {
            "True|True": 0.8, "False|True": 0.2,
            "True|False": 0.3, "False|False": 0.7
        })
        network.set_cpt(cpt1)
        network.set_cpt(cpt2)
        
        # Save network
        save_result = network.save_network("test_network.json")
        assert save_result is True
        
        # Load network
        loaded_network = BayesianNetwork.load_network("test_network.json")
        
        # Verify loaded network
        assert loaded_network.name == network.name
        assert len(loaded_network.nodes) == len(network.nodes)
        assert len(loaded_network.edges) == len(network.edges)
        assert len(loaded_network.cpts) == len(network.cpts)
        
        # Clean up
        import os
        os.remove("test_network.json")
    
    def test_create_random_cpt(self):
        """Test creating random CPT for node."""
        network = BayesianNetwork("Test Network")
        node = Node("TestNode", ["True", "False"])
        network.add_node(node)
        
        cpt = network.create_random_cpt("TestNode")
        
        assert cpt.node_name == "TestNode"
        assert cpt.parent_names == []
        assert cpt.validate() is True
    
    def test_create_random_cpt_with_parents(self):
        """Test creating random CPT for node with parents."""
        network = BayesianNetwork("Test Network")
        
        # Create parent and child nodes
        parent = Node("Parent", ["A", "B"])
        child = Node("Child", ["True", "False"])
        network.add_node(parent)
        network.add_node(child)
        network.add_edge("Parent", "Child")
        
        cpt = network.create_random_cpt("Child")
        
        assert cpt.node_name == "Child"
        assert cpt.parent_names == ["Parent"]
        assert cpt.validate() is True
    
    def test_str_repr(self):
        """Test string representation of network."""
        network = BayesianNetwork("Test Network")
        node = Node("TestNode", ["True", "False"])
        network.add_node(node)
        
        str_repr = str(network)
        assert "Test Network" in str_repr
        assert "nodes=1" in str_repr
        assert "edges=0" in str_repr


if __name__ == "__main__":
    pytest.main([__file__]) 