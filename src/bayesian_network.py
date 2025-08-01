"""
Bayesian Network implementation with validation and management capabilities.
"""

import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import json
import pickle
from pathlib import Path

from .config import MAX_NODES, MAX_STATES_PER_NODE, DEFAULT_NODE_STATES


@dataclass
class Node:
    """Represents a node in the Bayesian Network."""
    name: str
    states: List[str] = field(default_factory=lambda: DEFAULT_NODE_STATES.copy())
    description: str = ""
    position: Tuple[float, float] = (0.0, 0.0)
    
    def __post_init__(self):
        if not self.states:
            self.states = DEFAULT_NODE_STATES.copy()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary representation."""
        return {
            "name": self.name,
            "states": self.states,
            "description": self.description,
            "position": self.position
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Node':
        """Create node from dictionary representation."""
        return cls(
            name=data["name"],
            states=data.get("states", DEFAULT_NODE_STATES.copy()),
            description=data.get("description", ""),
            position=data.get("position", (0.0, 0.0))
        )


@dataclass
class CPT:
    """Conditional Probability Table for a node."""
    node_name: str
    parent_names: List[str]
    probabilities: Dict[str, float]
    
    def validate(self) -> bool:
        """Validate that probabilities sum to 1.0 for each parent configuration."""
        if not self.probabilities:
            return False
        
        # Group by parent configurations
        parent_configs = {}
        for key, prob in self.probabilities.items():
            parent_config = key.rsplit('|', 1)[1] if '|' in key else 'root'
            if parent_config not in parent_configs:
                parent_configs[parent_config] = []
            parent_configs[parent_config].append(prob)
        
        # Check each configuration sums to 1.0
        for config, probs in parent_configs.items():
            if abs(sum(probs) - 1.0) > 1e-6:
                return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert CPT to dictionary representation."""
        return {
            "node_name": self.node_name,
            "parent_names": self.parent_names,
            "probabilities": self.probabilities
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CPT':
        """Create CPT from dictionary representation."""
        return cls(
            node_name=data["node_name"],
            parent_names=data["parent_names"],
            probabilities=data["probabilities"]
        )


class BayesianNetwork:
    """
    Bayesian Network implementation with comprehensive functionality.
    """
    
    def __init__(self, name: str = "Bayesian Network"):
        self.name = name
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Tuple[str, str]] = []
        self.cpts: Dict[str, CPT] = {}
        self.graph = nx.DiGraph()
        
    def add_node(self, node: Node) -> bool:
        """Add a node to the network."""
        if len(self.nodes) >= MAX_NODES:
            raise ValueError(f"Maximum number of nodes ({MAX_NODES}) exceeded")
        
        if len(node.states) > MAX_STATES_PER_NODE:
            raise ValueError(f"Maximum states per node ({MAX_STATES_PER_NODE}) exceeded")
        
        if node.name in self.nodes:
            raise ValueError(f"Node '{node.name}' already exists")
        
        self.nodes[node.name] = node
        self.graph.add_node(node.name, **node.to_dict())
        return True
    
    def remove_node(self, node_name: str) -> bool:
        """Remove a node from the network."""
        if node_name not in self.nodes:
            return False
        
        # Remove associated CPT
        if node_name in self.cpts:
            del self.cpts[node_name]
        
        # Remove edges
        edges_to_remove = []
        for edge in self.edges:
            if edge[0] == node_name or edge[1] == node_name:
                edges_to_remove.append(edge)
        
        for edge in edges_to_remove:
            self.edges.remove(edge)
        
        # Remove from graph
        self.graph.remove_node(node_name)
        del self.nodes[node_name]
        
        return True
    
    def add_edge(self, from_node: str, to_node: str) -> bool:
        """Add a directed edge between nodes."""
        if from_node not in self.nodes or to_node not in self.nodes:
            raise ValueError("Both nodes must exist in the network")
        
        if from_node == to_node:
            raise ValueError("Self-loops are not allowed")
        
        # Check for cycles
        temp_graph = self.graph.copy()
        temp_graph.add_edge(from_node, to_node)
        
        if not nx.is_directed_acyclic_graph(temp_graph):
            raise ValueError("Adding this edge would create a cycle")
        
        self.edges.append((from_node, to_node))
        self.graph.add_edge(from_node, to_node)
        return True
    
    def remove_edge(self, from_node: str, to_node: str) -> bool:
        """Remove a directed edge between nodes."""
        edge = (from_node, to_node)
        if edge in self.edges:
            self.edges.remove(edge)
            self.graph.remove_edge(from_node, to_node)
            return True
        return False
    
    def set_cpt(self, cpt: CPT) -> bool:
        """Set the CPT for a node."""
        if cpt.node_name not in self.nodes:
            raise ValueError(f"Node '{cpt.node_name}' does not exist")
        
        if not cpt.validate():
            raise ValueError("Invalid CPT: probabilities must sum to 1.0 for each parent configuration")
        
        self.cpts[cpt.node_name] = cpt
        return True
    
    def get_parents(self, node_name: str) -> List[str]:
        """Get parent nodes of a given node."""
        return list(self.graph.predecessors(node_name))
    
    def get_children(self, node_name: str) -> List[str]:
        """Get child nodes of a given node."""
        return list(self.graph.successors(node_name))
    
    def is_acyclic(self) -> bool:
        """Check if the network is acyclic."""
        return nx.is_directed_acyclic_graph(self.graph)
    
    def get_topological_order(self) -> List[str]:
        """Get topological ordering of nodes."""
        if not self.is_acyclic():
            raise ValueError("Network contains cycles")
        return list(nx.topological_sort(self.graph))
    
    def validate_network(self) -> Tuple[bool, List[str]]:
        """Validate the entire network structure and CPTs."""
        errors = []
        
        # Check for cycles
        if not self.is_acyclic():
            errors.append("Network contains cycles")
        
        # Check all nodes have CPTs
        for node_name in self.nodes:
            if node_name not in self.cpts:
                errors.append(f"Node '{node_name}' missing CPT")
        
        # Validate all CPTs
        for node_name, cpt in self.cpts.items():
            if not cpt.validate():
                errors.append(f"Invalid CPT for node '{node_name}'")
        
        # Check CPT parent consistency
        for node_name, cpt in self.cpts.items():
            actual_parents = self.get_parents(node_name)
            if set(cpt.parent_names) != set(actual_parents):
                errors.append(f"CPT parent mismatch for node '{node_name}'")
        
        return len(errors) == 0, errors
    
    def get_network_info(self) -> Dict[str, Any]:
        """Get comprehensive network information."""
        is_valid, errors = self.validate_network()
        
        return {
            "name": self.name,
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "is_acyclic": self.is_acyclic(),
            "is_valid": is_valid,
            "validation_errors": errors,
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": self.edges,
            "topological_order": self.get_topological_order() if self.is_acyclic() else None
        }
    
    def save_network(self, filepath: str) -> bool:
        """Save network to file."""
        try:
            data = {
                "name": self.name,
                "nodes": {name: node.to_dict() for name, node in self.nodes.items()},
                "edges": self.edges,
                "cpts": {name: cpt.to_dict() for name, cpt in self.cpts.items()}
            }
            
            if filepath.endswith('.json'):
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2)
            elif filepath.endswith('.pkl'):
                with open(filepath, 'wb') as f:
                    pickle.dump(data, f)
            else:
                raise ValueError("Unsupported file format")
            
            return True
        except Exception as e:
            print(f"Error saving network: {e}")
            return False
    
    @classmethod
    def load_network(cls, filepath: str) -> 'BayesianNetwork':
        """Load network from file."""
        try:
            if filepath.endswith('.json'):
                with open(filepath, 'r') as f:
                    data = json.load(f)
            elif filepath.endswith('.pkl'):
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
            else:
                raise ValueError("Unsupported file format")
            
            network = cls(data["name"])
            
            # Load nodes
            for name, node_data in data["nodes"].items():
                node = Node.from_dict(node_data)
                network.add_node(node)
            
            # Load edges
            for from_node, to_node in data["edges"]:
                network.add_edge(from_node, to_node)
            
            # Load CPTs
            for name, cpt_data in data["cpts"].items():
                cpt = CPT.from_dict(cpt_data)
                network.set_cpt(cpt)
            
            return network
        except Exception as e:
            raise ValueError(f"Error loading network: {e}")
    
    def create_random_cpt(self, node_name: str) -> CPT:
        """Create a random valid CPT for a node."""
        if node_name not in self.nodes:
            raise ValueError(f"Node '{node_name}' does not exist")
        
        node = self.nodes[node_name]
        parents = self.get_parents(node_name)
        
        # Generate all parent configurations
        parent_states = []
        for parent in parents:
            parent_states.append(self.nodes[parent].states)
        
        if parent_states:
            import itertools
            parent_configs = list(itertools.product(*parent_states))
        else:
            parent_configs = [()]
        
        probabilities = {}
        
        for config in parent_configs:
            # Generate random probabilities that sum to 1.0
            probs = np.random.dirichlet(np.ones(len(node.states)))
            
            for i, state in enumerate(node.states):
                if config:
                    key = f"{state}|{','.join(config)}"
                else:
                    key = state
                probabilities[key] = float(probs[i])
        
        return CPT(node_name, parents, probabilities)
    
    def get_marginal_probability(self, node_name: str, state: str) -> float:
        """Get marginal probability of a node state (requires inference engine)."""
        # This is a placeholder - actual implementation requires inference
        return 0.5  # Default uniform distribution
    
    def __str__(self) -> str:
        """String representation of the network."""
        info = self.get_network_info()
        return f"BayesianNetwork(name='{self.name}', nodes={info['node_count']}, edges={info['edge_count']}, valid={info['is_valid']})"
    
    def __repr__(self) -> str:
        return self.__str__() 