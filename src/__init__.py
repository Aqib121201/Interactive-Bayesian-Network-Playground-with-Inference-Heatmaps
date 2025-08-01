"""
Interactive Bayesian Network Playground with Inference Visualizations

A comprehensive toolkit for constructing, visualizing, and performing inference
on Bayesian Networks with advanced visualization capabilities.
"""

__version__ = "1.0.0"
__author__ = "Bayesian Network Research Team"
__email__ = "contact@bayesian-playground.org"

from .bayesian_network import BayesianNetwork
from .inference_engine import InferenceEngine
from .visualization import NetworkVisualizer
from .network_examples import load_example_networks

__all__ = [
    "BayesianNetwork",
    "InferenceEngine", 
    "NetworkVisualizer",
    "load_example_networks"
] 