"""
Configuration settings for the Bayesian Network Playground.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
VISUALIZATIONS_DIR = PROJECT_ROOT / "visualizations"
REPORT_DIR = PROJECT_ROOT / "report"

# Ensure directories exist
for directory in [DATA_DIR, MODELS_DIR, VISUALIZATIONS_DIR, REPORT_DIR]:
    directory.mkdir(exist_ok=True)

# Network configuration
MAX_NODES = 20
MAX_STATES_PER_NODE = 10
DEFAULT_NODE_STATES = ["True", "False"]

# Inference configuration
DEFAULT_SAMPLES = 10000
CONVERGENCE_THRESHOLD = 1e-6
MAX_ITERATIONS = 1000

# Visualization configuration
FIGURE_SIZE = (12, 8)
DPI = 300
COLOR_MAP = "viridis"
NODE_COLORS = {
    "default": "#1f77b4",
    "evidence": "#ff7f0e", 
    "query": "#2ca02c",
    "intermediate": "#d62728"
}

# Streamlit configuration
STREAMLIT_CONFIG = {
    "page_title": "Bayesian Network Playground",
    "page_icon": "🧠",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# pgmpy configuration
PGMPY_CONFIG = {
    "inference_methods": [
        "VariableElimination",
        "BeliefPropagation", 
        "JunctionTree",
        "GibbsSampling",
        "ParticleFiltering"
    ],
    "default_method": "VariableElimination"
}

# UI configuration
UI_CONFIG = {
    "sidebar_width": 300,
    "main_width": 800,
    "animation_speed": 1000,  # milliseconds
    "auto_save_interval": 30,  # seconds
}

# File extensions
SUPPORTED_FORMATS = [".pkl", ".json", ".xml", ".bif"]

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": PROJECT_ROOT / "logs" / "app.log"
}

# Performance thresholds
PERFORMANCE_THRESHOLDS = {
    "max_inference_time": 5000,  # milliseconds
    "max_memory_usage": 512,     # MB
    "max_network_size": 15,      # nodes
}

# Educational content
TUTORIAL_STEPS = [
    "Network Construction",
    "CPT Definition", 
    "Evidence Setting",
    "Inference Execution",
    "Result Interpretation"
]

# Example networks
EXAMPLE_NETWORKS = {
    "medical": {
        "name": "Medical Diagnosis",
        "description": "Network for diagnosing diseases based on symptoms",
        "nodes": 8,
        "complexity": "intermediate"
    },
    "student": {
        "name": "Student Performance", 
        "description": "Modeling academic performance factors",
        "nodes": 6,
        "complexity": "beginner"
    },
    "weather": {
        "name": "Weather Prediction",
        "description": "Weather forecasting based on atmospheric conditions", 
        "nodes": 5,
        "complexity": "beginner"
    }
} 