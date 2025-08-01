"""
Pre-built example Bayesian Networks for educational and demonstration purposes.
"""

from typing import Dict, List, Tuple, Any
from .bayesian_network import BayesianNetwork, Node, CPT
from .config import EXAMPLE_NETWORKS


def create_medical_diagnosis_network() -> BayesianNetwork:
    """
    Create a medical diagnosis Bayesian Network.
    
    Network structure:
    - Diseases: Cancer, Heart Disease
    - Symptoms: Chest Pain, Shortness of Breath, Fatigue, Weight Loss
    - Tests: Blood Test, X-Ray
    - Risk Factors: Smoking, Age
    
    Returns:
        BayesianNetwork: Configured medical diagnosis network
    """
    network = BayesianNetwork("Medical Diagnosis Network")
    
    # Create nodes
    nodes = {
        "Smoking": Node("Smoking", ["Yes", "No"], "Smoking history"),
        "Age": Node("Age", ["Young", "Middle", "Old"], "Age group"),
        "Cancer": Node("Cancer", ["Present", "Absent"], "Cancer diagnosis"),
        "HeartDisease": Node("HeartDisease", ["Present", "Absent"], "Heart disease diagnosis"),
        "ChestPain": Node("ChestPain", ["Yes", "No"], "Chest pain symptom"),
        "ShortnessOfBreath": Node("ShortnessOfBreath", ["Yes", "No"], "Shortness of breath"),
        "Fatigue": Node("Fatigue", ["Yes", "No"], "Fatigue symptom"),
        "WeightLoss": Node("WeightLoss", ["Yes", "No"], "Weight loss"),
        "BloodTest": Node("BloodTest", ["Positive", "Negative"], "Blood test result"),
        "XRay": Node("XRay", ["Positive", "Negative"], "X-ray result")
    }
    
    # Add nodes to network
    for node in nodes.values():
        network.add_node(node)
    
    # Add edges
    edges = [
        ("Smoking", "Cancer"),
        ("Smoking", "HeartDisease"),
        ("Age", "Cancer"),
        ("Age", "HeartDisease"),
        ("Cancer", "WeightLoss"),
        ("Cancer", "Fatigue"),
        ("Cancer", "BloodTest"),
        ("HeartDisease", "ChestPain"),
        ("HeartDisease", "ShortnessOfBreath"),
        ("HeartDisease", "Fatigue"),
        ("Cancer", "XRay")
    ]
    
    for from_node, to_node in edges:
        network.add_edge(from_node, to_node)
    
    # Define CPTs
    cpts = {
        # Smoking (root node)
        "Smoking": CPT("Smoking", [], {
            "Yes": 0.3,
            "No": 0.7
        }),
        
        # Age (root node)
        "Age": CPT("Age", [], {
            "Young": 0.4,
            "Middle": 0.35,
            "Old": 0.25
        }),
        
        # Cancer (depends on Smoking, Age)
        "Cancer": CPT("Cancer", ["Smoking", "Age"], {
            "Present|Yes,Young": 0.05,
            "Absent|Yes,Young": 0.95,
            "Present|Yes,Middle": 0.15,
            "Absent|Yes,Middle": 0.85,
            "Present|Yes,Old": 0.25,
            "Absent|Yes,Old": 0.75,
            "Present|No,Young": 0.01,
            "Absent|No,Young": 0.99,
            "Present|No,Middle": 0.05,
            "Absent|No,Middle": 0.95,
            "Present|No,Old": 0.10,
            "Absent|No,Old": 0.90
        }),
        
        # Heart Disease (depends on Smoking, Age)
        "HeartDisease": CPT("HeartDisease", ["Smoking", "Age"], {
            "Present|Yes,Young": 0.10,
            "Absent|Yes,Young": 0.90,
            "Present|Yes,Middle": 0.20,
            "Absent|Yes,Middle": 0.80,
            "Present|Yes,Old": 0.35,
            "Absent|Yes,Old": 0.65,
            "Present|No,Young": 0.02,
            "Absent|No,Young": 0.98,
            "Present|No,Middle": 0.08,
            "Absent|No,Middle": 0.92,
            "Present|No,Old": 0.15,
            "Absent|No,Old": 0.85
        }),
        
        # Chest Pain (depends on Heart Disease)
        "ChestPain": CPT("ChestPain", ["HeartDisease"], {
            "Yes|Present": 0.80,
            "No|Present": 0.20,
            "Yes|Absent": 0.05,
            "No|Absent": 0.95
        }),
        
        # Shortness of Breath (depends on Heart Disease)
        "ShortnessOfBreath": CPT("ShortnessOfBreath", ["HeartDisease"], {
            "Yes|Present": 0.70,
            "No|Present": 0.30,
            "Yes|Absent": 0.03,
            "No|Absent": 0.97
        }),
        
        # Fatigue (depends on Cancer, Heart Disease)
        "Fatigue": CPT("Fatigue", ["Cancer", "HeartDisease"], {
            "Yes|Present,Present": 0.95,
            "No|Present,Present": 0.05,
            "Yes|Present,Absent": 0.85,
            "No|Present,Absent": 0.15,
            "Yes|Absent,Present": 0.75,
            "No|Absent,Present": 0.25,
            "Yes|Absent,Absent": 0.10,
            "No|Absent,Absent": 0.90
        }),
        
        # Weight Loss (depends on Cancer)
        "WeightLoss": CPT("WeightLoss", ["Cancer"], {
            "Yes|Present": 0.60,
            "No|Present": 0.40,
            "Yes|Absent": 0.05,
            "No|Absent": 0.95
        }),
        
        # Blood Test (depends on Cancer)
        "BloodTest": CPT("BloodTest", ["Cancer"], {
            "Positive|Present": 0.85,
            "Negative|Present": 0.15,
            "Positive|Absent": 0.10,
            "Negative|Absent": 0.90
        }),
        
        # X-Ray (depends on Cancer)
        "XRay": CPT("XRay", ["Cancer"], {
            "Positive|Present": 0.90,
            "Negative|Present": 0.10,
            "Positive|Absent": 0.05,
            "Negative|Absent": 0.95
        })
    }
    
    # Add CPTs to network
    for cpt in cpts.values():
        network.set_cpt(cpt)
    
    return network


def create_student_performance_network() -> BayesianNetwork:
    """
    Create a student performance Bayesian Network.
    
    Network structure:
    - Background: Intelligence, Motivation
    - Study Factors: Study Time, Study Method
    - Performance: Exam Score, Assignment Grade
    - External: Sleep Quality, Stress Level
    
    Returns:
        BayesianNetwork: Configured student performance network
    """
    network = BayesianNetwork("Student Performance Network")
    
    # Create nodes
    nodes = {
        "Intelligence": Node("Intelligence", ["High", "Medium", "Low"], "Student intelligence level"),
        "Motivation": Node("Motivation", ["High", "Medium", "Low"], "Student motivation level"),
        "StudyTime": Node("StudyTime", ["High", "Medium", "Low"], "Study time per week"),
        "StudyMethod": Node("StudyMethod", ["Effective", "Ineffective"], "Study method quality"),
        "SleepQuality": Node("SleepQuality", ["Good", "Poor"], "Sleep quality"),
        "StressLevel": Node("StressLevel", ["Low", "High"], "Stress level"),
        "ExamScore": Node("ExamScore", ["Excellent", "Good", "Average", "Poor"], "Exam performance"),
        "AssignmentGrade": Node("AssignmentGrade", ["A", "B", "C", "D"], "Assignment grade")
    }
    
    # Add nodes to network
    for node in nodes.values():
        network.add_node(node)
    
    # Add edges
    edges = [
        ("Intelligence", "ExamScore"),
        ("Intelligence", "AssignmentGrade"),
        ("Motivation", "StudyTime"),
        ("Motivation", "StudyMethod"),
        ("StudyTime", "ExamScore"),
        ("StudyTime", "AssignmentGrade"),
        ("StudyMethod", "ExamScore"),
        ("StudyMethod", "AssignmentGrade"),
        ("SleepQuality", "StressLevel"),
        ("StressLevel", "ExamScore"),
        ("StressLevel", "AssignmentGrade")
    ]
    
    for from_node, to_node in edges:
        network.add_edge(from_node, to_node)
    
    # Define CPTs
    cpts = {
        # Intelligence (root node)
        "Intelligence": CPT("Intelligence", [], {
            "High": 0.2,
            "Medium": 0.5,
            "Low": 0.3
        }),
        
        # Motivation (root node)
        "Motivation": CPT("Motivation", [], {
            "High": 0.3,
            "Medium": 0.4,
            "Low": 0.3
        }),
        
        # Study Time (depends on Motivation)
        "StudyTime": CPT("StudyTime", ["Motivation"], {
            "High|High": 0.8,
            "Medium|High": 0.15,
            "Low|High": 0.05,
            "High|Medium": 0.4,
            "Medium|Medium": 0.5,
            "Low|Medium": 0.1,
            "High|Low": 0.1,
            "Medium|Low": 0.3,
            "Low|Low": 0.6
        }),
        
        # Study Method (depends on Motivation)
        "StudyMethod": CPT("StudyMethod", ["Motivation"], {
            "Effective|High": 0.7,
            "Ineffective|High": 0.3,
            "Effective|Medium": 0.5,
            "Ineffective|Medium": 0.5,
            "Effective|Low": 0.2,
            "Ineffective|Low": 0.8
        }),
        
        # Sleep Quality (root node)
        "SleepQuality": CPT("SleepQuality", [], {
            "Good": 0.6,
            "Poor": 0.4
        }),
        
        # Stress Level (depends on Sleep Quality)
        "StressLevel": CPT("StressLevel", ["SleepQuality"], {
            "Low|Good": 0.8,
            "High|Good": 0.2,
            "Low|Poor": 0.3,
            "High|Poor": 0.7
        }),
        
        # Exam Score (depends on Intelligence, Study Time, Study Method, Stress Level)
        "ExamScore": CPT("ExamScore", ["Intelligence", "StudyTime", "StudyMethod", "StressLevel"], {
            "Excellent|High,High,Effective,Low": 0.95,
            "Good|High,High,Effective,Low": 0.05,
            "Average|High,High,Effective,Low": 0.0,
            "Poor|High,High,Effective,Low": 0.0,
            "Excellent|High,High,Effective,High": 0.7,
            "Good|High,High,Effective,High": 0.25,
            "Average|High,High,Effective,High": 0.05,
            "Poor|High,High,Effective,High": 0.0,
            # Simplified for brevity - would include all combinations
            "Excellent|Medium,Medium,Effective,Low": 0.4,
            "Good|Medium,Medium,Effective,Low": 0.5,
            "Average|Medium,Medium,Effective,Low": 0.1,
            "Poor|Medium,Medium,Effective,Low": 0.0,
            "Excellent|Low,Low,Ineffective,High": 0.0,
            "Good|Low,Low,Ineffective,High": 0.1,
            "Average|Low,Low,Ineffective,High": 0.4,
            "Poor|Low,Low,Ineffective,High": 0.5
        }),
        
        # Assignment Grade (similar structure to Exam Score)
        "AssignmentGrade": CPT("AssignmentGrade", ["Intelligence", "StudyTime", "StudyMethod", "StressLevel"], {
            "A|High,High,Effective,Low": 0.9,
            "B|High,High,Effective,Low": 0.1,
            "C|High,High,Effective,Low": 0.0,
            "D|High,High,Effective,Low": 0.0,
            "A|Medium,Medium,Effective,Low": 0.3,
            "B|Medium,Medium,Effective,Low": 0.6,
            "C|Medium,Medium,Effective,Low": 0.1,
            "D|Medium,Medium,Effective,Low": 0.0,
            "A|Low,Low,Ineffective,High": 0.0,
            "B|Low,Low,Ineffective,High": 0.1,
            "C|Low,Low,Ineffective,High": 0.5,
            "D|Low,Low,Ineffective,High": 0.4
        })
    }
    
    # Add CPTs to network
    for cpt in cpts.values():
        network.set_cpt(cpt)
    
    return network


def create_weather_prediction_network() -> BayesianNetwork:
    """
    Create a weather prediction Bayesian Network.
    
    Network structure:
    - Atmospheric: Pressure, Humidity, Temperature
    - Weather: Rain, Wind
    - Forecast: Tomorrow's Weather
    
    Returns:
        BayesianNetwork: Configured weather prediction network
    """
    network = BayesianNetwork("Weather Prediction Network")
    
    # Create nodes
    nodes = {
        "Pressure": Node("Pressure", ["High", "Low"], "Atmospheric pressure"),
        "Humidity": Node("Humidity", ["High", "Low"], "Humidity level"),
        "Temperature": Node("Temperature", ["Hot", "Mild", "Cold"], "Temperature"),
        "Rain": Node("Rain", ["Yes", "No"], "Rain occurrence"),
        "Wind": Node("Wind", ["Strong", "Weak"], "Wind strength"),
        "TomorrowWeather": Node("TomorrowWeather", ["Sunny", "Cloudy", "Rainy"], "Tomorrow's weather")
    }
    
    # Add nodes to network
    for node in nodes.values():
        network.add_node(node)
    
    # Add edges
    edges = [
        ("Pressure", "Rain"),
        ("Humidity", "Rain"),
        ("Temperature", "Rain"),
        ("Pressure", "Wind"),
        ("Rain", "TomorrowWeather"),
        ("Wind", "TomorrowWeather"),
        ("Temperature", "TomorrowWeather")
    ]
    
    for from_node, to_node in edges:
        network.add_edge(from_node, to_node)
    
    # Define CPTs
    cpts = {
        # Pressure (root node)
        "Pressure": CPT("Pressure", [], {
            "High": 0.6,
            "Low": 0.4
        }),
        
        # Humidity (root node)
        "Humidity": CPT("Humidity", [], {
            "High": 0.5,
            "Low": 0.5
        }),
        
        # Temperature (root node)
        "Temperature": CPT("Temperature", [], {
            "Hot": 0.3,
            "Mild": 0.5,
            "Cold": 0.2
        }),
        
        # Rain (depends on Pressure, Humidity, Temperature)
        "Rain": CPT("Rain", ["Pressure", "Humidity", "Temperature"], {
            "Yes|Low,High,Hot": 0.8,
            "No|Low,High,Hot": 0.2,
            "Yes|Low,High,Mild": 0.9,
            "No|Low,High,Mild": 0.1,
            "Yes|Low,High,Cold": 0.7,
            "No|Low,High,Cold": 0.3,
            "Yes|Low,Low,Hot": 0.3,
            "No|Low,Low,Hot": 0.7,
            "Yes|Low,Low,Mild": 0.5,
            "No|Low,Low,Mild": 0.5,
            "Yes|Low,Low,Cold": 0.4,
            "No|Low,Low,Cold": 0.6,
            "Yes|High,High,Hot": 0.4,
            "No|High,High,Hot": 0.6,
            "Yes|High,High,Mild": 0.6,
            "No|High,High,Mild": 0.4,
            "Yes|High,High,Cold": 0.3,
            "No|High,High,Cold": 0.7,
            "Yes|High,Low,Hot": 0.1,
            "No|High,Low,Hot": 0.9,
            "Yes|High,Low,Mild": 0.2,
            "No|High,Low,Mild": 0.8,
            "Yes|High,Low,Cold": 0.1,
            "No|High,Low,Cold": 0.9
        }),
        
        # Wind (depends on Pressure)
        "Wind": CPT("Wind", ["Pressure"], {
            "Strong|Low": 0.7,
            "Weak|Low": 0.3,
            "Strong|High": 0.2,
            "Weak|High": 0.8
        }),
        
        # Tomorrow's Weather (depends on Rain, Wind, Temperature)
        "TomorrowWeather": CPT("TomorrowWeather", ["Rain", "Wind", "Temperature"], {
            "Sunny|No,Weak,Hot": 0.8,
            "Cloudy|No,Weak,Hot": 0.15,
            "Rainy|No,Weak,Hot": 0.05,
            "Sunny|No,Weak,Mild": 0.6,
            "Cloudy|No,Weak,Mild": 0.3,
            "Rainy|No,Weak,Mild": 0.1,
            "Sunny|No,Weak,Cold": 0.4,
            "Cloudy|No,Weak,Cold": 0.4,
            "Rainy|No,Weak,Cold": 0.2,
            "Sunny|No,Strong,Hot": 0.5,
            "Cloudy|No,Strong,Hot": 0.3,
            "Rainy|No,Strong,Hot": 0.2,
            "Sunny|No,Strong,Mild": 0.3,
            "Cloudy|No,Strong,Mild": 0.4,
            "Rainy|No,Strong,Mild": 0.3,
            "Sunny|No,Strong,Cold": 0.2,
            "Cloudy|No,Strong,Cold": 0.3,
            "Rainy|No,Strong,Cold": 0.5,
            "Sunny|Yes,Weak,Hot": 0.1,
            "Cloudy|Yes,Weak,Hot": 0.3,
            "Rainy|Yes,Weak,Hot": 0.6,
            "Sunny|Yes,Weak,Mild": 0.05,
            "Cloudy|Yes,Weak,Mild": 0.2,
            "Rainy|Yes,Weak,Mild": 0.75,
            "Sunny|Yes,Weak,Cold": 0.0,
            "Cloudy|Yes,Weak,Cold": 0.1,
            "Rainy|Yes,Weak,Cold": 0.9,
            "Sunny|Yes,Strong,Hot": 0.0,
            "Cloudy|Yes,Strong,Hot": 0.1,
            "Rainy|Yes,Strong,Hot": 0.9,
            "Sunny|Yes,Strong,Mild": 0.0,
            "Cloudy|Yes,Strong,Mild": 0.05,
            "Rainy|Yes,Strong,Mild": 0.95,
            "Sunny|Yes,Strong,Cold": 0.0,
            "Cloudy|Yes,Strong,Cold": 0.0,
            "Rainy|Yes,Strong,Cold": 1.0
        })
    }
    
    # Add CPTs to network
    for cpt in cpts.values():
        network.set_cpt(cpt)
    
    return network


def create_simple_network() -> BayesianNetwork:
    """
    Create a simple 3-node Bayesian Network for basic demonstrations.
    
    Network structure:
    A -> B -> C
    
    Returns:
        BayesianNetwork: Simple 3-node network
    """
    network = BayesianNetwork("Simple Network")
    
    # Create nodes
    nodes = {
        "A": Node("A", ["True", "False"], "Root node A"),
        "B": Node("B", ["True", "False"], "Intermediate node B"),
        "C": Node("C", ["True", "False"], "Leaf node C")
    }
    
    # Add nodes to network
    for node in nodes.values():
        network.add_node(node)
    
    # Add edges
    edges = [("A", "B"), ("B", "C")]
    
    for from_node, to_node in edges:
        network.add_edge(from_node, to_node)
    
    # Define CPTs
    cpts = {
        # A (root node)
        "A": CPT("A", [], {
            "True": 0.6,
            "False": 0.4
        }),
        
        # B (depends on A)
        "B": CPT("B", ["A"], {
            "True|True": 0.8,
            "False|True": 0.2,
            "True|False": 0.3,
            "False|False": 0.7
        }),
        
        # C (depends on B)
        "C": CPT("C", ["B"], {
            "True|True": 0.7,
            "False|True": 0.3,
            "True|False": 0.2,
            "False|False": 0.8
        })
    }
    
    # Add CPTs to network
    for cpt in cpts.values():
        network.set_cpt(cpt)
    
    return network


def load_example_networks() -> Dict[str, BayesianNetwork]:
    """
    Load all example networks.
    
    Returns:
        Dictionary mapping network names to BayesianNetwork instances
    """
    networks = {
        "medical": create_medical_diagnosis_network(),
        "student": create_student_performance_network(),
        "weather": create_weather_prediction_network(),
        "simple": create_simple_network()
    }
    
    return networks


def get_network_info(network_name: str) -> Dict[str, Any]:
    """
    Get information about a specific example network.
    
    Args:
        network_name: Name of the network
        
    Returns:
        Dictionary with network information
    """
    if network_name not in EXAMPLE_NETWORKS:
        raise ValueError(f"Unknown network: {network_name}")
    
    return EXAMPLE_NETWORKS[network_name]


def list_available_networks() -> List[str]:
    """
    Get list of available example networks.
    
    Returns:
        List of network names
    """
    return list(EXAMPLE_NETWORKS.keys()) 