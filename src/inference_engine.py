"""
Inference engine for Bayesian Networks using pgmpy backend.
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging

try:
    from pgmpy.models import BayesianNetwork as PgmpyBayesianNetwork
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.inference import VariableElimination, BeliefPropagation, JunctionTree
    from pgmpy.sampling import GibbsSampling
    from pgmpy.readwrite import XMLBIFReader
    PGMPY_AVAILABLE = True
except ImportError:
    PGMPY_AVAILABLE = False
    logging.warning("pgmpy not available. Inference functionality will be limited.")

from .bayesian_network import BayesianNetwork, Node, CPT
from .config import DEFAULT_SAMPLES, CONVERGENCE_THRESHOLD, MAX_ITERATIONS


@dataclass
class InferenceResult:
    """Container for inference results."""
    query_variables: List[str]
    evidence: Dict[str, str]
    probabilities: Dict[str, Dict[str, float]]
    inference_time: float
    method: str
    samples: Optional[int] = None
    convergence_iterations: Optional[int] = None
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame."""
        data = []
        for var, probs in self.probabilities.items():
            for state, prob in probs.items():
                data.append({
                    'variable': var,
                    'state': state,
                    'probability': prob
                })
        return pd.DataFrame(data)
    
    def get_marginal(self, variable: str) -> Dict[str, float]:
        """Get marginal distribution for a specific variable."""
        return self.probabilities.get(variable, {})
    
    def get_max_probability_state(self, variable: str) -> Tuple[str, float]:
        """Get the most probable state for a variable."""
        if variable not in self.probabilities:
            return None, 0.0
        
        probs = self.probabilities[variable]
        max_state = max(probs, key=probs.get)
        return max_state, probs[max_state]


class InferenceEngine:
    """
    Inference engine supporting multiple inference algorithms.
    """
    
    def __init__(self, network: BayesianNetwork):
        self.network = network
        self.pgmpy_model = None
        self._build_pgmpy_model()
    
    def _build_pgmpy_model(self) -> None:
        """Build pgmpy Bayesian Network model from our network."""
        if not PGMPY_AVAILABLE:
            raise ImportError("pgmpy is required for inference")
        
        # Create pgmpy network
        edges = [(edge[0], edge[1]) for edge in self.network.edges]
        self.pgmpy_model = PgmpyBayesianNetwork(edges)
        
        # Add CPDs (Conditional Probability Distributions)
        for node_name, cpt in self.network.cpts.items():
            node = self.network.nodes[node_name]
            parents = self.network.get_parents(node_name)
            
            # Convert CPT to pgmpy format
            cpd_values = self._convert_cpt_to_pgmpy_format(cpt, node, parents)
            
            # Create TabularCPD
            cpd = TabularCPD(
                variable=node_name,
                variable_card=len(node.states),
                evidence=parents,
                evidence_card=[len(self.network.nodes[parent].states) for parent in parents],
                values=cpd_values
            )
            
            self.pgmpy_model.add_cpds(cpd)
    
    def _convert_cpt_to_pgmpy_format(self, cpt: CPT, node: Node, parents: List[str]) -> np.ndarray:
        """Convert our CPT format to pgmpy format."""
        if not parents:
            # Root node
            values = np.zeros(len(node.states))
            for i, state in enumerate(node.states):
                values[i] = cpt.probabilities.get(state, 0.0)
            return values.reshape(-1, 1)
        
        # Node with parents
        parent_states = [self.network.nodes[parent].states for parent in parents]
        total_parent_configs = np.prod([len(states) for states in parent_states])
        
        # Initialize array
        values = np.zeros((len(node.states), total_parent_configs))
        
        # Generate all parent configurations
        import itertools
        parent_configs = list(itertools.product(*parent_states))
        
        for config_idx, parent_config in enumerate(parent_configs):
            for state_idx, state in enumerate(node.states):
                # Create key in our CPT format
                key = f"{state}|{','.join(parent_config)}"
                values[state_idx, config_idx] = cpt.probabilities.get(key, 0.0)
        
        return values
    
    def query(self, 
              variables: List[str], 
              evidence: Optional[Dict[str, str]] = None,
              method: str = "VariableElimination",
              **kwargs) -> InferenceResult:
        """
        Perform inference query on the network.
        
        Args:
            variables: List of variables to query
            evidence: Dictionary of evidence {variable: state}
            method: Inference method to use
            **kwargs: Additional arguments for specific methods
        
        Returns:
            InferenceResult object with query results
        """
        if not PGMPY_AVAILABLE:
            raise ImportError("pgmpy is required for inference")
        
        start_time = time.time()
        
        # Validate inputs
        if not variables:
            raise ValueError("At least one variable must be specified for query")
        
        for var in variables:
            if var not in self.network.nodes:
                raise ValueError(f"Variable '{var}' not found in network")
        
        if evidence:
            for var, state in evidence.items():
                if var not in self.network.nodes:
                    raise ValueError(f"Evidence variable '{var}' not found in network")
                if state not in self.network.nodes[var].states:
                    raise ValueError(f"State '{state}' not valid for variable '{var}'")
        
        # Perform inference based on method
        if method == "VariableElimination":
            result = self._variable_elimination_query(variables, evidence)
        elif method == "BeliefPropagation":
            result = self._belief_propagation_query(variables, evidence)
        elif method == "JunctionTree":
            result = self._junction_tree_query(variables, evidence)
        elif method == "GibbsSampling":
            result = self._gibbs_sampling_query(variables, evidence, **kwargs)
        else:
            raise ValueError(f"Unsupported inference method: {method}")
        
        inference_time = time.time() - start_time
        
        return InferenceResult(
            query_variables=variables,
            evidence=evidence or {},
            probabilities=result,
            inference_time=inference_time,
            method=method,
            samples=kwargs.get('samples'),
            convergence_iterations=kwargs.get('convergence_iterations')
        )
    
    def _variable_elimination_query(self, 
                                   variables: List[str], 
                                   evidence: Optional[Dict[str, str]] = None) -> Dict[str, Dict[str, float]]:
        """Perform variable elimination inference."""
        inference = VariableElimination(self.pgmpy_model)
        
        # Convert evidence format if needed
        pgmpy_evidence = {}
        if evidence:
            for var, state in evidence.items():
                state_idx = self.network.nodes[var].states.index(state)
                pgmpy_evidence[var] = state_idx
        
        # Perform query
        query_result = inference.query(variables=variables, evidence=pgmpy_evidence)
        
        # Convert result to our format
        result = {}
        for var in variables:
            node = self.network.nodes[var]
            result[var] = {}
            for i, state in enumerate(node.states):
                result[var][state] = float(query_result.values[i])
        
        return result
    
    def _belief_propagation_query(self, 
                                 variables: List[str], 
                                 evidence: Optional[Dict[str, str]] = None) -> Dict[str, Dict[str, float]]:
        """Perform belief propagation inference."""
        inference = BeliefPropagation(self.pgmpy_model)
        
        # Convert evidence format
        pgmpy_evidence = {}
        if evidence:
            for var, state in evidence.items():
                state_idx = self.network.nodes[var].states.index(state)
                pgmpy_evidence[var] = state_idx
        
        # Perform query
        query_result = inference.query(variables=variables, evidence=pgmpy_evidence)
        
        # Convert result
        result = {}
        for var in variables:
            node = self.network.nodes[var]
            result[var] = {}
            for i, state in enumerate(node.states):
                result[var][state] = float(query_result.values[i])
        
        return result
    
    def _junction_tree_query(self, 
                            variables: List[str], 
                            evidence: Optional[Dict[str, str]] = None) -> Dict[str, Dict[str, float]]:
        """Perform junction tree inference."""
        inference = JunctionTree(self.pgmpy_model)
        
        # Convert evidence format
        pgmpy_evidence = {}
        if evidence:
            for var, state in evidence.items():
                state_idx = self.network.nodes[var].states.index(state)
                pgmpy_evidence[var] = state_idx
        
        # Perform query
        query_result = inference.query(variables=variables, evidence=pgmpy_evidence)
        
        # Convert result
        result = {}
        for var in variables:
            node = self.network.nodes[var]
            result[var] = {}
            for i, state in enumerate(node.states):
                result[var][state] = float(query_result.values[i])
        
        return result
    
    def _gibbs_sampling_query(self, 
                             variables: List[str], 
                             evidence: Optional[Dict[str, str]] = None,
                             samples: int = DEFAULT_SAMPLES) -> Dict[str, Dict[str, float]]:
        """Perform Gibbs sampling inference."""
        # Convert evidence format
        pgmpy_evidence = {}
        if evidence:
            for var, state in evidence.items():
                state_idx = self.network.nodes[var].states.index(state)
                pgmpy_evidence[var] = state_idx
        
        # Perform sampling
        sampler = GibbsSampling(self.pgmpy_model)
        samples_data = sampler.sample(size=samples, evidence=pgmpy_evidence)
        
        # Calculate probabilities from samples
        result = {}
        for var in variables:
            node = self.network.nodes[var]
            result[var] = {}
            
            # Count occurrences of each state
            for state in node.states:
                count = (samples_data[var] == node.states.index(state)).sum()
                result[var][state] = count / samples
        
        return result
    
    def get_marginal_probabilities(self, 
                                  variables: Optional[List[str]] = None,
                                  evidence: Optional[Dict[str, str]] = None) -> Dict[str, Dict[str, float]]:
        """Get marginal probabilities for variables."""
        if variables is None:
            variables = list(self.network.nodes.keys())
        
        result = self.query(variables, evidence, method="VariableElimination")
        return result.probabilities
    
    def get_conditional_probability(self, 
                                   query_var: str, 
                                   query_state: str,
                                   evidence: Dict[str, str]) -> float:
        """Get conditional probability P(query_var=query_state | evidence)."""
        result = self.query([query_var], evidence, method="VariableElimination")
        return result.probabilities[query_var].get(query_state, 0.0)
    
    def sensitivity_analysis(self, 
                           target_var: str, 
                           target_state: str,
                           parameter_var: str,
                           parameter_states: List[str],
                           evidence: Optional[Dict[str, str]] = None) -> Dict[str, float]:
        """Perform sensitivity analysis on a parameter."""
        results = {}
        
        for param_state in parameter_states:
            # Create modified evidence
            modified_evidence = evidence.copy() if evidence else {}
            modified_evidence[parameter_var] = param_state
            
            # Query target variable
            prob = self.get_conditional_probability(target_var, target_state, modified_evidence)
            results[param_state] = prob
        
        return results
    
    def most_probable_explanation(self, 
                                 evidence: Dict[str, str],
                                 query_variables: Optional[List[str]] = None) -> Dict[str, str]:
        """Find the most probable explanation for given evidence."""
        if query_variables is None:
            # Query all non-evidence variables
            query_variables = [var for var in self.network.nodes.keys() if var not in evidence]
        
        result = self.query(query_variables, evidence, method="VariableElimination")
        
        # Find most probable state for each variable
        mpe = {}
        for var in query_variables:
            max_state, _ = result.get_max_probability_state(var)
            mpe[var] = max_state
        
        return mpe
    
    def get_network_statistics(self) -> Dict[str, Any]:
        """Get comprehensive network statistics."""
        info = self.network.get_network_info()
        
        # Add inference-specific statistics
        stats = {
            "network_info": info,
            "inference_methods": ["VariableElimination", "BeliefPropagation", "JunctionTree", "GibbsSampling"],
            "pgmpy_available": PGMPY_AVAILABLE,
            "max_query_variables": len(self.network.nodes),
            "supported_evidence_types": "discrete_states"
        }
        
        return stats
    
    def validate_inference_setup(self) -> Tuple[bool, List[str]]:
        """Validate that the network is ready for inference."""
        errors = []
        
        # Check pgmpy availability
        if not PGMPY_AVAILABLE:
            errors.append("pgmpy is not available")
            return False, errors
        
        # Check network validity
        is_valid, network_errors = self.network.validate_network()
        if not is_valid:
            errors.extend(network_errors)
        
        # Check pgmpy model
        if self.pgmpy_model is None:
            errors.append("pgmpy model not built")
        
        return len(errors) == 0, errors 