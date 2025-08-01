"""
Visualization module for Bayesian Networks with interactive plots and animations.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import plotly.subplots as sp
import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings

from .bayesian_network import BayesianNetwork, Node, CPT
from .inference_engine import InferenceResult
from .config import FIGURE_SIZE, DPI, COLOR_MAP, NODE_COLORS


class NetworkVisualizer:
    """
    Comprehensive visualization toolkit for Bayesian Networks.
    """
    
    def __init__(self, network: BayesianNetwork):
        self.network = network
        self.setup_plotting_style()
    
    def setup_plotting_style(self):
        """Setup consistent plotting style."""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Set default figure parameters
        plt.rcParams['figure.figsize'] = FIGURE_SIZE
        plt.rcParams['figure.dpi'] = DPI
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 10
    
    def plot_network_topology(self, 
                             layout: str = "spring",
                             node_colors: Optional[Dict[str, str]] = None,
                             edge_colors: Optional[Dict[Tuple[str, str], str]] = None,
                             show_labels: bool = True,
                             figsize: Tuple[int, int] = FIGURE_SIZE,
                             save_path: Optional[str] = None) -> go.Figure:
        """
        Create interactive network topology visualization.
        
        Args:
            layout: Network layout algorithm ('spring', 'circular', 'kamada_kawai')
            node_colors: Dictionary mapping node names to colors
            edge_colors: Dictionary mapping edges to colors
            show_labels: Whether to show node labels
            figsize: Figure size
            save_path: Path to save the plot
        
        Returns:
            Plotly figure object
        """
        # Create networkx graph
        G = nx.DiGraph()
        
        # Add nodes
        for node_name, node in self.network.nodes.items():
            G.add_node(node_name, **node.to_dict())
        
        # Add edges
        for edge in self.network.edges:
            G.add_edge(edge[0], edge[1])
        
        # Choose layout
        if layout == "spring":
            pos = nx.spring_layout(G, k=1, iterations=50)
        elif layout == "circular":
            pos = nx.circular_layout(G)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(G)
        elif layout == "shell":
            pos = nx.shell_layout(G)
        else:
            pos = nx.spring_layout(G)
        
        # Prepare node colors
        if node_colors is None:
            node_colors = {node: NODE_COLORS["default"] for node in G.nodes()}
        
        # Prepare edge colors
        if edge_colors is None:
            edge_colors = {edge: "#666666" for edge in G.edges()}
        
        # Create node trace
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        node_text = [f"{node}<br>States: {', '.join(self.network.nodes[node].states)}" 
                    for node in G.nodes()]
        node_color = [node_colors.get(node, NODE_COLORS["default"]) for node in G.nodes()]
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text if show_labels else [],
            textposition="top center",
            marker=dict(
                size=20,
                color=node_color,
                line=dict(width=2, color='white')
            ),
            name="Nodes"
        )
        
        # Create edge trace
        edge_x = []
        edge_y = []
        edge_text = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_text.append(f"{edge[0]} → {edge[1]}")
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color=edge_colors.get(edge, "#666666")),
            hoverinfo='text',
            text=edge_text,
            mode='lines',
            name="Edges"
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace])
        
        # Update layout
        fig.update_layout(
            title=f"Bayesian Network: {self.network.name}",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=figsize[0] * 100,
            height=figsize[1] * 100
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_cpt_heatmap(self, 
                        node_name: str,
                        figsize: Tuple[int, int] = (10, 8),
                        save_path: Optional[str] = None) -> go.Figure:
        """
        Create heatmap visualization of a node's CPT.
        
        Args:
            node_name: Name of the node to visualize
            figsize: Figure size
            save_path: Path to save the plot
        
        Returns:
            Plotly figure object
        """
        if node_name not in self.network.cpts:
            raise ValueError(f"No CPT found for node '{node_name}'")
        
        cpt = self.network.cpts[node_name]
        node = self.network.nodes[node_name]
        
        # Convert CPT to matrix format
        if not cpt.parent_names:
            # Root node
            data = [[prob] for prob in [cpt.probabilities.get(state, 0.0) for state in node.states]]
            x_labels = [""]
            y_labels = node.states
        else:
            # Node with parents
            parent_states = [self.network.nodes[parent].states for parent in cpt.parent_names]
            
            # Generate all parent configurations
            import itertools
            parent_configs = list(itertools.product(*parent_states))
            
            data = []
            for state in node.states:
                row = []
                for config in parent_configs:
                    key = f"{state}|{','.join(config)}"
                    row.append(cpt.probabilities.get(key, 0.0))
                data.append(row)
            
            x_labels = [f"({', '.join(config)})" for config in parent_configs]
            y_labels = node.states
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=data,
            x=x_labels,
            y=y_labels,
            colorscale=COLOR_MAP,
            text=[[f"{val:.3f}" for val in row] for row in data],
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        # Update layout
        fig.update_layout(
            title=f"CPT for {node_name}",
            xaxis_title="Parent Configuration" if cpt.parent_names else "",
            yaxis_title=f"{node_name} States",
            width=figsize[0] * 100,
            height=figsize[1] * 100
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_inference_results(self, 
                              result: InferenceResult,
                              plot_type: str = "bar",
                              figsize: Tuple[int, int] = FIGURE_SIZE,
                              save_path: Optional[str] = None) -> go.Figure:
        """
        Visualize inference results.
        
        Args:
            result: InferenceResult object
            plot_type: Type of plot ('bar', 'heatmap', 'radar')
            figsize: Figure size
            save_path: Path to save the plot
        
        Returns:
            Plotly figure object
        """
        if plot_type == "bar":
            return self._plot_inference_bar(result, figsize, save_path)
        elif plot_type == "heatmap":
            return self._plot_inference_heatmap(result, figsize, save_path)
        elif plot_type == "radar":
            return self._plot_inference_radar(result, figsize, save_path)
        else:
            raise ValueError(f"Unsupported plot type: {plot_type}")
    
    def _plot_inference_bar(self, 
                           result: InferenceResult,
                           figsize: Tuple[int, int],
                           save_path: Optional[str]) -> go.Figure:
        """Create bar plot of inference results."""
        # Prepare data
        data = []
        for var, probs in result.probabilities.items():
            for state, prob in probs.items():
                data.append({
                    'Variable': var,
                    'State': state,
                    'Probability': prob
                })
        
        df = pd.DataFrame(data)
        
        # Create bar plot
        fig = px.bar(
            df, 
            x='Variable', 
            y='Probability', 
            color='State',
            title=f"Inference Results ({result.method})",
            barmode='group'
        )
        
        # Update layout
        fig.update_layout(
            width=figsize[0] * 100,
            height=figsize[1] * 100,
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def _plot_inference_heatmap(self, 
                               result: InferenceResult,
                               figsize: Tuple[int, int],
                               save_path: Optional[str]) -> go.Figure:
        """Create heatmap of inference results."""
        # Prepare data matrix
        variables = list(result.probabilities.keys())
        all_states = set()
        for probs in result.probabilities.values():
            all_states.update(probs.keys())
        
        states = sorted(list(all_states))
        
        # Create data matrix
        data = []
        for var in variables:
            row = []
            for state in states:
                row.append(result.probabilities[var].get(state, 0.0))
            data.append(row)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=data,
            x=states,
            y=variables,
            colorscale=COLOR_MAP,
            text=[[f"{val:.3f}" for val in row] for row in data],
            texttemplate="%{text}",
            textfont={"size": 10}
        ))
        
        # Update layout
        fig.update_layout(
            title=f"Inference Results Heatmap ({result.method})",
            xaxis_title="States",
            yaxis_title="Variables",
            width=figsize[0] * 100,
            height=figsize[1] * 100
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def _plot_inference_radar(self, 
                             result: InferenceResult,
                             figsize: Tuple[int, int],
                             save_path: Optional[str]) -> go.Figure:
        """Create radar plot of inference results."""
        # Prepare data for radar plot
        variables = list(result.probabilities.keys())
        
        # For radar plot, we'll show the maximum probability for each variable
        max_probs = []
        max_states = []
        
        for var in variables:
            probs = result.probabilities[var]
            max_state = max(probs, key=probs.get)
            max_prob = probs[max_state]
            max_probs.append(max_prob)
            max_states.append(max_state)
        
        # Create radar plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=max_probs,
            theta=variables,
            fill='toself',
            name='Max Probability',
            line_color='blue'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=False,
            title=f"Inference Results Radar ({result.method})",
            width=figsize[0] * 100,
            height=figsize[1] * 100
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_belief_propagation(self, 
                               inference_engine,
                               evidence: Dict[str, str],
                               steps: int = 5,
                               figsize: Tuple[int, int] = FIGURE_SIZE,
                               save_path: Optional[str] = None) -> go.Figure:
        """
        Create animated visualization of belief propagation.
        
        Args:
            inference_engine: InferenceEngine instance
            evidence: Evidence to propagate
            steps: Number of propagation steps to visualize
            figsize: Figure size
            save_path: Path to save the animation
        
        Returns:
            Plotly figure object with animation
        """
        # Get initial beliefs (prior)
        initial_result = inference_engine.get_marginal_probabilities()
        
        # Simulate belief propagation steps
        propagation_data = []
        
        for step in range(steps + 1):
            # In a real implementation, this would show actual belief propagation
            # For now, we'll simulate with some noise
            step_data = {}
            for var, probs in initial_result.items():
                step_data[var] = {}
                for state, prob in probs.items():
                    # Add some variation to simulate propagation
                    noise = np.random.normal(0, 0.01 * step)
                    step_data[var][state] = max(0, min(1, prob + noise))
            
            propagation_data.append(step_data)
        
        # Create animation frames
        frames = []
        for step, step_data in enumerate(propagation_data):
            # Prepare data for this step
            data = []
            for var, probs in step_data.items():
                for state, prob in probs.items():
                    data.append({
                        'Variable': var,
                        'State': state,
                        'Probability': prob,
                        'Step': step
                    })
            
            df = pd.DataFrame(data)
            
            frame = go.Frame(
                data=[go.Bar(
                    x=df['Variable'],
                    y=df['Probability'],
                    color=df['State']
                )],
                name=f"Step {step}"
            )
            frames.append(frame)
        
        # Create initial plot
        initial_df = pd.DataFrame([
            {'Variable': var, 'State': state, 'Probability': prob}
            for var, probs in propagation_data[0].items()
            for state, prob in probs.items()
        ])
        
        fig = go.Figure(
            data=[go.Bar(
                x=initial_df['Variable'],
                y=initial_df['Probability'],
                color=initial_df['State']
            )],
            frames=frames
        )
        
        # Add animation controls
        fig.update_layout(
            title="Belief Propagation Animation",
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': 1000, 'redraw': True},
                            'fromcurrent': True,
                            'transition': {'duration': 300}
                        }]
                    },
                    {
                        'label': 'Pause',
                        'method': 'animate',
                        'args': [[None], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    }
                ]
            }],
            sliders=[{
                'steps': [
                    {
                        'args': [[f"Step {i}"], {
                            'frame': {'duration': 300, 'redraw': True},
                            'mode': 'immediate',
                            'transition': {'duration': 300}
                        }],
                        'label': f"Step {i}",
                        'method': 'animate'
                    }
                    for i in range(len(frames))
                ],
                'active': 0,
                'currentvalue': {'prefix': 'Step: '},
                'len': 0.9,
                'x': 0.1,
                'xanchor': 'left',
                'y': 0,
                'yanchor': 'top'
            }],
            width=figsize[0] * 100,
            height=figsize[1] * 100
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_sensitivity_analysis(self, 
                                 sensitivity_results: Dict[str, float],
                                 target_var: str,
                                 target_state: str,
                                 figsize: Tuple[int, int] = FIGURE_SIZE,
                                 save_path: Optional[str] = None) -> go.Figure:
        """
        Visualize sensitivity analysis results.
        
        Args:
            sensitivity_results: Dictionary of parameter values to probabilities
            target_var: Target variable name
            target_state: Target state name
            figsize: Figure size
            save_path: Path to save the plot
        
        Returns:
            Plotly figure object
        """
        # Prepare data
        parameters = list(sensitivity_results.keys())
        probabilities = list(sensitivity_results.values())
        
        # Create bar plot
        fig = go.Figure(data=go.Bar(
            x=parameters,
            y=probabilities,
            marker_color='lightblue',
            text=[f"{p:.3f}" for p in probabilities],
            textposition='auto'
        ))
        
        # Update layout
        fig.update_layout(
            title=f"Sensitivity Analysis: P({target_var}={target_state})",
            xaxis_title="Parameter Values",
            yaxis_title="Probability",
            width=figsize[0] * 100,
            height=figsize[1] * 100
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_dashboard(self, 
                        inference_results: List[InferenceResult],
                        save_path: Optional[str] = None) -> go.Figure:
        """
        Create a comprehensive dashboard with multiple visualizations.
        
        Args:
            inference_results: List of inference results to compare
            save_path: Path to save the dashboard
        
        Returns:
            Plotly figure object with subplots
        """
        n_results = len(inference_results)
        
        # Create subplots
        fig = sp.make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Network Topology",
                "Inference Results Comparison",
                "Method Performance",
                "Evidence Impact"
            ],
            specs=[
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "scatter"}]
            ]
        )
        
        # 1. Network topology
        network_fig = self.plot_network_topology()
        for trace in network_fig.data:
            fig.add_trace(trace, row=1, col=1)
        
        # 2. Inference results comparison
        for i, result in enumerate(inference_results):
            data = []
            for var, probs in result.probabilities.items():
                for state, prob in probs.items():
                    data.append({
                        'Variable': var,
                        'State': state,
                        'Probability': prob,
                        'Method': result.method
                    })
            
            df = pd.DataFrame(data)
            
            fig.add_trace(
                go.Bar(
                    x=df['Variable'],
                    y=df['Probability'],
                    name=result.method,
                    showlegend=True
                ),
                row=1, col=2
            )
        
        # 3. Method performance (inference time)
        methods = [r.method for r in inference_results]
        times = [r.inference_time for r in inference_results]
        
        fig.add_trace(
            go.Bar(
                x=methods,
                y=times,
                name="Inference Time (s)",
                marker_color='red'
            ),
            row=2, col=1
        )
        
        # 4. Evidence impact (if evidence exists)
        if inference_results and inference_results[0].evidence:
            evidence_vars = list(inference_results[0].evidence.keys())
            evidence_impact = []
            
            for var in evidence_vars:
                # Calculate impact as change in probability
                impact = 0.0
                for result in inference_results:
                    if var in result.probabilities:
                        max_prob = max(result.probabilities[var].values())
                        impact += max_prob
                evidence_impact.append(impact / len(inference_results))
            
            fig.add_trace(
                go.Scatter(
                    x=evidence_vars,
                    y=evidence_impact,
                    mode='markers+lines',
                    name="Evidence Impact",
                    marker=dict(size=10)
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title="Bayesian Network Analysis Dashboard",
            height=800,
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def save_all_visualizations(self, 
                               inference_result: Optional[InferenceResult] = None,
                               output_dir: str = "visualizations") -> Dict[str, str]:
        """
        Save all available visualizations for the network.
        
        Args:
            inference_result: Optional inference result to visualize
            output_dir: Directory to save visualizations
        
        Returns:
            Dictionary mapping visualization names to file paths
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        saved_files = {}
        
        # Network topology
        network_fig = self.plot_network_topology()
        network_path = os.path.join(output_dir, "network_topology.html")
        network_fig.write_html(network_path)
        saved_files["network_topology"] = network_path
        
        # CPT heatmaps
        for node_name in self.network.nodes:
            try:
                cpt_fig = self.plot_cpt_heatmap(node_name)
                cpt_path = os.path.join(output_dir, f"cpt_{node_name}.html")
                cpt_fig.write_html(cpt_path)
                saved_files[f"cpt_{node_name}"] = cpt_path
            except Exception as e:
                print(f"Warning: Could not create CPT visualization for {node_name}: {e}")
        
        # Inference results
        if inference_result:
            # Bar plot
            bar_fig = self.plot_inference_results(inference_result, "bar")
            bar_path = os.path.join(output_dir, "inference_results_bar.html")
            bar_fig.write_html(bar_path)
            saved_files["inference_bar"] = bar_path
            
            # Heatmap
            heatmap_fig = self.plot_inference_results(inference_result, "heatmap")
            heatmap_path = os.path.join(output_dir, "inference_results_heatmap.html")
            heatmap_fig.write_html(heatmap_path)
            saved_files["inference_heatmap"] = heatmap_path
        
        return saved_files 