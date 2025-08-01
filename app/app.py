"""
Main Streamlit application for the Bayesian Network Playground.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Any
import json
import tempfile
import os
from pathlib import Path

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.bayesian_network import BayesianNetwork, Node, CPT
from src.inference_engine import InferenceEngine, InferenceResult
from src.visualization import NetworkVisualizer
from src.network_examples import load_example_networks, get_network_info
from src.config import STREAMLIT_CONFIG, PGMPY_CONFIG, UI_CONFIG


def setup_page_config():
    """Setup Streamlit page configuration."""
    st.set_page_config(
        page_title=STREAMLIT_CONFIG["page_title"],
        page_icon=STREAMLIT_CONFIG["page_icon"],
        layout=STREAMLIT_CONFIG["layout"],
        initial_sidebar_state=STREAMLIT_CONFIG["initial_sidebar_state"]
    )


def main():
    """Main application function."""
    setup_page_config()
    
    # Initialize session state
    if 'current_network' not in st.session_state:
        st.session_state.current_network = None
    if 'inference_engine' not in st.session_state:
        st.session_state.inference_engine = None
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = None
    if 'inference_results' not in st.session_state:
        st.session_state.inference_results = []
    
    # Sidebar
    sidebar()
    
    # Main content
    st.title("🧠 Interactive Bayesian Network Playground")
    st.markdown("Build, visualize, and perform inference on Bayesian Networks with advanced visualizations.")
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏗️ Network Builder", 
        "📊 Visualizations", 
        "🔍 Inference", 
        "📈 Analysis", 
        "💾 Save/Load"
    ])
    
    with tab1:
        network_builder_tab()
    
    with tab2:
        visualizations_tab()
    
    with tab3:
        inference_tab()
    
    with tab4:
        analysis_tab()
    
    with tab5:
        save_load_tab()


def sidebar():
    """Create sidebar with network selection and info."""
    st.sidebar.title("🎛️ Controls")
    
    # Network selection
    st.sidebar.subheader("📋 Example Networks")
    
    example_networks = load_example_networks()
    network_names = list(example_networks.keys())
    
    selected_example = st.sidebar.selectbox(
        "Choose an example network:",
        ["None"] + network_names,
        help="Select a pre-built example network to get started"
    )
    
    if selected_example != "None" and st.sidebar.button("Load Example"):
        load_example_network(selected_example, example_networks[selected_example])
        st.sidebar.success(f"Loaded {selected_example} network!")
    
    # Current network info
    if st.session_state.current_network:
        st.sidebar.subheader("📊 Current Network")
        network = st.session_state.current_network
        info = network.get_network_info()
        
        st.sidebar.metric("Nodes", info["node_count"])
        st.sidebar.metric("Edges", info["edge_count"])
        st.sidebar.metric("Valid", "✅" if info["is_valid"] else "❌")
        
        if not info["is_valid"]:
            st.sidebar.error("Network has validation errors:")
            for error in info["validation_errors"]:
                st.sidebar.error(f"• {error}")
    
    # Quick actions
    st.sidebar.subheader("⚡ Quick Actions")
    
    if st.sidebar.button("Clear Network"):
        st.session_state.current_network = None
        st.session_state.inference_engine = None
        st.session_state.visualizer = None
        st.session_state.inference_results = []
        st.sidebar.success("Network cleared!")
    
    if st.sidebar.button("Validate Network"):
        if st.session_state.current_network:
            validate_and_show_results()
        else:
            st.sidebar.warning("No network loaded")


def network_builder_tab():
    """Network construction tab."""
    st.header("🏗️ Network Builder")
    
    # Create new network
    col1, col2 = st.columns([2, 1])
    
    with col1:
        network_name = st.text_input(
            "Network Name:",
            value="My Bayesian Network",
            help="Enter a name for your network"
        )
        
        if st.button("Create New Network"):
            st.session_state.current_network = BayesianNetwork(network_name)
            st.session_state.inference_engine = None
            st.session_state.visualizer = None
            st.success(f"Created new network: {network_name}")
    
    with col2:
        st.info("💡 **Tip**: Start with an example network or create a new one from scratch.")
    
    # Network construction interface
    if st.session_state.current_network:
        network = st.session_state.current_network
        
        # Node management
        st.subheader("🔧 Node Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Add Node**")
            node_name = st.text_input("Node Name:", key="add_node_name")
            node_states = st.text_input(
                "States (comma-separated):", 
                value="True,False",
                key="add_node_states"
            )
            node_description = st.text_input("Description:", key="add_node_desc")
            
            if st.button("Add Node"):
                if node_name and node_states:
                    states = [s.strip() for s in node_states.split(",")]
                    node = Node(node_name, states, node_description)
                    try:
                        network.add_node(node)
                        st.success(f"Added node: {node_name}")
                        # Clear inputs
                        st.session_state.add_node_name = ""
                        st.session_state.add_node_states = "True,False"
                        st.session_state.add_node_desc = ""
                    except ValueError as e:
                        st.error(f"Error adding node: {e}")
        
        with col2:
            st.write("**Remove Node**")
            if network.nodes:
                node_to_remove = st.selectbox(
                    "Select node to remove:",
                    list(network.nodes.keys())
                )
                if st.button("Remove Node"):
                    if network.remove_node(node_to_remove):
                        st.success(f"Removed node: {node_to_remove}")
                    else:
                        st.error(f"Failed to remove node: {node_to_remove}")
        
        # Edge management
        st.subheader("🔗 Edge Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Add Edge**")
            if len(network.nodes) >= 2:
                from_node = st.selectbox("From:", list(network.nodes.keys()), key="edge_from")
                to_node = st.selectbox("To:", list(network.nodes.keys()), key="edge_to")
                
                if st.button("Add Edge"):
                    try:
                        network.add_edge(from_node, to_node)
                        st.success(f"Added edge: {from_node} → {to_node}")
                    except ValueError as e:
                        st.error(f"Error adding edge: {e}")
        
        with col2:
            st.write("**Remove Edge**")
            if network.edges:
                edge_to_remove = st.selectbox(
                    "Select edge to remove:",
                    [f"{e[0]} → {e[1]}" for e in network.edges]
                )
                if st.button("Remove Edge"):
                    from_node, to_node = edge_to_remove.split(" → ")
                    if network.remove_edge(from_node, to_node):
                        st.success(f"Removed edge: {edge_to_remove}")
                    else:
                        st.error(f"Failed to remove edge: {edge_to_remove}")
        
        # CPT management
        st.subheader("📊 Conditional Probability Tables")
        
        if network.nodes:
            selected_node = st.selectbox(
                "Select node for CPT:",
                list(network.nodes.keys())
            )
            
            if selected_node in network.nodes:
                node = network.nodes[selected_node]
                parents = network.get_parents(selected_node)
                
                st.write(f"**Node:** {selected_node}")
                st.write(f"**States:** {', '.join(node.states)}")
                st.write(f"**Parents:** {', '.join(parents) if parents else 'None'}")
                
                # Generate CPT interface
                generate_cpt_interface(network, selected_node)


def generate_cpt_interface(network: BayesianNetwork, node_name: str):
    """Generate interface for editing CPT of a specific node."""
    node = network.nodes[node_name]
    parents = network.get_parents(node_name)
    
    if not parents:
        # Root node - simple probability distribution
        st.write("**Root Node Probabilities:**")
        
        cpt_data = {}
        for state in node.states:
            prob = st.number_input(
                f"P({node_name}={state}):",
                min_value=0.0,
                max_value=1.0,
                value=1.0/len(node.states),
                step=0.01,
                key=f"cpt_{node_name}_{state}"
            )
            cpt_data[state] = prob
        
        # Check if probabilities sum to 1.0
        total_prob = sum(cpt_data.values())
        st.write(f"**Total Probability:** {total_prob:.3f}")
        
        if abs(total_prob - 1.0) > 0.01:
            st.warning("⚠️ Probabilities should sum to 1.0")
        else:
            st.success("✅ Probabilities sum to 1.0")
        
        if st.button(f"Set CPT for {node_name}"):
            cpt = CPT(node_name, [], cpt_data)
            try:
                network.set_cpt(cpt)
                st.success(f"CPT set for {node_name}")
            except ValueError as e:
                st.error(f"Error setting CPT: {e}")
    
    else:
        # Node with parents - conditional probability table
        st.write("**Conditional Probability Table:**")
        
        # Generate all parent configurations
        parent_states = [network.nodes[parent].states for parent in parents]
        import itertools
        parent_configs = list(itertools.product(*parent_states))
        
        # Create a DataFrame for easier editing
        cpt_data = {}
        
        for config in parent_configs:
            st.write(f"**Given:** {', '.join([f'{p}={s}' for p, s in zip(parents, config)])}")
            
            config_probs = {}
            for state in node.states:
                prob = st.number_input(
                    f"P({node_name}={state}):",
                    min_value=0.0,
                    max_value=1.0,
                    value=1.0/len(node.states),
                    step=0.01,
                    key=f"cpt_{node_name}_{state}_{'_'.join(config)}"
                )
                config_probs[state] = prob
                cpt_data[f"{state}|{','.join(config)}"] = prob
            
            # Check configuration probabilities
            config_total = sum(config_probs.values())
            st.write(f"**Sum:** {config_total:.3f}")
            
            if abs(config_total - 1.0) > 0.01:
                st.warning("⚠️ Probabilities should sum to 1.0")
            else:
                st.success("✅ Probabilities sum to 1.0")
            
            st.divider()
        
        if st.button(f"Set CPT for {node_name}"):
            cpt = CPT(node_name, parents, cpt_data)
            try:
                network.set_cpt(cpt)
                st.success(f"CPT set for {node_name}")
            except ValueError as e:
                st.error(f"Error setting CPT: {e}")


def visualizations_tab():
    """Network visualization tab."""
    st.header("📊 Visualizations")
    
    if not st.session_state.current_network:
        st.warning("No network loaded. Please create or load a network first.")
        return
    
    network = st.session_state.current_network
    
    # Initialize visualizer if needed
    if not st.session_state.visualizer:
        st.session_state.visualizer = NetworkVisualizer(network)
    
    visualizer = st.session_state.visualizer
    
    # Visualization options
    viz_type = st.selectbox(
        "Select visualization type:",
        ["Network Topology", "CPT Heatmaps", "Inference Results", "Dashboard"]
    )
    
    if viz_type == "Network Topology":
        st.subheader("🌐 Network Topology")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            layout = st.selectbox("Layout:", ["spring", "circular", "kamada_kawai", "shell"])
            show_labels = st.checkbox("Show Labels", value=True)
        
        with col2:
            fig = visualizer.plot_network_topology(layout=layout, show_labels=show_labels)
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "CPT Heatmaps":
        st.subheader("🔥 CPT Heatmaps")
        
        if network.nodes:
            selected_node = st.selectbox(
                "Select node for CPT visualization:",
                list(network.nodes.keys())
            )
            
            if selected_node in network.cpts:
                fig = visualizer.plot_cpt_heatmap(selected_node)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"No CPT defined for node '{selected_node}'")
    
    elif viz_type == "Inference Results":
        st.subheader("📈 Inference Results")
        
        if st.session_state.inference_results:
            selected_result = st.selectbox(
                "Select inference result:",
                range(len(st.session_state.inference_results)),
                format_func=lambda x: f"Result {x+1} ({st.session_state.inference_results[x].method})"
            )
            
            result = st.session_state.inference_results[selected_result]
            
            plot_type = st.selectbox("Plot type:", ["bar", "heatmap", "radar"])
            
            fig = visualizer.plot_inference_results(result, plot_type=plot_type)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show result details
            st.write("**Result Details:**")
            st.write(f"- Method: {result.method}")
            st.write(f"- Inference Time: {result.inference_time:.3f}s")
            st.write(f"- Query Variables: {', '.join(result.query_variables)}")
            if result.evidence:
                st.write(f"- Evidence: {result.evidence}")
            
            # Show probabilities table
            st.write("**Probabilities:**")
            result_df = result.to_dataframe()
            st.dataframe(result_df, use_container_width=True)
        else:
            st.info("No inference results available. Run inference first.")
    
    elif viz_type == "Dashboard":
        st.subheader("📊 Analysis Dashboard")
        
        if st.session_state.inference_results:
            fig = visualizer.create_dashboard(st.session_state.inference_results)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No inference results available for dashboard. Run inference first.")


def inference_tab():
    """Inference tab."""
    st.header("🔍 Inference")
    
    if not st.session_state.current_network:
        st.warning("No network loaded. Please create or load a network first.")
        return
    
    network = st.session_state.current_network
    
    # Validate network
    is_valid, errors = network.validate_network()
    if not is_valid:
        st.error("Network has validation errors:")
        for error in errors:
            st.error(f"• {error}")
        return
    
    # Initialize inference engine if needed
    if not st.session_state.inference_engine:
        try:
            st.session_state.inference_engine = InferenceEngine(network)
        except ImportError:
            st.error("pgmpy is required for inference. Please install it: `pip install pgmpy`")
            return
    
    inference_engine = st.session_state.inference_engine
    
    # Inference configuration
    st.subheader("⚙️ Inference Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Query variables
        query_variables = st.multiselect(
            "Query Variables:",
            list(network.nodes.keys()),
            default=list(network.nodes.keys())[:3] if network.nodes else [],
            help="Select variables to query"
        )
        
        # Inference method
        method = st.selectbox(
            "Inference Method:",
            PGMPY_CONFIG["inference_methods"],
            help="Select the inference algorithm to use"
        )
    
    with col2:
        # Evidence
        st.write("**Evidence (Optional):**")
        evidence = {}
        
        for node_name in network.nodes:
            if node_name not in query_variables:  # Don't set evidence for query variables
                node = network.nodes[node_name]
                state = st.selectbox(
                    f"{node_name}:",
                    ["None"] + node.states,
                    key=f"evidence_{node_name}"
                )
                if state != "None":
                    evidence[node_name] = state
        
        # Additional parameters
        if method == "GibbsSampling":
            samples = st.number_input(
                "Number of samples:",
                min_value=1000,
                max_value=50000,
                value=10000,
                step=1000
            )
        else:
            samples = None
    
    # Run inference
    if st.button("🚀 Run Inference", type="primary"):
        if not query_variables:
            st.error("Please select at least one query variable.")
            return
        
        with st.spinner("Running inference..."):
            try:
                kwargs = {}
                if samples:
                    kwargs['samples'] = samples
                
                result = inference_engine.query(
                    variables=query_variables,
                    evidence=evidence if evidence else None,
                    method=method,
                    **kwargs
                )
                
                # Store result
                st.session_state.inference_results.append(result)
                
                st.success(f"Inference completed in {result.inference_time:.3f}s!")
                
                # Show results
                st.subheader("📊 Results")
                
                # Create tabs for different result views
                tab1, tab2, tab3 = st.tabs(["Bar Chart", "Heatmap", "Table"])
                
                with tab1:
                    visualizer = NetworkVisualizer(network)
                    fig = visualizer.plot_inference_results(result, "bar")
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    fig = visualizer.plot_inference_results(result, "heatmap")
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab3:
                    result_df = result.to_dataframe()
                    st.dataframe(result_df, use_container_width=True)
                
            except Exception as e:
                st.error(f"Inference failed: {e}")


def analysis_tab():
    """Analysis tab."""
    st.header("📈 Analysis")
    
    if not st.session_state.current_network:
        st.warning("No network loaded. Please create or load a network first.")
        return
    
    network = st.session_state.current_network
    
    # Analysis options
    analysis_type = st.selectbox(
        "Select analysis type:",
        ["Network Statistics", "Sensitivity Analysis", "Most Probable Explanation", "Belief Propagation"]
    )
    
    if analysis_type == "Network Statistics":
        st.subheader("📊 Network Statistics")
        
        info = network.get_network_info()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Network Name", info["name"])
            st.metric("Number of Nodes", info["node_count"])
            st.metric("Number of Edges", info["edge_count"])
        
        with col2:
            st.metric("Is Acyclic", "✅" if info["is_acyclic"] else "❌")
            st.metric("Is Valid", "✅" if info["is_valid"] else "❌")
            if info["topological_order"]:
                st.metric("Topological Order", "→".join(info["topological_order"][:3]) + "...")
        
        # Node details
        st.write("**Node Details:**")
        node_data = []
        for node in info["nodes"]:
            node_data.append({
                "Name": node["name"],
                "States": ", ".join(node["states"]),
                "Description": node["description"]
            })
        
        st.dataframe(pd.DataFrame(node_data), use_container_width=True)
    
    elif analysis_type == "Sensitivity Analysis":
        st.subheader("🎯 Sensitivity Analysis")
        
        if not st.session_state.inference_engine:
            st.warning("Inference engine not initialized. Run inference first.")
            return
        
        inference_engine = st.session_state.inference_engine
        
        col1, col2 = st.columns(2)
        
        with col1:
            target_var = st.selectbox("Target Variable:", list(network.nodes.keys()))
            target_state = st.selectbox(
                "Target State:",
                network.nodes[target_var].states if target_var else []
            )
        
        with col2:
            parameter_var = st.selectbox("Parameter Variable:", list(network.nodes.keys()))
            parameter_states = network.nodes[parameter_var].states if parameter_var else []
        
        if st.button("Run Sensitivity Analysis"):
            try:
                results = inference_engine.sensitivity_analysis(
                    target_var=target_var,
                    target_state=target_state,
                    parameter_var=parameter_var,
                    parameter_states=parameter_states
                )
                
                # Plot results
                visualizer = NetworkVisualizer(network)
                fig = visualizer.plot_sensitivity_analysis(
                    results, target_var, target_state
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Show results table
                st.write("**Sensitivity Results:**")
                sensitivity_df = pd.DataFrame([
                    {"Parameter Value": param, "Probability": prob}
                    for param, prob in results.items()
                ])
                st.dataframe(sensitivity_df, use_container_width=True)
                
            except Exception as e:
                st.error(f"Sensitivity analysis failed: {e}")
    
    elif analysis_type == "Most Probable Explanation":
        st.subheader("🔍 Most Probable Explanation")
        
        if not st.session_state.inference_engine:
            st.warning("Inference engine not initialized. Run inference first.")
            return
        
        inference_engine = st.session_state.inference_engine
        
        # Evidence input
        st.write("**Evidence:**")
        evidence = {}
        
        for node_name in network.nodes:
            node = network.nodes[node_name]
            state = st.selectbox(
                f"{node_name}:",
                ["None"] + node.states,
                key=f"mpe_evidence_{node_name}"
            )
            if state != "None":
                evidence[node_name] = state
        
        if st.button("Find MPE"):
            if not evidence:
                st.warning("Please provide some evidence.")
                return
            
            try:
                mpe = inference_engine.most_probable_explanation(evidence)
                
                st.write("**Most Probable Explanation:**")
                mpe_df = pd.DataFrame([
                    {"Variable": var, "Most Probable State": state}
                    for var, state in mpe.items()
                ])
                st.dataframe(mpe_df, use_container_width=True)
                
            except Exception as e:
                st.error(f"MPE calculation failed: {e}")
    
    elif analysis_type == "Belief Propagation":
        st.subheader("🔄 Belief Propagation")
        
        if not st.session_state.inference_engine:
            st.warning("Inference engine not initialized. Run inference first.")
            return
        
        inference_engine = st.session_state.inference_engine
        
        # Evidence for propagation
        st.write("**Evidence for Propagation:**")
        evidence = {}
        
        for node_name in network.nodes:
            node = network.nodes[node_name]
            state = st.selectbox(
                f"{node_name}:",
                ["None"] + node.states,
                key=f"propagation_evidence_{node_name}"
            )
            if state != "None":
                evidence[node_name] = state
        
        steps = st.slider("Number of propagation steps:", 3, 10, 5)
        
        if st.button("Animate Belief Propagation"):
            if not evidence:
                st.warning("Please provide some evidence.")
                return
            
            try:
                visualizer = NetworkVisualizer(network)
                fig = visualizer.plot_belief_propagation(
                    inference_engine, evidence, steps=steps
                )
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Belief propagation failed: {e}")


def save_load_tab():
    """Save/Load tab."""
    st.header("💾 Save/Load")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💾 Save Network")
        
        if st.session_state.current_network:
            network = st.session_state.current_network
            
            save_format = st.selectbox("Save format:", ["JSON", "Pickle"])
            filename = st.text_input("Filename:", value=f"{network.name.lower().replace(' ', '_')}")
            
            if st.button("Save Network"):
                try:
                    if save_format == "JSON":
                        filepath = f"{filename}.json"
                    else:
                        filepath = f"{filename}.pkl"
                    
                    if network.save_network(filepath):
                        st.success(f"Network saved to {filepath}")
                        
                        # Provide download link
                        with open(filepath, 'rb') as f:
                            st.download_button(
                                label="Download File",
                                data=f.read(),
                                file_name=filepath,
                                mime="application/octet-stream"
                            )
                    else:
                        st.error("Failed to save network")
                        
                except Exception as e:
                    st.error(f"Error saving network: {e}")
        else:
            st.warning("No network to save")
    
    with col2:
        st.subheader("📂 Load Network")
        
        uploaded_file = st.file_uploader(
            "Choose a network file:",
            type=['json', 'pkl'],
            help="Upload a previously saved network file"
        )
        
        if uploaded_file is not None:
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_filepath = tmp_file.name
                
                # Load network
                network = BayesianNetwork.load_network(tmp_filepath)
                
                # Clean up temporary file
                os.unlink(tmp_filepath)
                
                if st.button("Load Network"):
                    st.session_state.current_network = network
                    st.session_state.inference_engine = None
                    st.session_state.visualizer = None
                    st.session_state.inference_results = []
                    st.success(f"Network loaded: {network.name}")
                    
            except Exception as e:
                st.error(f"Error loading network: {e}")


def load_example_network(name: str, network: BayesianNetwork):
    """Load an example network into the session."""
    st.session_state.current_network = network
    st.session_state.inference_engine = None
    st.session_state.visualizer = None
    st.session_state.inference_results = []


def validate_and_show_results():
    """Validate current network and show results."""
    network = st.session_state.current_network
    is_valid, errors = network.validate_network()
    
    if is_valid:
        st.sidebar.success("✅ Network is valid!")
    else:
        st.sidebar.error("❌ Network has validation errors:")
        for error in errors:
            st.sidebar.error(f"• {error}")


if __name__ == "__main__":
    main() 