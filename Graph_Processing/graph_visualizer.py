# Graph_Processing/graph_visualizer.py

import networkx as nx
import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config
from pyvis.network import Network
import streamlit.components.v1 as components
import tempfile
import os

class GraphVisualizer:
    def __init__(self, graph):
        """
        Initializes the GraphVisualizer with a NetworkX graph.

        Parameters:
        - graph (nx.DiGraph): The graph to visualize.
        """
        self.graph = graph

    def visualize_agraph(self, heading="### Heterogeneous Graph Visualization"):
        """
        Generates and displays the graph visualization in Streamlit using streamlit-agraph.

        Parameters:
        - heading (str): The heading to display above the graph.
        """
        try:
            st.markdown(heading)

            # Define nodes and edges for streamlit-agraph
            degrees = dict(self.graph.degree())
            node_ids = [str(node) for node in self.graph.nodes]
            
            # Check for duplicate node IDs after string conversion
            if len(node_ids) != len(set(node_ids)):
                duplicates = set([x for x in node_ids if node_ids.count(x) > 1])
                st.error(f"❌ Duplicated node IDs after string conversion: {duplicates}")
                return  # Halt visualization to prevent errors
            else:
                st.success("✅ All node IDs are unique.")

            # Create nodes using list comprehension
            nodes = [
                Node(
                    id=str(node_id),
                    label=str(node_id),
                    title=f"ID: {str(node_id)}\nDegree: {degrees.get(node_id, 0)}",
                    size=degrees.get(node_id, 5) * 5,  # Adjust size based on degree
                    color='blue' if self.graph.nodes[node_id].get('internal', True) else 'red'
                )
                for node_id in self.graph.nodes
            ]

            # Create edges using list comprehension
            edges = [
                Edge(
                    source=str(source),
                    target=str(target),
                    label=data.get('relationship', ''),
                    color='#888'  # Customize edge color if needed
                )
                for source, target, data in self.graph.edges(data=True)
            ]

            # Configure graph appearance
            config = Config(
                height=600,
                width='100%',
                directed=True,
                nodeHighlightBehavior=True,
                highlightColor="#F7A7A6",
                collapsible=True,
                node={'fontSize': 12, 'size': 20},
                link={'labelProperty': 'label'}
            )

            # Render the graph
            agraph(nodes=nodes, edges=edges, config=config)

        except Exception as e:
            st.error(f"❌ Error during agraph visualization:\n{e}")

    def visualize_pyvis(self, heading="### Heterogeneous Graph Visualization with PyVis"):
        """
        Generates and displays the graph visualization in Streamlit using PyVis.

        Parameters:
        - heading (str): The heading to display above the graph.
        """
        try:
            st.markdown(heading)

            # Initialize PyVis Network
            net = Network(height='600px', width='100%', directed=True, notebook=False)

            # Customize PyVis options for performance
            net.barnes_hut(gravity=-80000, central_gravity=0.3, spring_length=250, spring_strength=0.001, damping=0.09)

            # Add nodes with attributes
            degrees = dict(self.graph.degree())
            for node_id, data in self.graph.nodes(data=True):
                net.add_node(
                    str(node_id),
                    label=str(node_id),
                    title=f"ID: {str(node_id)}\nDegree: {degrees.get(node_id, 0)}",
                    size=degrees.get(node_id, 5) * 2,  # Adjust size based on degree
                    color='blue' if data.get('internal', True) else 'red'
                )

            # Add edges with attributes
            for source, target, data in self.graph.edges(data=True):
                net.add_edge(
                    str(source),
                    str(target),
                    title=data.get('relationship', ''),
                    color='#888'
                )

            # Use TemporaryDirectory for safer temporary file handling
            with tempfile.TemporaryDirectory() as tmpdirname:
                tmp_file_path = os.path.join(tmpdirname, "graph.html")
                net.save_graph(tmp_file_path)

                # Verify that the file was created
                if not os.path.exists(tmp_file_path):
                    st.error("❌ Failed to create the PyVis HTML file for visualization.")
                    return

                # Check if the file is not empty
                if os.path.getsize(tmp_file_path) == 0:
                    st.error("❌ The generated PyVis HTML file is empty.")
                    return

                # Read the generated HTML
                with open(tmp_file_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()

                # Check if 'html_content' is not empty
                if not html_content.strip():
                    st.error("❌ PyVis graph HTML content is empty.")
                    return

                # Display the PyVis graph in Streamlit
                components.html(html_content, height=600, scrolling=True)

        except Exception as e:
            st.error(f"❌ Error during PyVis visualization:\n{e}")
