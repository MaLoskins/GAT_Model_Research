# Graph_Processing/graph_visualizer.py

import networkx as nx
import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config

class GraphVisualizer:
    def __init__(self, graph):
        """
        Initializes the GraphVisualizer with a NetworkX graph.

        Parameters:
        - graph (nx.DiGraph): The graph to visualize.
        """
        self.graph = graph

    def visualize(self, heading="### Heterogeneous Graph Visualization"):
        """
        Generates and displays the graph visualization in Streamlit using streamlit-agraph.

        Parameters:
        - heading (str): The heading to display above the graph.
        """
        try:
            st.markdown(heading)

            # Define nodes and edges for streamlit-agraph
            nodes = []
            edges = []

            # Customize node attributes
            degrees = dict(self.graph.degree())
            node_id_set = set()
            duplicates = set()

            for node_id, data in self.graph.nodes(data=True):
                node_id_str = str(node_id)
                if node_id_str in node_id_set:
                    duplicates.add(node_id_str)
                node_id_set.add(node_id_str)
                
                node = Node(
                    id=node_id_str,
                    label=node_id_str,
                    title=f"ID: {node_id_str}\nDegree: {degrees.get(node_id, 0)}",
                    size=degrees.get(node_id, 5) * 5,  # Adjust size based on degree
                    color='blue' if data.get('internal', True) else 'red'
                )
                nodes.append(node)

            if duplicates:
                st.error(f"❌ Duplicated node IDs after string conversion: {duplicates}")
                return  # Halt visualization to prevent errors
            else:
                st.success("✅ All node IDs are unique.")

            # Customize edge attributes
            for source, target, data in self.graph.edges(data=True):
                edge = Edge(
                    source=str(source),
                    target=str(target),
                    label=data.get('relationship', ''),
                    color='#888'  # Customize edge color if needed
                )
                edges.append(edge)

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
            st.error(f"❌ Error during graph visualization:\n{e}")
