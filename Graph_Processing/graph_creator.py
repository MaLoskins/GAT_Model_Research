# Graph_Processing/graph_creator.py

import pandas as pd
import networkx as nx

class GraphCreator:
    def __init__(self, dataframe, source_prefix="source_", target_prefix="target_"):
        """
        Initializes the GraphCreator with a dataframe.

        Parameters:
        - dataframe (pd.DataFrame): The dataframe containing node and edge information.
        - source_prefix (str): Prefix to differentiate source nodes.
        - target_prefix (str): Prefix to differentiate target nodes.
        """
        self.dataframe = dataframe.copy()
        self.source_prefix = source_prefix
        self.target_prefix = target_prefix
        self.graph = nx.MultiDiGraph()  # Using a MultiDiGraph to allow multiple edge types

    def create_graph(self, source_id_col, target_id_col, relationship_type="replies_to"):
        """
        Creates a heterogeneous graph based on source and target ID columns.

        Parameters:
        - source_id_col (str): Column name representing the source node ID.
        - target_id_col (str): Column name representing the target node ID.
        - relationship_type (str): Type of relationship between nodes.

        Returns:
        - nx.MultiDiGraph: The constructed graph.
        """
        # Prefix node IDs to ensure uniqueness across different types
        self.dataframe[source_id_col] = self.source_prefix + self.dataframe[source_id_col].astype(str)
        self.dataframe[target_id_col] = self.target_prefix + self.dataframe[target_id_col].astype(str)

        # Identify internal and external target IDs
        internal_ids = set(self.dataframe[source_id_col].unique())
        external_ids = set(self.dataframe[target_id_col].unique()) - internal_ids

        # Add nodes and edges
        for _, row in self.dataframe.iterrows():
            source_id = row[source_id_col]
            target_id = row[target_id_col]
            
            # Determine if target is internal or external
            is_internal = target_id in internal_ids

            # Add source node
            if not self.graph.has_node(source_id):
                self.graph.add_node(source_id, internal=True)  # Internal node

            # Add target node
            if not self.graph.has_node(target_id):
                self.graph.add_node(target_id, internal=is_internal)  # Internal or External

            # Add edge with relationship type
            self.graph.add_edge(source_id, target_id, relationship=relationship_type)
        
        return self.graph

    def get_graph(self):
        """
        Returns the constructed graph.

        Returns:
        - nx.MultiDiGraph: The constructed graph.
        """
        return self.graph
