from abc import ABC, abstractmethod
from typing import List, Any, Tuple
import numpy as np
import networkx as nx

class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query_embedding: np.ndarray, top_k: int = 10) -> Tuple[list[str], nx.Graph]:
        """
        Returns:
            node_ids: list of node identifiers
            subgraph: NetworkX graph containing the retrieved subgraph
        """
        raise NotImplementedError