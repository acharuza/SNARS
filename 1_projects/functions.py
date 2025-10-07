import matplotlib.pyplot as plt
import numpy as np


def square_lattice(n: int, m: int) -> np.ndarray:
    size = n * m
    adj_matrix = np.zeros((size, size), dtype=int)
    # i count edges from left to right and top to bottom
    # so possible connections are:
    # i - 1
    # i + 1
    # i - m
    # i + m
    # but we can mark them all by just marking the right and bottom connections from each node
    right = np.arange(size - 1)
    right = right[(right % m) != (m - 1)]
    adj_matrix[right, right + 1] = 1
    adj_matrix[right + 1, right] = 1

    bottom = np.arange(size - m)
    adj_matrix[bottom, bottom + m] = 1
    adj_matrix[bottom + m, bottom] = 1

    return adj_matrix

def chain(n: int) -> np.ndarray:
    """Generate a chain of size n."""
    pass

def complete_graph(n: int) -> np.ndarray:
    """Generate a complete graph of size n."""
    pass

def weighted_graph(n: int, weight: float) -> np.ndarray:
    """Generate a weighted complete graph of size n with given weight."""
    pass

def directed_graph(n: int) -> np.ndarray:
    """Generate a directed graph of size n."""
    pass

def draw(adj_matrix: np.ndarray) -> None:
    """Draw the graph represented by the adjacency matrix."""
    pass

def compute_vertex_degrees(adj_matrix: np.ndarray) -> np.ndarray:
    """Compute the vertex degrees of the graph represented by the adjacency matrix."""
    pass


if __name__ == "__main__":
    print(square_lattice(3, 5))