# Implementation of a configuration model loosely based on implementation existing in networkx and wikipedia
# https://en.wikipedia.org/wiki/Configuration_model#Generating_from_the_Micro-Canonical_Configuration_Model
# + not allowing self-loops and multiple edges
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

# I assume we want to have simple graph based on lecture slides


def configuration_model(deg_sequence, seed=123):
    if not check_if_we_can_draw_simple_graph(deg_sequence):
        return "A simple graph cannot be drawn with given degree sequence"
    n = len(deg_sequence)
    G = nx.empty_graph(n)

    if n == 0:
        return G

    stubs = []  # half edges
    for node, deg in enumerate(deg_sequence):
        if deg > 0:
            stubs.extend([node] * deg)

    rng = np.random.default_rng(seed)
    rng.shuffle(stubs)

    while len(stubs) >= 2:
        # selecting random stubs
        u = stubs.pop()
        v = stubs.pop()

        if u == v or G.has_edge(u, v):
            # putting back the stubs
            stubs.append(u)
            stubs.append(v)
            rng.shuffle(stubs)
            continue

        # connecting the nodes
        G.add_edge(u, v)

    if len(stubs) == 0:
        return G

    return "Couldn't connect all stubs"


def check_if_we_can_draw_simple_graph(deg_sequence):
    # I use Havel–Hakimi algorithm based on https://www.geeksforgeeks.org/dsa/find-if-a-degree-sequence-can-form-a-simple-graph-havel-hakimi-algorithm/
    a = deg_sequence.copy()

    while True:
        a = sorted(a, reverse=True)

        if all(x == 0 for x in a):
            return True

        v = a[0]
        a = a[1:]

        if v > len(a):
            return False

        for i in range(v):
            a[i] -= 1

            if a[i] < 0:
                return False


def draw_configuration_model(G):
    pos = nx.spring_layout(G)
    degree_sequence = [d for n, d in G.degree()]

    plt.figure(figsize=(4, 4))
    nx.draw(
        G,
        pos,
        node_color="lightblue",
        node_size=[50 + d * 50 for d in degree_sequence],
        font_weight="bold",
        edge_color="gray",
    )
    plt.title("Configuration Model Graph")
    plt.show()
