import argparse
import networkx as nx
from pyvis.network import Network
from scipy.io import mmread

NETWORK_PARAMS = {
    "height": "900px",
    "width": "100%",
    "bgcolor": "#1a1a2e",
    "font_color": "white"
}


def physics_config(net):
    net.toggle_physics(False)

def read_network(file_path):
    matrix = mmread(file_path)
    G = nx.from_scipy_sparse_array(matrix)
    return G

def draw_network(G, output_html="results/network.html"):
    try:
        pos = nx.kamada_kawai_layout(G, scale=2000)
    except:
        pos = nx.spring_layout(G, k=3, iterations=100, seed=42, scale=2000)

    net = Network(**NETWORK_PARAMS)
    physics_config(net)

    # nodes
    for node in G.nodes():
        degree = G.degree(node)
        x, y = pos[node]
        net.add_node(node, 
                     label=str(node), 
                     size=10 + degree * 2,
                     color="#16a085",
                     x=x,
                     y=y,
                     title=f"Node {node}<br>Degree: {degree}")
    
    # edges
    for src, dst in G.edges():
        net.add_edge(src, dst, color="#555555", width=1.5)
        
    net.write_html(output_html)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to the input .mtx file")
    parser.add_argument("--output", "-o", type=str, default="results/network.html", help="Path to the output HTML file")
    args = parser.parse_args()

    G = read_network(args.input)
    draw_network(G, args.output)