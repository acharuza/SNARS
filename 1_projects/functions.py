import matplotlib.pyplot as plt
import numpy as np

# utility functions


def _plot_settings():
    # i used genai for this function
    plt.xticks([])
    plt.yticks([])
    plt.box(False)
    plt.axis("equal")


def _vertices_on_circle(n):
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    radius = 5
    return np.array([[radius * np.cos(t), radius * np.sin(t)] for t in theta])


def _random_graph(n_vertices, n_edges, seed=123):
    np.random.seed(seed)

    # generate all vertices from the same circle
    vertices = _vertices_on_circle(n_vertices)

    # generate random edges
    edges = set()
    while len(edges) < n_edges:
        u = np.random.randint(0, n_vertices)
        v = np.random.randint(0, n_vertices)
        if u != v:
            edges.add((min(u, v), max(u, v)))  # avoid duplicate edges

    return vertices, list(edges)


# P 1.1
def draw_square_lattice(n, m):
    plt.figure(figsize=(n // 2, m // 2))
    _plot_settings()

    # generate all vertices
    vertices = np.array([[i, j] for i in range(n) for j in range(m)])

    # draw edges
    # for that we can draw from each vertex its right neighbor and its bottom neighbor to create square lattice
    for i in range(n):
        for j in range(m):
            if i < n - 1:
                plt.plot([i, i + 1], [j, j], color="black")
            if j < m - 1:
                plt.plot([i, i], [j, j + 1], color="black")

    # draw vertices
    plt.scatter(vertices[:, 0], vertices[:, 1], color="blue", zorder=2)

    # show plot
    plt.title(f"{n} x {m} Square Lattice")
    plt.show()


# P 1.2
def draw_chain(n):
    plt.figure(figsize=(5, 5))
    _plot_settings()

    # generate all vertices from the same circle
    vertices = _vertices_on_circle(n)

    # draw edges
    # to draw edges we can connect each vertex to the next one in the list and connect the last one to the first one
    for i in range(n):
        plt.plot(
            [vertices[i][0], vertices[(i + 1) % n][0]],
            [vertices[i][1], vertices[(i + 1) % n][1]],
            color="black",
        )

    # draw vertices
    plt.scatter(vertices[:, 0], vertices[:, 1], color="blue", zorder=2)

    plt.title(f"Chain of length {n}")
    plt.show()


# P 1.3
def draw_complete_graph(n):
    plt.figure(figsize=(5, 5))
    _plot_settings()

    # generate all vertices from the same circle
    vertices = _vertices_on_circle(n)

    # draw edges
    # we need to connect every vertex to everything else
    for i in range(n):
        for j in range(i + 1, n):
            plt.plot(
                [vertices[i][0], vertices[j][0]],
                [vertices[i][1], vertices[j][1]],
                color="black",
            )

    # draw vertices
    plt.scatter(vertices[:, 0], vertices[:, 1], color="blue", zorder=2)

    plt.title(f"Complete graph on {n} vertices")
    plt.show()


# P 1.4
def draw_weighted_graph(n_vertices, n_edges, seed=123):
    # generate random graph
    vertices, edges = _random_graph(n_vertices, n_edges, seed)
    # generate random weights for edges
    np.random.seed(seed)
    weights = np.random.randint(1, 10, size=len(edges))

    plt.figure(figsize=(5, 5))
    _plot_settings()

    # draw edges with weights
    for (u, v), weight in zip(edges, weights):
        plt.plot(
            [vertices[u][0], vertices[v][0]],
            [vertices[u][1], vertices[v][1]],
            color="black",
        )
        mid_x = (vertices[u][0] + vertices[v][0]) / 2
        mid_y = (vertices[u][1] + vertices[v][1]) / 2
        plt.text(mid_x, mid_y, str(weight), color="blue", fontsize=12, zorder=3)

    # draw vertices
    plt.scatter(vertices[:, 0], vertices[:, 1], color="blue", zorder=2)

    plt.title(f"Random weighted graph\nwith {n_vertices} vertices and {n_edges} edges")
    plt.show()


# P 1.5
def draw_directed_graph(n_vertices, n_edges, seed=123):
    # generate random graph
    vertices, edges = _random_graph(n_vertices, n_edges, seed)
    # generate random directions for edges
    np.random.seed(seed)

    directions = []
    i = 0
    while i < len(edges):
        directions.append(np.random.choice([1, 0, -1]))
        i = i + 1 if directions[-1] != 0 else i + 2

    # 1 for u->v, -1 for v->u, 0 for both (counted as two edges)

    plt.figure(figsize=(7, 5))
    _plot_settings()

    # draw edges with directions
    for (u, v), direction in zip(edges, directions):
        color = "red" if direction == 0 else "black"
        start, end = (u, v) if direction == 1 else (v, u)
        plt.arrow(
            x=vertices[start][0],
            y=vertices[start][1],
            dx=(vertices[end][0] - vertices[start][0]),
            dy=(vertices[end][1] - vertices[start][1]),
            head_width=0.2,
            length_includes_head=True,
            color=color,
        )
        if direction == 0:
            plt.arrow(
                x=vertices[end][0],
                y=vertices[end][1],
                dx=(vertices[start][0] - vertices[end][0]),
                dy=(vertices[start][1] - vertices[end][1]),
                head_width=0.2,
                length_includes_head=True,
                color=color,
            )

    # draw vertices
    plt.scatter(vertices[:, 0], vertices[:, 1], color="blue", zorder=2)

    # i used genai for this legend
    plt.legend(
        handles=[
            plt.Line2D([0], [0], color="black", lw=2, label="one edge"),
            plt.Line2D(
                [0], [0], color="red", lw=2, label="two edges\nopposite directions"
            ),
        ],
        loc="upper right",
    )
    plt.title(f"Random directed graph\nwith {n_vertices} vertices and {n_edges} edges")
    plt.show()


if __name__ == "__main__":
    draw_directed_graph(5, 5)
