import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.io import mmread


def total_energy(positions, edges, q=1, k_spring=0.1, L0=1.0):
    n = len(positions)
    energy = 0.0

    # vertices
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(positions[i] - positions[j]) + 1e-9
            energy += q**2 / r

    # edges
    for i, j in edges:
        diff = positions[i] - positions[j]
        dist = np.linalg.norm(diff)
        energy += 0.5 * k_spring * (dist - L0) ** 2

    return energy


def annealing_step(positions, edges, T, q=1, k_spring=0.1, L0=1.0, step_size=0.3):
    n = len(positions)
    i = np.random.randint(n)

    old_pos = positions[i].copy()
    old_energy = total_energy(positions, edges, q, k_spring, L0)

    # random move
    positions[i] += np.random.randn(2) * step_size

    new_energy = total_energy(positions, edges, q, k_spring, L0)
    dE = new_energy - old_energy

    # accept/reject move
    if dE > 0 and np.random.rand() >= np.exp(-dE / T):
        positions[i] = old_pos  # reject move

    return positions, dE


def animate_network_layout(G, steps=20000, T_start=1.0, T_end=0.0001):
    edges = list(G.edges)
    n = len(G.nodes)
    positions = np.random.rand(n, 2) * 5

    fig, ax = plt.subplots(figsize=(6, 6))
    scat = ax.scatter(positions[:, 0], positions[:, 1], s=80, c="skyblue", zorder=3)
    scat.set_visible(False)
    lines = [ax.plot([], [], "-", lw=1, color="lightgray")[0] for _ in edges]

    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.axis("off")
    title = ax.set_title("")

    # cooling schedule
    T = T_start
    cooling_rate = (T_end / T_start) ** (1 / (steps))

    def update(frame):
        nonlocal positions, T

        scat.set_visible(True)

        if frame * 100 >= steps:
            print("Annealing complete!")
            ani.event_source.stop()
            return

        for _ in range(100):
            positions, dE = annealing_step(positions, edges, T)
            if -1e-5 <= dE < 0:
                print("Converged!")
                ani.event_source.stop()
                break
            T *= cooling_rate

        # update visuals
        scat.set_offsets(positions)
        for line, (i, j) in zip(lines, edges):
            line.set_data(
                [positions[i, 0], positions[j, 0]],
                [positions[i, 1], positions[j, 1]],
            )

        # dynamic axes
        x_min, y_min = positions.min(axis=0)
        x_max, y_max = positions.max(axis=0)
        pad = 0.5
        ax.set_xlim(x_min - pad, x_max + pad)
        ax.set_ylim(y_min - pad, y_max + pad)

        title.set_text(f"Simulated annealing step={frame * 100} T={T:.4f}")

        return scat, *lines

    ani = FuncAnimation(fig, update, frames=steps // 100, interval=30, blit=False)
    ani.save("network_layout.gif", writer=PillowWriter(fps=5))


def read_network(file_path):
    matrix = mmread(file_path)
    G = nx.from_scipy_sparse_array(matrix)
    return G


if __name__ == "__main__":
    G = read_network("dolphins/dolphins.mtx")
    animate_network_layout(G)
