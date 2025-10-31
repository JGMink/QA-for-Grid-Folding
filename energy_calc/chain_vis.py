import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_configuration(chain, bitstring, adj, filename="visualization.png",
                            show_edges=True, show_chain=True, layout="grid"):
    """
    Visualize a protein folding configuration on a lattice graph and save as an image.

    Parameters
    ----------
    chain : list of str
        Amino acid sequence, e.g., ['H', 'C', 'H', 'P'].
    bitstring : str
        Binary string representing amino acid positions, length = N*N for N acids and N sites.
    adj : np.ndarray
        NxN adjacency matrix for the lattice (1 = adjacent).
    filename : str
        Output file name for saved image.
    show_edges : bool
        Whether to show lattice adjacency edges.
    show_chain : bool
        Whether to connect acids in chain order.
    layout : {'grid', 'spring'}
        Layout mode for positioning sites visually.
    """
    N = len(chain)

    # --- Validate bitstring ---
    if len(bitstring) != N*N:
        raise ValueError(f"Bitstring length {len(bitstring)} does not match N^2 = {N*N}")

    # Convert bitstring to NxN matrix
    b = np.array([list(map(int, bitstring[i*N:(i+1)*N])) for i in range(N)])

    # --- Create lattice graph ---
    G = nx.Graph()
    for i in range(N):
        for j in range(N):
            if adj[i, j] == 1:
                G.add_edge(i, j)

    # --- Layout positions ---
    if layout == "grid":
        side = int(np.ceil(np.sqrt(N)))
        pos = {i: (i % side, -(i // side)) for i in range(N)}
    else:
        pos = nx.spring_layout(G, seed=42)

    # --- Map site occupancy ---
    site_to_acid = {}
    for acid_idx in range(N):
        for site_idx in range(N):
            if b[acid_idx, site_idx] == 1:
                site_to_acid[site_idx] = chain[acid_idx]

    # --- Node colors ---
    color_map = {'H': 'red', 'C': 'green', 'P': 'blue'}
    node_colors = [color_map.get(site_to_acid.get(i, 'P'), 'gray') for i in range(N)]

    # --- Plot graph ---
    plt.figure(figsize=(5, 5))
    if show_edges:
        nx.draw_networkx_edges(G, pos, alpha=0.4)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=700)
    nx.draw_networkx_labels(G, pos,
                            labels={i: f"{i+1}\n{site_to_acid.get(i, '')}"},
                            font_color='white')

    # --- Draw chain connections ---
    if show_chain:
        acid_positions = []
        for acid_idx in range(N):
            for site_idx in range(N):
                if b[acid_idx, site_idx] == 1:
                    pos_val = pos.get(site_idx)
                    if pos_val is not None:
                        acid_positions.append(np.array(pos_val, dtype=float))

        # Draw edges only if all positions are valid
        if len(acid_positions) == N:
            for p1, p2 in zip(acid_positions[:-1], acid_positions[1:]):
                plt.plot([p1[0], p2[0]], [p1[1], p2[1]],
                         color='black', linestyle='--', alpha=0.7)

    # --- Final touches ---
    plt.title("Protein Configuration Visualization")
    plt.axis('off')

    # Save to PNG
    output_path = os.path.join(os.getcwd(), filename)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✅ Visualization saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    chain = ['H', 'C', 'H', 'P']
    bitstring = '0100100000011000'
    adj = np.array([
        [0, 1, 1, 0],
        [1, 0, 0, 1],
        [1, 0, 0, 1],
        [0, 1, 1, 0]
    ])

    visualize_configuration(chain, bitstring, adj)
