import numpy as np

# === Lagrange Penalties (Globals) ===
L1, L2, L3 = 1.0, 1.0, 1.0

# === Simplified MJ Interaction Matrix Example ===
C = {
    ('H', 'H'): 1, ('C', 'C'): -1,
    ('H', 'C'): 0, ('C', 'H'): 0,
    ('H', 'P'): 0, ('P', 'H'): 0,
    ('C', 'P'): 0, ('P', 'C'): 0,
    ('P', 'P'): 0
}

# === Energy Function Definitions ===
def compute_E_MJ(chain, b, adj, C):
    N = len(chain)
    E = 0
    for i in range(N):
        for j in range(i+1, N):
            if abs(i - j) > 1:
                a_i, a_j = chain[i], chain[j]
                for n in range(N):
                    for m in range(N):
                        E += C.get((a_i, a_j), 0) * b[i, n] * b[j, m] * adj[n, m]
    return -E

def compute_E1(b):
    # Ensures each acid occupies exactly one site
    return np.sum((np.sum(b, axis=1) - 1) ** 2)

def compute_E2(b):
    # Ensures each site has exactly one acid
    N = b.shape[0]
    E2 = 0
    for n in range(N):
        for i in range(N):
            for j in range(N):
                if i != j:
                    E2 += b[i, n] * b[j, n]
    return int(0.5 * E2) # to correct double counting

def compute_E3(b, adj):
    # Enforces that consecutive acids occupy adjacent sites
    N = b.shape[0]
    E3 = 0
    for i in range(N - 1):
        for n in range(N):
            for m in range(N):
                if adj[n, m] == 0:  # non-adjacent sites
                    E3 += b[i, n] * b[i + 1, m]
    return E3

# === Total Energy Function ===
def total_energy(chain, bitstring, adj, C, L1=1.0, L2=1.0, L3=1.0):
    N = len(chain)
    b = np.array([list(map(int, bitstring[i*N:(i+1)*N])) for i in range(N)])
    E_MJ = compute_E_MJ(chain, b, adj, C)
    E_1  = compute_E1(b)
    E_2  = compute_E2(b)
    E_3  = compute_E3(b, adj)
    return E_MJ + L1*E_1 + L2*E_2 + L3*E_3, (E_MJ, E_1, E_2, E_3)

if __name__ == "__main__":
    # Example usage
    chain = ['H', 'P', 'C', 'H']
    bitstring = '0010100001000001'  # Example bitstring for 4 acids and 4 sites
    adj = np.array([[0, 1, 1, 0],
                    [1, 0, 0, 1],
                    [1, 0, 0, 1],
                    [0, 1, 1, 0]])  # Example adjacency matrix

    total_E, breakdown = total_energy(chain, bitstring, adj, C, L1, L2, L3)
    print(f"Total Energy: {total_E}")
    print(f"Energy Breakdown: MJ={breakdown[0]}, E1={breakdown[1]}, E2={breakdown[2]}, E3={breakdown[3]}")