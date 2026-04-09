import numpy as np
from itertools import product
from scipy import sparse
from qiskit.quantum_info import SparsePauliOp
try:
    import pymatching
    _HAVE_PYMATCHING = True
except Exception:
    _HAVE_PYMATCHING = False

def pauli_list_to_sparseop(pauli_list):
    n = len(pauli_list)
    s = ''.join(reversed(pauli_list))
    return SparsePauliOp.from_list([(s, 1.0)])

def pauli_list_to_matrix(pauli_list):
    return pauli_list_to_sparseop(pauli_list).to_matrix()

def pauli_list_weight(pauli_list):
    return sum(1 for p in pauli_list if p != 'I')

def binary_pair_to_pauli_list(eX, eZ):
    p = []
    for x,z in zip(eX, eZ):
        if x==0 and z==0:
            p.append('I')
        elif x==1 and z==0:
            p.append('X')
        elif x==0 and z==1:
            p.append('Z')
        elif x==1 and z==1:
            p.append('Y')
    return p

def pauli_list_to_binary_pair(plist):
    eX = []
    eZ = []
    for p in plist:
        if p == 'I':
            eX.append(0); eZ.append(0)
        elif p == 'X':
            eX.append(1); eZ.append(0)
        elif p == 'Z':
            eX.append(0); eZ.append(1)
        elif p == 'Y':
            eX.append(1); eZ.append(1)
        else:
            raise ValueError("unknown pauli "+p)
    return np.array(eX, dtype=int), np.array(eZ, dtype=int)

'''def syndrome_from_eX(eX, HZ):
    return (HZ.dot(eX) % 2).astype(int)

def syndrome_from_eZ(eZ, HX):
    return (HX.dot(eZ) % 2).astype(int)'''

def mwpm_initialize_e_given_syndrome(H, syndrome):
    m, n = H.shape
    if _HAVE_PYMATCHING:
        Ms = sparse.csr_matrix(H)
        M = pymatching.Matching(Ms)
        e = M.decode(syndrome.tolist())
        e = np.array(e, dtype=int)
        return e
    else:
        # fallback: simple gaussian elimination mod 2 to find a particular solution
        e= ge_initialize_given_syndrome(H, syndrome)
        return e
    
def ge_initialize_given_syndrome(H, syndrome):
    m, n = H.shape
    # Solve H x = s over GF(2).
    A = np.concatenate([H.copy() % 2, syndrome.reshape(-1,1)], axis=1).astype(int)
    # row reduce (m x (n+1))
    r = 0
    pivots = []
    for c in range(n):
        # find row with 1 in column c at or below row r
        for i in range(r, m):
            if A[i, c] == 1:
                A[[r, i]] = A[[i, r]]
                break
        else:
            continue
        pivots.append(c)
        # eliminate other rows
        for i in range(m):
            if i != r and A[i, c] == 1:
                A[i, :] ^= A[r, :]
        r += 1
        if r == m:
            break
    # now set free variables to 0 and back-substitute to get x
    x = np.zeros(n, dtype=int)
    # for each pivot row, find pivot column
    for i_row in range(min(m, len(pivots))):
        c = pivots[i_row]
        x[c] = A[i_row, -1]  # RHS
    return x

def coset_weight_enum(eX, eZ, code):
    n = code.n
    # Toric codes have one redundant X and Z stabilizer; Planar codes do not.
    is_toric = code.__class__.__name__ == 'ToricCode'
    X_stabs = code.X_stabilizers[:-1] if is_toric else code.X_stabilizers
    Z_stabs = code.Z_stabilizers[:-1] if is_toric else code.Z_stabilizers
    
    # Convert index-based stabilizers to binary vectors
    X_vecs = []
    for stab in X_stabs:
        v = np.zeros(n, dtype=int)
        v[stab] = 1
        X_vecs.append(v)

    Z_vecs = []
    for stab in Z_stabs:
        v = np.zeros(n, dtype=int)
        v[stab] = 1
        Z_vecs.append(v)

    A = np.zeros(n + 1, dtype=np.int64)
    eX_arr, eZ_arr = np.array(eX), np.array(eZ)

    # Iterate through all possible stabilizer combinations
    for xb in product([0, 1], repeat=len(X_vecs)):
        xs = np.zeros(n, dtype=int)
        for b, v in zip(xb, X_vecs):
            if b: xs ^= v
            
        for zb in product([0, 1], repeat=len(Z_vecs)):
            zs = np.zeros(n, dtype=int)
            for b, v in zip(zb, Z_vecs):
                if b: zs ^= v

            # Compute weight using the existing utility function
            w = np.sum(((eX_arr ^ xs) | (eZ_arr ^ zs)))#weight(eX_arr ^ xs, eZ_arr ^ zs)
            A[w] += 1

    return A

def coset_weight_distr(eX, eZ, code, p):
    n = code.n
    A = coset_weight_enum(eX, eZ, code)
    P_coset = 0
    for w, count in enumerate(A):
        P_coset += count * ((p/3)**w) * ((1-p)**(n-w)) 
    return P_coset

def generate_all_sectors(eX, eZ, code):
    """
    Generates error configurations for all logical sectors by applying all 
    combinations of logical X and Z operators to the initial error configuration.
    """
    n = code.n
    log_X_supports = [s for s in [code.logical_X_support(), code.logical_X_conjugate()] if s]
    log_Z_supports = [s for s in [code.logical_Z_support(), code.logical_Z_conjugate()] if s]

    num_logical_qubits = len(log_X_supports)
    if num_logical_qubits != len(log_Z_supports):
        raise ValueError("Inconsistent number of logical X and Z operators.")

    log_X_op_vecs = []
    for support in log_X_supports:
        vec = np.zeros(n, dtype=int)
        vec[support] = 1
        log_X_op_vecs.append(vec)

    log_Z_op_vecs = []
    for support in log_Z_supports:
        vec = np.zeros(n, dtype=int)
        vec[support] = 1
        log_Z_op_vecs.append(vec)

    lX_combinations = []
    for b_bits in product([0, 1], repeat=num_logical_qubits):
        lX = np.zeros(n, dtype=int)
        for i, b in enumerate(b_bits):
            if b: lX ^= log_X_op_vecs[i]
        lX_combinations.append(lX)

    lZ_combinations = []
    for c_bits in product([0, 1], repeat=num_logical_qubits):
        lZ = np.zeros(n, dtype=int)
        for i, c in enumerate(c_bits):
            if c: lZ ^= log_Z_op_vecs[i]
        lZ_combinations.append(lZ)

    eX_arr, eZ_arr = np.array(eX, dtype=int), np.array(eZ, dtype=int)
    return [(eX_arr ^ lX, eZ_arr ^ lZ) for lZ in lZ_combinations for lX in lX_combinations]
