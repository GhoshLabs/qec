from utils import coset_weight_distr, generate_all_sectors, ge_initialize_given_syndrome
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from MH_sampler import metropolis_hastings_coset_probs
from noise import depolarizing_noise
import syndrome as synd

def coset_probs_exact(eX, eZ, code, p):
    """Calculates exact probabilities and returns (probs, labels)."""
    all_sectors = generate_all_sectors(eX, eZ, code)
    num_logical_qubits = int(np.round(np.log2(len(all_sectors)) / 2))
    
    labels = []
    for c_bits in product([0, 1], repeat=num_logical_qubits):
        for b_bits in product([0, 1], repeat=num_logical_qubits):
            pauli_str = ""
            for i in range(num_logical_qubits):
                b, c = b_bits[i], c_bits[i]
                mapping = {(0,0):'I', (1,0):'X', (0,1):'Z', (1,1):'Y'}
                pauli_str += mapping[(b, c)]
            labels.append(pauli_str)
            
    P = []
    for lX, lZ in all_sectors:
        P.append(coset_weight_distr(lX, lZ, code, p))
    return P, labels

def coset_probs_mcmc(eX, eZ, code, p, n_samples=20000, burn_in=5000):
    """
    Estimates the probability of all logical cosets using MCMC tracking.
    Returns (coset_probs, min_weight_error_probs).
    """
    # 1. Setup sampler parameters
    q = p / (3 - 2 * p) # Conversion for depolarizing noise
    HZ, HX = code.stabilizer_matrices()
    Zstab_vecs = [HZ[i] for i in range(HZ.shape[0])]
    Xstab_vecs = [HX[i] for i in range(HX.shape[0])]
    all_stabs = Xstab_vecs + Zstab_vecs
    n_X_stabs = len(Xstab_vecs)

    # 2. Generate logical operators for tracking
    n = code.n
    log_X_supports = [s for s in [code.logical_X_support(), code.logical_X_conjugate()] if s]
    log_Z_supports = [s for s in [code.logical_Z_support(), code.logical_Z_conjugate()] if s]
    num_logical_qubits = len(log_X_supports)

    # 2.1 Generate logical operators in the same order as generate_all_sectors
    lX_vecs = []
    for support in log_X_supports:
        v = np.zeros(n, dtype=int); v[support] = 1
        lX_vecs.append(v)
    lZ_vecs = []
    for support in log_Z_supports:
        v = np.zeros(n, dtype=int); v[support] = 1
        lZ_vecs.append(v)

    logicals_X, logicals_Z = [], []
    labels = []
    for c_bits in product([0, 1], repeat=num_logical_qubits):
        for b_bits in product([0, 1], repeat=num_logical_qubits):
            lX, lZ = np.zeros(n, dtype=int), np.zeros(n, dtype=int)
            pauli_str = ""
            for i, b in enumerate(b_bits):
                if b: lX ^= lX_vecs[i]
            for i, c in enumerate(c_bits):
                if c: lZ ^= lZ_vecs[i]
                mapping = {(0,0):'I', (1,0):'X', (0,1):'Z', (1,1):'Y'}
                pauli_str += mapping[(b_bits[i], c_bits[i])]
            logicals_X.append(lX)
            logicals_Z.append(lZ)
            labels.append(pauli_str)

    # 3. Run the sampler (using independent parallel chains for each class)
    Z_ratios, min_weights = metropolis_hastings_coset_probs(
        np.array(eX, dtype=int), np.array(eZ, dtype=int), all_stabs, n_X_stabs, 
        q, int(n_samples), int(burn_in), logicals_X, logicals_Z
    )

    # 4. Calculate individual probabilities for the minimum weight errors found
    p_min_weights = [(p/3)**w * (1-p)**(n-w) for w in min_weights]

    # Normalize Z ratios to get relative coset probabilities
    return Z_ratios / np.sum(Z_ratios), p_min_weights / np.sum(p_min_weights), labels

def bar_graph(exact_probs, mcmc_probs, min_weight_probs, labels=None, title="Coset Probabilities Comparison"):
    """
    Plots a bar graph comparing exact coset probabilities, MCMC estimated 
    probabilities, and the probability of the minimum weight error in each coset.
    """
    num_cosets = len(exact_probs)
    indices = np.arange(num_cosets)
    width = 0.25

    plt.figure(figsize=(12, 7))
    plt.bar(indices - width, exact_probs, width, label='Exact Coset Prob', color='skyblue', alpha=0.8)
    plt.bar(indices, mcmc_probs, width, label='MCMC Coset Prob', color='orange', alpha=0.8)
    plt.bar(indices + width, min_weight_probs, width, label='Min Weight Error Prob', color='green', alpha=0.8)

    plt.xlabel('Logical Coset Index')
    plt.ylabel('Probability')
    plt.title(title)
    if labels:
        plt.xticks(indices, labels, rotation=45)
    else:
        plt.xticks(indices)
    plt.yscale('log')
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('coset_probs_comparison_p08.pdf')
    plt.close()

def get_all_syndromes(code):
    def _gen_single_type(n_stabs):
        for bits in product([0, 1], repeat=n_stabs):
            yield np.array(bits, dtype=int)

    syndZs = list(_gen_single_type(len(code.Z_stabilizers)))
    syndXs = list(_gen_single_type(len(code.X_stabilizers)))
    
    return [(sz, sx) for sz in syndZs for sx in syndXs]

def syndrome_probs(code, p):  
    HZ, HX = code.stabilizer_matrices()
    all_syndromes = get_all_syndromes(code)
    
    probs_dict = {}
    total_sum = 0.0
    
    for sz, sx in all_syndromes:
        # Find representative error vectors for the given syndrome bitstrings
        eX = ge_initialize_given_syndrome(HZ, sz)
        eZ = ge_initialize_given_syndrome(HX, sx)
        
        # Sum probabilities across all logical cosets to get total syndrome probability
        probs, _ = coset_probs_exact(eX, eZ, code, p)
        synd_prob = sum(probs)
        
        # Store using hashable tuple keys
        key = (tuple(sz), tuple(sx))
        probs_dict[key] = synd_prob
        total_sum += synd_prob
        
    # Normalize probabilities so they sum to 1
    if total_sum > 0:
        for key in probs_dict:
            probs_dict[key] /= total_sum
            
    return probs_dict

def bar_graph_syndrome_avg(code, p, n_synd_samples=1000):
    HZ, HX = code.stabilizer_matrices()
    #probs_dict = syndrome_probs(code, p)
    n = code.n
    syndrome_counts = {}
    for _ in range(n_synd_samples):
        ex, ez = depolarizing_noise(n, p)
        sZ = synd.syndrome_from_eX(ex, code.Z_stabilizers)
        sX = synd.syndrome_from_eZ(ez, code.X_stabilizers)
        key = (tuple(sZ), tuple(sX))
        if key not in syndrome_counts:
            syndrome_counts[key] = 0
        syndrome_counts[key] += 1
    
    avg_probs = None
    avg_mcmc_probs = None
    avg_min_weight_probs = None
    labels = None

    total = sum(syndrome_counts.values())
    
    for (sz_tuple, sx_tuple), count in syndrome_counts.items():
        '''if p_syndrome == 0:
            continue
            
        sz = np.array(sz_tuple)
        sx = np.array(sx_tuple)
        
        # Find representative error configuration for this syndrome
        eX = ge_initialize_given_syndrome(HZ, sz)
        eZ = ge_initialize_given_syndrome(HX, sx)'''

        w = count / total  # empirical syndrome weight
        sz = np.array(sz_tuple)
        sx = np.array(sx_tuple)

        # Use a valid representative error for this syndrome
        eX = ge_initialize_given_syndrome(HZ, sz)
        eZ = ge_initialize_given_syndrome(HX, sx)
        
        # Get exact coset probabilities: P(L_i and S)
        probs, current_labels = coset_probs_exact(eX, eZ, code, p)
        
        # Get MCMC estimates and min weight probs for this syndrome
        mcmc_probs, min_weight_probs, _ = coset_probs_mcmc(eX, eZ, code, p)
        
        # Calculate P(L_i | S) = P(L_i and S) / P(S)
        s_sum = sum(probs)
        if s_sum > 0:
            cond_probs = np.array(probs) / s_sum
            cond_mcmc_probs = np.array(mcmc_probs)
            cond_min_weight_probs = np.array(min_weight_probs) / s_sum
            
            if avg_probs is None:
                avg_probs = np.zeros(len(probs))
                avg_mcmc_probs = np.zeros(len(probs))
                avg_min_weight_probs = np.zeros(len(probs))
                labels = current_labels
            
            '''# Accumulate the weighted contribution: P(S) * P(L | S)
            avg_probs += p_syndrome * cond_probs
            avg_mcmc_probs += p_syndrome * cond_mcmc_probs
            avg_min_weight_probs += p_syndrome * cond_min_weight_probs'''
            # Accumulate weighted contribution using empirical syndrome frequency
            avg_probs += w * cond_probs
            avg_mcmc_probs += w * cond_mcmc_probs
            avg_min_weight_probs += w * cond_min_weight_probs
    
    if avg_probs is None:
        return

    num_cosets = len(avg_probs)
    indices = np.arange(num_cosets)
    width = 0.25

    plt.figure(figsize=(12, 7))
    plt.bar(indices - width, avg_probs, width, label='Exact Coset Prob', color='skyblue', alpha=0.8)
    plt.bar(indices, avg_mcmc_probs, width, label='MCMC Coset Prob', color='orange', alpha=0.8)
    plt.bar(indices + width, avg_min_weight_probs, width, label='Min Weight Error Prob', color='green', alpha=0.8)

    plt.xlabel('Logical Coset')
    plt.ylabel('Expected Probability')
    plt.title(f'Syndrome-Averaged Logical Coset Probabilities (L={code.L}, p={p})')
    if labels:
        plt.xticks(indices, labels, rotation=45)
    plt.yscale('log')
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f'syndrome_avg_coset_probs_L{code.L}_p{p}.pdf')
    plt.close()