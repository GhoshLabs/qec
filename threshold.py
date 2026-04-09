import numpy as np
import matplotlib.pyplot as plt
from code import ToricCode, PlanarSurfaceCode
from simulation import run_trial
import os
import pandas as pd
import csv
from decoder import MHDecoderSingleChain, MWPMDecoder, MHDecoderParallel, BPDecoder

def logical_error_rate(code, p, decoder, n_trials=1000):
    failures = 0
    for _ in range(n_trials):
        failures += run_trial(code, p, decoder)
    return failures / n_trials

def P_vs_L_plot(L_list, p_list, decoder_factory, trials=2000, code_type='Toric'):
    results = experiment(L_list, p_list, decoder_factory, trials, code_type)

    for p in p_list:
        rates = results[p]
        #plt.plot(L_list, rates, marker='s', label=f"p={p}")

    # Save data to CSV
    with open(f'p_vs_l_data_{code_type}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['p'] + [f'L={L}' for L in L_list])
        for p in p_list:
            rates = results[p]
            writer.writerow([p] + rates)

    '''plt.xlabel("Lattice Size L")
    plt.ylabel("Logical error rate")
    plt.legend()
    plt.yscale("log")
    plt.grid(True, which="both", ls="--")
    plt.show()'''

def threshold_plot(L_list, p_list, decoder_factory, trials=2000, code_type='Toric'):
    """Plots the logical error rate P vs physical error rate p for all L."""
    results = experiment(L_list, p_list, decoder_factory, trials, code_type)

    for i, L in enumerate(L_list):
        rates = [results[p][i] for p in p_list]
        #plt.plot(p_list, rates, marker='o', label=f"L={L}")

    # Save data to CSV
    with open(f'threshold_data_{code_type}_mh.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['p'] + [f'L={L}' for L in L_list])  # header row
        for i, p in enumerate(p_list):
            rates = [results[p][j] for j in range(len(L_list))]
            writer.writerow([p] + rates)

    '''plt.xlabel("Physical error rate p")
    plt.ylabel("Logical error rate")
    plt.legend()
    plt.yscale("log")
    plt.grid(True, which="both", ls="--")
    plt.savefig(f'threshold_plot_{code_type}_MH.pdf')''' 

def experiment(L_list, p_list, decoder_factory, trials=2000, code_type='Toric'):
    results = {} # rates for every L and p

    for p in p_list:
        rates = []
        for L in L_list:
            if code_type == 'Toric':
                code = ToricCode(L)
            elif code_type == 'Planar':
                code = PlanarSurfaceCode(L)
            else:
                raise ValueError(f"Unknown code_type: {code_type}")
            decoder = decoder_factory(code, p)
            rate = logical_error_rate(code, p, decoder, trials)
            rates.append(rate)

        results[p] = rates

    return results

def comparison_plot(p_list, trials=2000, L=8, code_type='Toric'):
    if code_type == 'Toric':
        code = ToricCode(L)
    elif code_type == 'Planar':
        code = PlanarSurfaceCode(L)
    mh_rates = []
    mwpm_rates = []
    bp_rates = []

    for p in p_list:
        mh_decoder = MHDecoderParallel(code, q_error=p/(3-2*p), n_samples=L**4)
        mwpm_decoder = MWPMDecoder(code)
        bp_decoder = BPDecoder(code, p)

        mh_rate = logical_error_rate(code, p, mh_decoder, trials)
        mwpm_rate = logical_error_rate(code, p, mwpm_decoder, trials)
        bp_rate = logical_error_rate(code, p, bp_decoder, trials)

        mh_rates.append(mh_rate)
        mwpm_rates.append(mwpm_rate)
        bp_rates.append(bp_rate)

    # Save data to CSV
    with open(f'comparison_data_L{L}_{code_type}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['p', 'MH_rate', 'MWPM_rate', 'BP_rate'])
        for i, p in enumerate(p_list):
            writer.writerow([p, mh_rates[i], mwpm_rates[i], bp_rates[i]])

    plt.plot(p_list, mh_rates, marker='o', label="MH Decoder")
    plt.plot(p_list, mwpm_rates, marker='s', label="MWPM Decoder")
    plt.plot(p_list, bp_rates, marker='^', label="BP Decoder")
    plt.xlabel("Physical error rate p")
    plt.ylabel("Logical error rate")
    plt.legend()
    plt.yscale("log")
    plt.grid(True, which="both", ls="--")
    plt.savefig(f'comparison_plot_L{L}_{code_type}.pdf')                                                                                                                
    #plt.show()