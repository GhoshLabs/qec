from code import ToricCode
from noise import depolarizing_noise
from syndrome import syndrome_from_eX, syndrome_from_eZ
from decoder import MWPMDecoder, MHDecoder, GEDecoder
from logical import logical_parity
import numpy as np
import matplotlib.pyplot as plt
from simulation import run_trial
from mh_diagnostics import plot_mh_traces, error_rate_vs_n_sample
from threshold import comparison_plot, threshold_plot
from plot_lattice import LatticePlotter

def run_single_experiment(L=5, p=0.05, decoder_type="MWPM", init_method='MWPM'):
    code = ToricCode(L)
    
    # --- Pauli-frame noise ---
    eX, eZ = depolarizing_noise(code.n, p)
    
    # --- Syndrome extraction ---
    syndZ = syndrome_from_eX(eX, code.Z_stabilizers)
    syndX = syndrome_from_eZ(eZ, code.X_stabilizers)

    # Choose decoder
    if decoder_type == "MWPM":
        decoder = MWPMDecoder(code)
        # --- Decode ---
        eX_hat, eZ_hat = decoder.decode(syndZ, syndX)
        plotter = LatticePlotter(code, [eX,eZ], syndromes=(syndX, syndZ))
        # Pass corrections to the plot method
        plotter.plot(corrections=(eX_hat, eZ_hat))
    elif decoder_type == "MH":
        q = 2*p/3
        decoder = MHDecoder(code, q_error=q)
        # --- Decode ---
        if init_method == "GE":
            eX_hat, eZ_hat = decoder.decode(syndZ, syndX, init_method='GE')
        else:
            eX_hat, eZ_hat = decoder.decode(syndZ, syndX)
        plotter = LatticePlotter(code, [eX,eZ], syndromes=(syndX, syndZ))
        # Pass corrections to the plot method
        plotter.plot(corrections=(eX_hat, eZ_hat))
    else:
        decoder = GEDecoder(code)
        # --- Decode ---
        eX_hat, eZ_hat = decoder.decode(syndZ, syndX)
        plotter = LatticePlotter(code, [eX,eZ], syndromes=(syndX, syndZ))
        # Pass corrections to the plot method
        plotter.plot(corrections=(eX_hat, eZ_hat))

    # --- Residual error ---
    rX = [a ^ b for a, b in zip(eX, eX_hat)]
    rZ = [a ^ b for a, b in zip(eZ, eZ_hat)]

    # --- Logical failure ---
    fail_X1 = logical_parity(rX, code.logical_Z_conjugate())
    fail_X2 = logical_parity(rX, code.logical_X_conjugate())
    fail_Z1 = logical_parity(rZ, code.logical_X_support())
    fail_Z2 = logical_parity(rZ, code.logical_Z_support())

    logical_failure = fail_X1 or fail_Z1 or fail_X2 or fail_Z2
    return logical_failure


if __name__ == "__main__":
    L=8
    p=0.17

    failed = run_single_experiment(L=L, p=p, decoder_type="MH", init_method='MWPM')
    print("Logical failure:", failed)
    
    code = ToricCode(L)
    plot_mh_traces(code, p=p, n_samples=4000, burn_in=1000)

    L_list = [4,6,8]                 # lattice sizes
    p_list = np.linspace(0.10, 0.20, 10)  # physical error rates
    trials = 2000                      # Monte Carlo trials per point

    '''threshold_plot(L_list, p_list, lambda c, p_val: MHDecoder(c, q_error=2*p_val/3), trials)'''

    comparison_plot(p_list, trials)

    '''rates = error_rate_vs_n_sample(code, p, MHDecoder(code, p), n_samples=70000)
    plt.plot(rates)
    plt.xlabel("Number of samples")
    plt.ylabel("Logical error rate")
    plt.show()'''