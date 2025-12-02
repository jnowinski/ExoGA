"""
Plotting functions for transit analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from transit_models import limb_darkened_transit, expected_duration


def plot_results(best, time, flux, stellar, fitness_history, save_path='grid_ga_stellar_fit.png'):
    """Generate fit plots with convergence graph"""
    
    # Generate best-fit model with limb darkening
    model = limb_darkened_transit(time, best.T0, best.RpRs, best.P, best.impact, stellar)
    residuals = flux - model
    
    # Phase fold
    phase = ((time - best.T0) % best.P) / best.P
    phase[phase > 0.5] -= 1
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Convergence graph
    ax = axes[0]
    ax.plot(range(1, len(fitness_history) + 1), fitness_history, 'b-', linewidth=2)
    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Best Fitness (χ²)', fontsize=12)
    ax.set_title('GA Convergence', fontsize=14)
    ax.set_yscale('log')
    ax.grid(alpha=0.3)
    
    # Plot 2: Phase-folded lightcurve
    ax = axes[1]
    
    # Bin data
    sort_idx = np.argsort(phase)
    phase_sorted = phase[sort_idx]
    flux_sorted = flux[sort_idx]
    
    bin_size = 100
    binned_phase = []
    binned_flux = []
    binned_err = []
    
    for i in range(0, len(phase_sorted), bin_size):
        chunk_phase = phase_sorted[i:i+bin_size]
        chunk_flux = flux_sorted[i:i+bin_size]
        if len(chunk_phase) > 0:
            binned_phase.append(np.mean(chunk_phase))
            binned_flux.append(np.mean(chunk_flux))
            binned_err.append(np.std(chunk_flux) / np.sqrt(len(chunk_flux)))
    
    # Plot binned data
    ax.errorbar(binned_phase, binned_flux, yerr=np.array(binned_err), 
                fmt='o', color='cornflowerblue', alpha=0.6, markersize=3,
                label='Binned data')
    
    # Plot model
    phase_model = np.linspace(-0.5, 0.5, 1000)
    time_model = phase_model * best.P + best.T0
    flux_model = limb_darkened_transit(time_model, best.T0, best.RpRs, best.P, best.impact, stellar)
    ax.plot(phase_model, flux_model, 'r-', linewidth=2, label='Limb-darkened model')
    
    duration = expected_duration(best.P, stellar, best.impact)
    ax.set_xlabel('Phase', fontsize=12)
    ax.set_ylabel('Normalized Flux', fontsize=12)
    ax.set_title(f'HAT-P-7b Transit (P={best.P:.6f}d, Rp/Rs={best.RpRs:.4f}, depth={best.depth*100:.4f}%, '
                f'impact={best.impact:.2f}, dur={duration*24:.2f}hr)', fontsize=13)
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.15, 0.15)
    
    plt.tight_layout(h_pad=3.0)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {save_path}")
    plt.close()
