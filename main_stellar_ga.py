"""
Main script for stellar-constrained GA transit detection
"""

import os
import sys
import argparse
from datetime import datetime
from data_loader import load_data, get_known_period
from ga_algorithms import run_ga
from transit_models import expected_duration
from plotting import plot_results


def main():
    """Run GA with stellar constraints and literature period"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='GA-based transit detection')
    parser.add_argument('--star', type=str, default='HAT-P-7',
                       help='Star name (HAT-P-7, WASP-12, etc.)')
    parser.add_argument('--model', type=str, default='batman', 
                       choices=['batman', 'pytransit'],
                       help='Transit model to use (batman or pytransit)')
    parser.add_argument('--fix-u2', action='store_true',
                       help='Fix u2 limb darkening at theoretical value (default: evolve u2)')
    args = parser.parse_args()
    
    # Load data and stellar parameters
    star_name = args.star
    time, flux, stellar = load_data(star_name)
    
    # Create runs folder with star name and run number
    runs_base = 'runs'
    os.makedirs(runs_base, exist_ok=True)
    
    # Find next run number for this star+model combination
    star_slug = star_name.replace(' ', '_').replace('-', '_')
    run_prefix = f'{star_slug}_{args.model}'
    existing_runs = [d for d in os.listdir(runs_base) if d.startswith(f'{run_prefix}_')]
    run_number = len(existing_runs) + 1
    
    run_dir = os.path.join(runs_base, f'{run_prefix}_{run_number}')
    os.makedirs(run_dir, exist_ok=True)
    
    # Set up logging to both console and file
    log_file = os.path.join(run_dir, 'output.log')
    
    class Logger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, 'w', encoding='utf-8')
        
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()
        
        def flush(self):
            self.terminal.flush()
            self.log.flush()
    
    sys.stdout = Logger(log_file)
    
    print("="*60)
    print("STELLAR-CONSTRAINED GA DETECTION")
    print(f"Star: {star_name} | Run #{run_number}")
    print(f"Transit Model: {args.model}")
    print("="*60)
    
    # Get known period (set USE_HARDCODED_PERIOD=True to skip archive query)
    planet_name = star_name + ' b'
    known_period = get_known_period(planet_name, use_hardcoded=True)
    print(f"\nUsing period: {known_period} days")
    
    # Run GA optimization
    print("\n" + "="*60)
    print("GA OPTIMIZATION")
    print("="*60)
    
    # Determine whether to evolve u2 (default: True, unless --fix-u2 flag is set)
    evolve_u2 = not args.fix_u2
    if not evolve_u2:
        print(f"*** u2 FIXED at theoretical value: {stellar.u2:.4f} ***\n")
    
    # Use default parameters from ga_algorithms.py
    best, fitness_history = run_ga(time, flux, stellar, known_period, model=args.model, evolve_u2=evolve_u2)
    
    # Calculate final metrics
    duration = expected_duration(best.P, stellar, best.impact)
    
    # Results
    print("\n" + "="*60)
    print("FINAL RESULT")
    print("="*60)
    print(f"Period:    {best.P:.6f} days")
    print(f"T0:        {best.T0:.3f} BJD-2454833")
    print(f"Rp/Rs:     {best.RpRs:.4f}")
    print(f"Depth:     {best.depth*100:.4f}% ({best.depth*1e6:.0f} ppm)")
    print(f"Impact:    {best.impact:.3f}")
    print(f"Duration:  {duration*24:.2f} hours (from stellar params)")
    print(f"u1:        {stellar.u1:.4f} (FIXED from Claret)")
    print(f"u2:        {best.u2:.4f} (Claret: {stellar.u2:.4f})")
    # print(f"ecc:       {best.ecc:.4f}")
    # print(f"omega:     {best.omega:.1f} degrees")
    print("="*60)
    
    # Plot results
    plot_file = os.path.join(run_dir, 'results.png')
    plot_results(best, time, flux, stellar, fitness_history, star_name=star_name, save_path=plot_file)
    print(f"\nResults saved to: {run_dir}")
    print(f"  - Plot: {plot_file}")
    print(f"  - Log: {log_file}")


if __name__ == "__main__":
    main()
