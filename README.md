# ExoGA: Genetic Algorithm for Exoplanet Transit Detection

Real-coded genetic algorithm for detecting and characterizing exoplanet transits with support for Transit Timing Variations (TTV), non-periodic transits, and complex noise patterns.

## Features

- **Boltzmann Selection**: Temperature-based selection that transitions from exploration to exploitation
- **BLX-α Crossover**: Blend crossover that preserves gene relationships
- **Non-uniform Mutation**: Adaptive mutation that decreases over generations
- **Student-t NLL Fitness**: Robust fitness function for noisy data
- **TTV Support**: Models transit timing variations from gravitational perturbations
- **Trapezoidal Transit Model**: Physically motivated transit shapes with ingress/egress

## Installation

```bash
pip install lightkurve matplotlib numpy scipy tqdm
```

## Project Structure

```
ExoGA/
├── exoga_ga.py          # Core GA components (genome, operators, fitness)
├── exoga_main.py        # Main GA loop and execution
├── ExoGA.py            # Data loading and visualization
└── README.md           # This file
```

## Quick Start

### 1. Test Transit Model

```python
python exoga_ga.py
```

This generates a test transit model to verify the trapezoidal transit function works correctly.

### 2. Load and Plot Kepler Data

```python
python ExoGA.py
```

Downloads Kepler-10 light curve data and plots:
- Full light curve (all quarters stitched)
- Zoomed view of first 30 days

### 3. Run Full GA Optimization

```python
python exoga_main.py
```

This will:
1. Load Kepler-10 PDCSAP data (~4 years)
2. Initialize population of 50 individuals
3. Evolve for 100 generations
4. Print discovered parameters
5. Generate convergence and fit plots
6. Save results to pickle file

## Genome Structure

### Per-Dataset Genome
- `F0`: Flux offset (normalization correction)
- `c1`: Linear baseline trend
- `c2`: Quadratic baseline trend
- `σ_jit`: White noise jitter parameter
- `D`: Third light dilution factor

### Per-Planet Genome
- `ln(P)`: Natural log of orbital period (days)
- `T0`: Mid-transit time (BJD - 2454833)
- `depth`: Transit depth (fractional)
- `duration`: Transit duration (days)
- `A_ttv`: TTV amplitude (days)
- `P_ttv`: TTV period (days)
- `φ_ttv`: TTV phase (radians)
- `Pd`: Period drift (days/year)

## Usage Examples

### Basic Usage

```python
from exoga_main import ExoGA

# Create GA
ga = ExoGA(
    pop_size=50,
    n_generations=100,
    n_planets=1
)

# Load data
ga.load_data('Kepler-10', use_pdcsap=True)

# Run optimization
ga.evolve()

# View results
ga.print_results()
ga.plot_results()
ga.save_results()
```

### Custom Target

```python
# Search any Kepler/TESS star
ga.load_data('KIC 11904151')  # By KIC number
ga.load_data('Kepler-88')     # Known TTV system
ga.load_data('Kepler-447')    # Grazing transit
```

### Adjust GA Parameters

```python
ga = ExoGA(
    pop_size=100,          # Larger population
    n_generations=200,     # More generations
    temp_initial=20.0,     # Higher initial exploration
    temp_final=0.05,       # Lower final exploitation
    mutation_rate=0.2,     # Higher mutation rate
    alpha=0.5,             # BLX-α crossover parameter
    elitism=5              # Keep top 5 individuals
)
```

### Multi-Planet Search

```python
# Search for multiple planets
ga = ExoGA(n_planets=2)  # Search for 2 planets
ga.load_data('Kepler-88')  # Known multi-planet system
ga.evolve()
```

## Known Exoplanet Parameters (for validation)

### Kepler-10b
- **Period**: 0.837 days (20.09 hours)
- **Depth**: ~0.15% (1500 ppm)
- **Duration**: ~1.8 hours
- **TTVs**: None (strictly periodic)

### Kepler-88b (TTV case)
- **Period**: ~10.95 days
- **Depth**: ~0.08%
- **TTVs**: Yes, significant variations from Kepler-88c

### Kepler-447b (Grazing transit)
- **Period**: ~7.79 days
- **Depth**: ~0.03%
- **Grazing**: Eclipses only stellar limb

## Output Files

- `*_convergence.png`: Fitness convergence and temperature decay plots
- `*_fit.png`: Model fit comparison (full, zoomed, phase-folded)
- `*_results.pkl`: Pickled results object with best individual and history

## Interpreting Results

### Good Fit Indicators
- Fitness (NLL) decreases over generations
- Phase-folded transit shows clear dip
- Recovered period matches known value (±5%)
- Transit depth matches literature value

### Poor Fit Indicators  
- Fitness plateaus early at high value
- No clear transit in phase-folded plot
- Unrealistic parameters (e.g., period > 100 days for hot Jupiter)
- Multiple transits don't align when phase-folded

## Troubleshooting

### "No module named 'lightkurve'"
```bash
pip install lightkurve
```

### "No data loaded" error
Make sure to call `ga.load_data()` before `ga.evolve()`

### Slow convergence
- Increase population size
- Increase number of generations
- Adjust temperature schedule (higher initial temp)

### Missing transits in results
- Try different initial random seed
- Increase mutation rate for more exploration
- Check if transit depth is above noise level

## Citation

Based on proposal by John Nowinski and Joseph Hughes (2025)
"ExoGA: Genetic Algorithm Optimization for Exoplanet Transit Modeling"
ECE848 Project

## References

1. NASA Exoplanet Archive: https://exoplanetarchive.ipac.caltech.edu/
2. Lightkurve: https://docs.lightkurve.org/
3. Box Least Squares (BLS): Kovács et al. (2002)
4. Transit Least Squares (TLS): Hippke & Heller (2019)
5. Kepler-88 TTVs: Nesvorný et al. (2013)
