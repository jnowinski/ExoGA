"""
Genetic Algorithm components for exoplanet transit detection
"""

import numpy as np
from dataclasses import dataclass
from transit_models import limb_darkened_transit


# =============================================================================
# GA PARAMETERS
# =============================================================================
POPULATION_SIZE = 300
NUM_GENERATIONS = 100
ELITE_COUNT = 6
CROSSOVER_RATE = 0.7
MUTATION_RATE_EARLY = 0.3
MUTATION_RATE_LATE = 0.1
TOURNAMENT_SIZE = 2
BLX_ALPHA = 0.5

# Control whether to evolve u2 limb darkening coefficient
EVOLVE_U2 = True  # Set to False to fix u2 at theoretical value

MUTATION_STD_P = 0.02
MUTATION_STD_T0 = 1.0
MUTATION_STD_RPRS = 0.005
MUTATION_STD_IMPACT = 0.15
MUTATION_STD_U2 = 0.05
# MUTATION_STD_ECC = 0.1
# MUTATION_STD_OMEGA = 20.0

PERIOD_RANGE = (0.98, 1.02)
RPRS_RANGE = (0.02, 0.15)
IMPACT_RANGE = (0.0, 0.95)
LD_VARIATION_PERCENT = 0.20
# ECC_RANGE = (0.0, 0.8)
# OMEGA_RANGE = (0.0, 360.0)

PERIOD_BOUNDS = (0.95, 1.05)
RPRS_BOUNDS = (0.02, 0.2)
IMPACT_BOUNDS = (0.0, 0.95)
# ECC_BOUNDS = (0.0, 0.8)
# OMEGA_BOUNDS = (0.0, 360.0)

CONVERGENCE_THRESHOLD = 1e-10
CONVERGENCE_WINDOW = 50

@dataclass
class Individual:
    """Single candidate solution"""
    T0: float
    RpRs: float
    impact: float
    P: float
    u2: float
    # ecc: float
    # omega: float
    # is_circular: bool = True  # Flag: True = circular orbit (ecc=0), False = eccentric
    fitness: float = np.inf
    
    @property
    def depth(self):
        return self.RpRs ** 2


# =============================================================================
# GA STEP FUNCTIONS
# =============================================================================

def initialize_population(time, known_period, stellar, pop_size=POPULATION_SIZE, evolve_u2=EVOLVE_U2):
    """
    Create initial population with random individuals
    Initialize u1/u2 close to Claret values for local search
    
    Args:
        time: observation times
        known_period: orbital period (days) - FIXED
        stellar: StellarParams object with u1, u2 from Claret
        pop_size: population size
    
    Returns:
        list of Individual objects
    """
    # Calculate ±20% bounds for u1/u2 based on Claret values
    u1_min = stellar.u1 * (1 - LD_VARIATION_PERCENT)
    u1_max = stellar.u1 * (1 + LD_VARIATION_PERCENT)
    u2_min = stellar.u2 * (1 - LD_VARIATION_PERCENT)
    u2_max = stellar.u2 * (1 + LD_VARIATION_PERCENT)
    
    # Ensure physical bounds (u1: 0-1, u2: -1 to 1)
    u1_min = max(0.0, u1_min)
    u1_max = min(1.0, u1_max)
    u2_min = max(-1.0, u2_min)
    u2_max = min(1.0, u2_max)
    
    population = []
    for i in range(pop_size):
        T0 = np.random.uniform(time[0], time[0] + known_period)
        RpRs = np.random.uniform(*RPRS_RANGE)
        impact = np.random.uniform(*IMPACT_RANGE)
        
        # Initialize u2 within ±20% of Claret values (u1 fixed in stellar params)
        # If evolve_u2 is False, fix u2 at theoretical value
        if evolve_u2:
            u2 = np.random.uniform(u2_min, u2_max)
        else:
            u2 = stellar.u2
        
        # # 80% circular, 20% eccentric (most hot Jupiters are circular)
        # is_circular = np.random.random() < 0.8
        # 
        # if is_circular:
        #     # Circular orbit: fix eccentricity
        #     ecc = 0.0
        #     omega = 0.0  # Omega undefined for circular orbits, set to 0
        # else:
        #     # Eccentric: use beta distribution biased toward low eccentricity
        #     ecc = np.random.beta(2, 5) * ECC_RANGE[1]  # Beta(2,5) peaks near 0
        #     omega = np.random.uniform(*OMEGA_RANGE)
        
        population.append(Individual(T0, RpRs, impact, known_period, u2))  # , ecc, omega, is_circular))
    
    return population


def evaluate_fitness(individual, time, flux, stellar, model='batman', transit_weight=50.0):
    """
    Calculate weighted fitness (chi-squared) for an individual
    Uses fixed u1 from stellar params, evolved u2 from individual
    Applies higher weight to in-transit points for sharper transit fits
    
    Args:
        individual: Individual to evaluate
        time: observation times
        flux: normalized flux values
        stellar: StellarParams object (for M_star, R_star, u1 fixed)
        model: transit model to use ('batman' or 'pytransit')
        transit_weight: weight multiplier for in-transit points (default: 10.0)
    
    Returns:
        fitness value (lower is better)
    """
    # Create temporary StellarParams with individual's u2
    from transit_models import StellarParams
    stellar_with_u2 = StellarParams(stellar.R_star, stellar.M_star, stellar.u1, individual.u2)
    
    transit_model = limb_darkened_transit(time, individual.T0, individual.RpRs, 
                                          individual.P, individual.impact, stellar_with_u2, model)
    
    # Identify in-transit points (where model shows transit)
    in_transit = transit_model < 0.999
    
    # Create weights: higher for in-transit points
    weights = np.ones_like(flux)
    weights[in_transit] = transit_weight
    
    residuals = flux - transit_model
    return np.sum(weights * residuals**2)


def tournament_selection(population, tournament_size=TOURNAMENT_SIZE):
    """
    Select an individual via tournament selection
    
    Args:
        population: list of individuals
        tournament_size: number of individuals to compete
    
    Returns:
        selected Individual
    """
    contestants = np.random.choice(population, tournament_size, replace=False)
    return min(contestants, key=lambda x: x.fitness)


def crossover(parent1, parent2, alpha=BLX_ALPHA):
    """
    BLX-α crossover - creates child from two parents
    
    Args:
        parent1, parent2: parent Individuals
        alpha: blend parameter
    
    Returns:
        child Individual
    """
    def blend(val1, val2):
        return np.random.uniform(
            min(val1, val2) - alpha * abs(val1 - val2),
            max(val1, val2) + alpha * abs(val1 - val2)
        )
    
    # Blend parent parameters with BLX-α
    T0 = blend(parent1.T0, parent2.T0)
    RpRs = blend(parent1.RpRs, parent2.RpRs)
    impact = blend(parent1.impact, parent2.impact)
    u2 = blend(parent1.u2, parent2.u2)
    
    # # Inherit circular flag from random parent, with 5% chance to flip
    # parent_flag = parent1.is_circular if np.random.random() < 0.5 else parent2.is_circular
    # is_circular = not parent_flag if np.random.random() < 0.05 else parent_flag
    # 
    # # Eccentricity parameters depend on flag
    # if is_circular:
    #     ecc = 0.0
    #     omega = 0.0
    # else:
    #     ecc = blend(parent1.ecc, parent2.ecc)
    #     omega = blend(parent1.omega, parent2.omega)
    
    return Individual(T0, RpRs, impact, parent1.P, u2)  # , ecc, omega, is_circular)


def mutate(individual, time, mutation_rate, stellar, evolve_u2=EVOLVE_U2):
    """
    Apply Gaussian mutation to individual's parameters
    u1 is FIXED, u2 is evolved within ±20% of Claret value if evolve_u2=True
    
    Args:
        individual: Individual to mutate (modified in-place)
        time: observation times
        mutation_rate: probability of mutation
        stellar: StellarParams with Claret u2 value for bounds
        evolve_u2: whether to evolve u2 or keep it fixed
    """
    if np.random.random() < mutation_rate:
        individual.T0 += np.random.normal(0, MUTATION_STD_T0)
    if np.random.random() < mutation_rate:
        individual.RpRs += np.random.normal(0, MUTATION_STD_RPRS)
    if np.random.random() < mutation_rate:
        individual.impact += np.random.normal(0, MUTATION_STD_IMPACT)
    if evolve_u2 and np.random.random() < mutation_rate:
        individual.u2 += np.random.normal(0, MUTATION_STD_U2)
    
    # # Only mutate eccentricity if not circular
    # if not individual.is_circular:
    #     if np.random.random() < mutation_rate:
    #         individual.ecc += np.random.normal(0, MUTATION_STD_ECC)
    #     if np.random.random() < mutation_rate:
    #         individual.omega += np.random.normal(0, MUTATION_STD_OMEGA)
    
    # Calculate ±20% bounds for u2 (only if evolving)
    if evolve_u2:
        u2_min = max(-1.0, stellar.u2 * (1 - LD_VARIATION_PERCENT))
        u2_max = min(1.0, stellar.u2 * (1 + LD_VARIATION_PERCENT))
    else:
        u2_min = u2_max = stellar.u2  # Fixed at theoretical value
    
    individual.T0 = np.clip(individual.T0, time[0], time[-1])
    individual.RpRs = np.clip(individual.RpRs, *RPRS_BOUNDS)
    individual.impact = np.clip(individual.impact, *IMPACT_BOUNDS)
    individual.u2 = np.clip(individual.u2, u2_min, u2_max)
    
    # # Only clip ecc/omega if eccentric
    # if not individual.is_circular:
    #     individual.ecc = np.clip(individual.ecc, *ECC_BOUNDS)
    #     individual.omega = np.clip(individual.omega, *OMEGA_BOUNDS)
    # else:
    #     individual.ecc = 0.0
    #     individual.omega = 0.0


def create_next_generation(population, pop_size, gen, n_gen, known_period, time, stellar, evolve_u2=EVOLVE_U2):
    """
    Create next generation through selection, crossover, and mutation
    
    Args:
        population: current population (sorted by fitness)
        pop_size: target population size
        gen: current generation number
        n_gen: total number of generations
        known_period: orbital period
        time: observation times (for mutation bounds)
        stellar: StellarParams with Claret u1/u2 for bounds
        evolve_u2: whether to evolve u2 or keep it fixed
    
    Returns:
        new population
    """
    mutation_rate = MUTATION_RATE_EARLY if gen < n_gen // 2 else MUTATION_RATE_LATE
    
    new_pop = population[:ELITE_COUNT]
    
    while len(new_pop) < pop_size:
        p1 = tournament_selection(population)
        p2 = tournament_selection(population)
        
        if np.random.random() < CROSSOVER_RATE:
            child = crossover(p1, p2)
        else:
            parent = p1 if np.random.random() < 0.5 else p2
            child = Individual(parent.T0, parent.RpRs, parent.impact, parent.P, 
                             parent.u2)  # , parent.ecc, parent.omega, parent.is_circular)
        
        mutate(child, time, mutation_rate, stellar, evolve_u2)
        new_pop.append(child)
    
    return new_pop


def print_generation_stats(gen, best, population, stellar):
    """
    Print statistics for current generation
    
    Args:
        gen: generation number
        best: best individual
        population: current population
        stellar: StellarParams object
    """
    from transit_models import expected_duration
    
    duration = expected_duration(best.P, stellar, best.impact)
    rprs_vals = [ind.RpRs for ind in population[:10]]
    impacts = [ind.impact for ind in population[:10]]
    
    # # Count circular vs eccentric in population
    # n_circular = sum(1 for ind in population if ind.is_circular)
    # 
    # orbit_type = "CIRCULAR" if best.is_circular else "ECCENTRIC"
    print(f"Gen {gen+1}: Fitness={best.fitness:.2e}, P={best.P:.6f}d, "
          f"Rp/Rs={best.RpRs:.4f} (depth={best.depth*100:.4f}%), impact={best.impact:.3f}, dur={duration*24:.2f}hr")
    print(f"  u1={stellar.u1:.4f} (fixed), u2={best.u2:.4f} (Claret: {stellar.u2:.4f})")
    # print(f"  {orbit_type}: ecc={best.ecc:.4f}, omega={best.omega:.1f}° | Pop: {n_circular} circular, {len(population)-n_circular} eccentric")
    print(f"  Top 10 Rp/Rs range: [{min(rprs_vals):.4f}, {max(rprs_vals):.4f}], "
          f"impact range: [{min(impacts):.3f}, {max(impacts):.3f}]")


# =============================================================================
# MAIN GA FUNCTION
# =============================================================================

def run_ga(time, flux, stellar, known_period, pop_size=POPULATION_SIZE, n_gen=NUM_GENERATIONS, model='batman', evolve_u2=EVOLVE_U2):
    """
    Run genetic algorithm to optimize transit parameters
    
    Args:
        time: observation times
        flux: normalized flux values
        stellar: StellarParams object
        known_period: orbital period (days)
        pop_size: population size
        n_gen: number of generations
        model: transit model to use ('batman' or 'pytransit')
        evolve_u2: whether to evolve u2 limb darkening coefficient (default: True)
    
    Returns:
        best individual, fitness history
    """
    # Initialize population
    population = initialize_population(time, known_period, stellar, pop_size, evolve_u2)
    
    # Evaluate initial fitness
    for ind in population:
        ind.fitness = evaluate_fitness(ind, time, flux, stellar, model)
    
    # Evolution loop
    best_fitness_history = []
    
    for gen in range(n_gen):
        # Sort by fitness
        population.sort(key=lambda x: x.fitness)
        
        # Track best
        best = population[0]
        best_fitness_history.append(best.fitness)
        
        # Check for convergence (after minimum generations)
        if gen >= CONVERGENCE_WINDOW:
            recent_fitness = best_fitness_history[-CONVERGENCE_WINDOW:]
            fitness_change = abs(recent_fitness[0] - recent_fitness[-1])
            relative_change = fitness_change / (recent_fitness[0] + 1e-10)
            
            if relative_change < CONVERGENCE_THRESHOLD:
                print(f"\n*** CONVERGED at generation {gen+1} ***")
                print(f"    Fitness change over last {CONVERGENCE_WINDOW} gens: {relative_change:.2e}")
                print(f"    (threshold: {CONVERGENCE_THRESHOLD:.2e})")
                break
        
        # Print progress
        if gen % 20 == 0 or gen == n_gen - 1:
            print_generation_stats(gen, best, population, stellar)
        
        # Create next generation
        population = create_next_generation(population, pop_size, gen, n_gen, known_period, time, stellar, evolve_u2)
        
        # Evaluate new individuals (skip elites which already have fitness)
        for ind in population[ELITE_COUNT:]:
            ind.fitness = evaluate_fitness(ind, time, flux, stellar, model)
    
    # Return best individual
    best = min(population, key=lambda x: x.fitness)
    return best, best_fitness_history
