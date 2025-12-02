"""
Genetic Algorithm components for exoplanet transit detection
"""

import numpy as np
from dataclasses import dataclass
from transit_models import limb_darkened_transit


# =============================================================================
# GA PARAMETERS
# =============================================================================
POPULATION_SIZE = 100
NUM_GENERATIONS = 300
ELITE_COUNT = 2
CROSSOVER_RATE = 0.7
MUTATION_RATE_EARLY = 0.3
MUTATION_RATE_LATE = 0.1
TOURNAMENT_SIZE = 2
BLX_ALPHA = 0.5

MUTATION_STD_P = 0.02
MUTATION_STD_T0 = 1.0
MUTATION_STD_RPRS = 0.005
MUTATION_STD_IMPACT = 0.15

PERIOD_RANGE = (0.98, 1.02)
RPRS_RANGE = (0.02, 0.15)
IMPACT_RANGE = (0.0, 0.95)

PERIOD_BOUNDS = (0.95, 1.05)
RPRS_BOUNDS = (0.02, 0.2)
IMPACT_BOUNDS = (0.0, 0.95)

CONVERGENCE_THRESHOLD = 1e-10
CONVERGENCE_WINDOW = 50

@dataclass
class Individual:
    """Single candidate solution"""
    T0: float
    RpRs: float
    impact: float
    P: float
    fitness: float = np.inf
    
    @property
    def depth(self):
        return self.RpRs ** 2


# =============================================================================
# GA STEP FUNCTIONS
# =============================================================================

def initialize_population(time, known_period, pop_size=POPULATION_SIZE):
    """
    Create initial population with random individuals
    
    Args:
        time: observation times
        known_period: orbital period (days) - FIXED
        pop_size: population size
    
    Returns:
        list of Individual objects
    """
    population = []
    for i in range(pop_size):
        T0 = np.random.uniform(time[0], time[0] + known_period)
        RpRs = np.random.uniform(*RPRS_RANGE)
        impact = np.random.uniform(*IMPACT_RANGE)
        
        population.append(Individual(T0, RpRs, impact, known_period))
    
    return population


def evaluate_fitness(individual, time, flux, stellar, model='batman'):
    """
    Calculate fitness (chi-squared) for an individual
    
    Args:
        individual: Individual to evaluate
        time: observation times
        flux: normalized flux values
        stellar: StellarParams object
        model: transit model to use ('batman', 'pytransit', or 'ellc')
    
    Returns:
        fitness value (lower is better)
    """
    transit_model = limb_darkened_transit(time, individual.T0, individual.RpRs, 
                                          individual.P, individual.impact, stellar, model)
    residuals = flux - transit_model
    return np.sum(residuals**2)


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
        child T0, RpRs, impact values
    """
    child_T0 = np.random.uniform(
        min(parent1.T0, parent2.T0) - alpha * abs(parent1.T0 - parent2.T0),
        max(parent1.T0, parent2.T0) + alpha * abs(parent1.T0 - parent2.T0)
    )
    child_RpRs = np.random.uniform(
        min(parent1.RpRs, parent2.RpRs) - alpha * abs(parent1.RpRs - parent2.RpRs),
        max(parent1.RpRs, parent2.RpRs) + alpha * abs(parent1.RpRs - parent2.RpRs)
    )
    child_impact = np.random.uniform(
        min(parent1.impact, parent2.impact) - alpha * abs(parent1.impact - parent2.impact),
        max(parent1.impact, parent2.impact) + alpha * abs(parent1.impact - parent2.impact)
    )
    
    return child_T0, child_RpRs, child_impact


def mutate(T0, RpRs, impact, mutation_rate):
    """
    Apply Gaussian mutation to parameters
    
    Args:
        T0, RpRs, impact: parameter values
        mutation_rate: probability of mutation
    
    Returns:
        mutated T0, RpRs, impact values
    """
    if np.random.random() < mutation_rate:
        T0 += np.random.normal(0, MUTATION_STD_T0)
    if np.random.random() < mutation_rate:
        RpRs += np.random.normal(0, MUTATION_STD_RPRS)
    if np.random.random() < mutation_rate:
        impact += np.random.normal(0, MUTATION_STD_IMPACT)
    
    RpRs = np.clip(RpRs, *RPRS_BOUNDS)
    impact = np.clip(impact, *IMPACT_BOUNDS)
    
    return T0, RpRs, impact


def create_next_generation(population, pop_size, gen, n_gen, known_period):
    """
    Create next generation through selection, crossover, and mutation
    
    Args:
        population: current population (sorted by fitness)
        pop_size: target population size
        gen: current generation number
        n_gen: total number of generations
        known_period: orbital period
    
    Returns:
        new population
    """
    mutation_rate = MUTATION_RATE_EARLY if gen < n_gen // 2 else MUTATION_RATE_LATE
    
    new_pop = population[:ELITE_COUNT]
    
    while len(new_pop) < pop_size:
        p1 = tournament_selection(population)
        p2 = tournament_selection(population)
        
        if np.random.random() < CROSSOVER_RATE:
            child_T0, child_RpRs, child_impact = crossover(p1, p2)
        else:
            parent = p1 if np.random.random() < 0.5 else p2
            child_T0 = parent.T0
            child_RpRs = parent.RpRs
            child_impact = parent.impact
        
        child_T0, child_RpRs, child_impact = mutate(
            child_T0, child_RpRs, child_impact, mutation_rate
        )
        
        new_pop.append(Individual(child_T0, child_RpRs, child_impact, known_period))
    
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
    
    print(f"Gen {gen+1}: Fitness={best.fitness:.2e}, P={best.P:.6f}d, "
          f"Rp/Rs={best.RpRs:.4f} (depth={best.depth*100:.4f}%), impact={best.impact:.3f}, dur={duration*24:.2f}hr")
    print(f"  Top 10 Rp/Rs range: [{min(rprs_vals):.4f}, {max(rprs_vals):.4f}], "
          f"impact range: [{min(impacts):.3f}, {max(impacts):.3f}]")


# =============================================================================
# MAIN GA FUNCTION
# =============================================================================

def run_ga(time, flux, stellar, known_period, pop_size=POPULATION_SIZE, n_gen=NUM_GENERATIONS, model='batman'):
    """
    Run genetic algorithm to optimize transit parameters
    
    Args:
        time: observation times
        flux: normalized flux values
        stellar: StellarParams object
        known_period: orbital period (days)
        pop_size: population size
        n_gen: number of generations
        model: transit model to use ('batman', 'pytransit', or 'ellc')
    
    Returns:
        best individual, fitness history
    """
    # Initialize population
    population = initialize_population(time, known_period, pop_size)
    
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
        population = create_next_generation(population, pop_size, gen, n_gen, known_period)
        
        # Evaluate new individuals (skip elites which already have fitness)
        for ind in population[ELITE_COUNT:]:
            ind.fitness = evaluate_fitness(ind, time, flux, stellar, model)
    
    # Return best individual
    best = min(population, key=lambda x: x.fitness)
    return best, best_fitness_history
