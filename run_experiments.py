from src.simulation import VaccineSimulation
from src.utils import load_simulation_data
import pandas as pd
from pathlib import Path


def run_vaccine_experiments():
    # Load data
    population_data, infection_rates, vaccine_params = load_simulation_data()

    # Debug print
    print("Infection rates columns:", infection_rates.columns)
    print("Sample data:", infection_rates.head())

    # Rest of the function remains the same
    sim = VaccineSimulation(population_data, infection_rates, vaccine_params)

    if not sim.validate_input_data():
        raise ValueError("Input data validation failed")

    if not sim.validate_simulation(n_runs=200):
        raise ValueError("Simulation validation failed")

    experiments = {
        'increased_efficacy': {'vaccine_efficacy': 1.2},
        'increased_availability': {'vaccine_availability': 1.5},
        'combined_improvement': {
            'vaccine_efficacy': 1.2,
            'vaccine_availability': 1.5
        }
    }

    results = {}
    for name, params in experiments.items():
        print(f"\nRunning experiment: {name}")
        results[name] = sim.run_experiment(n_runs=1000, experiment_params=params)

        print(f"Control Mean: {results[name]['control_mean']:.2f}")
        print(f"Experimental Mean: {results[name]['experimental_mean']:.2f}")
        print(f"Improvement: {results[name]['difference']:.2f}")
        print(f"Statistically Significant: {'Yes' if results[name]['significant'] else 'No'}")

if __name__ == "__main__":
    run_vaccine_experiments()