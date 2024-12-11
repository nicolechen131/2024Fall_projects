import pandas as pd
import numpy as np

# Read data from CSV files
population_data = pd.read_csv('/Users/nicole/monte_carlo_vaccine_sim/data/population_data.csv')
infection_rates = pd.read_csv('/Users/nicole/monte_carlo_vaccine_sim/data/infection_rates.csv')
vaccine_params = pd.read_csv('/Users/nicole/monte_carlo_vaccine_sim/data/vaccine_params.csv')


def allocate_vaccines(population_data, infection_rates, vaccine_params):
    """Allocate vaccines to regions based on infection rates"""
    # Sort by infection rate
    infection_rates_sorted = infection_rates.sort_values(by='infection_rate', ascending=False).reset_index(drop=True)

    # Initialize vaccine allocation column
    population_data = population_data.copy()
    population_data['vaccines_allocated'] = 0

    for idx, row in infection_rates_sorted.iterrows():
        region_id = row['region_id']
        infection_rate = row['infection_rate']

        # Retrieve population data for the region
        population_row = population_data.loc[population_data['region_id'] == region_id]
        population_size = population_row['population_size'].values[0]

        # Retrieve vaccine availability and efficacy for the region
        vaccine_row = vaccine_params.loc[vaccine_params['region_id'] == region_id]
        vaccine_availability = vaccine_row['vaccine_availability'].values[0]

        # Allocate vaccines
        allocation = min(population_size, vaccine_availability)
        population_data.loc[population_data['region_id'] == region_id, 'vaccines_allocated'] = allocation

    return population_data


def calculate_infections_prevented(population_data, infection_rates, vaccine_params):
    """Calculate infections prevented for each region"""
    population_data = population_data.copy()
    population_data = pd.merge(population_data, infection_rates, on='region_id')
    population_data = pd.merge(population_data, vaccine_params[['region_id', 'vaccine_efficacy']], on='region_id')

    # Calculate effective vaccine coverage
    population_data['effective_vaccines'] = population_data['vaccines_allocated'] * population_data['vaccine_efficacy']

    # Calculate infections prevented
    population_data['infections_prevented'] = population_data['effective_vaccines'] * population_data['infection_rate']

    total_infections_prevented = population_data['infections_prevented'].sum()
    return total_infections_prevented, population_data


def monte_carlo_simulation(population_data, infection_rates, vaccine_params, runs=1000):
    """Perform Monte Carlo simulation"""
    results = []
    for _ in range(runs):
        # Randomly adjust infection rates (simulate uncertainty)
        infection_rates['infection_rate'] = np.clip(
            infection_rates['infection_rate'] * np.random.uniform(0.9, 1.1, size=len(infection_rates)),
            0.01, 0.1
        )

        # Allocate vaccines
        allocated_population_data = allocate_vaccines(population_data, infection_rates, vaccine_params)

        # Calculate infections prevented
        infections_prevented, _ = calculate_infections_prevented(allocated_population_data, infection_rates,
                                                                 vaccine_params)
        results.append(infections_prevented)
    return results


import matplotlib.pyplot as plt  # Import plotting module

if __name__ == "__main__":
    # Population data
    population_data = pd.DataFrame({
        'region_id': range(1, 101),
        'region_name': [f"Region_{i}" for i in range(1, 101)],
        'population_size': pd.Series([1000 + i * 1000 for i in range(100)]),  # Example data
    })

    # Infection rate data
    infection_rates = pd.DataFrame({
        'region_id': range(1, 101),
        'infection_rate': pd.Series([0.02 + (i % 10) * 0.01 for i in range(100)]),  # Example data
    })

    # Load vaccine data from vaccine_params.csv
    vaccine_params = pd.read_csv('/Users/nicole/monte_carlo_vaccine_sim/data/vaccine_params.csv')

    # Allocate vaccines
    allocated_population_data = allocate_vaccines(population_data, infection_rates, vaccine_params)

    # Calculate infections prevented
    total_prevented, result_data = calculate_infections_prevented(allocated_population_data, infection_rates,
                                                                  vaccine_params)

    # Print results
    print(f"Total number of infections prevented: {total_prevented:.2f}")
    print(result_data.head())

    # Perform Monte Carlo simulation
    print("Running Monte Carlo Simulation...")
    results = monte_carlo_simulation(population_data, infection_rates, vaccine_params, runs=1000)

    # Print Monte Carlo simulation statistics
    print(f"Mean infections prevented: {np.mean(results):.2f}")
    print(f"Standard deviation: {np.std(results):.2f}")

    # Plot histogram
    plt.hist(results, bins=20, edgecolor='k', alpha=0.7)
    plt.title('Monte Carlo Simulation: Infections Prevented')
    plt.xlabel('Infections Prevented')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)

    # Save histogram
    plt.savefig('infections_prevented_histogram.png')  # Save as PNG file
    plt.show()  # Display the plot



