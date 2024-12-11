import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from typing import Tuple, List, Dict


class VaccineSimulation:
    def __init__(self, population_data: pd.DataFrame, infection_rates: pd.DataFrame,
                 vaccine_params: pd.DataFrame):
        self.population_data = population_data.copy()
        self.infection_rates = infection_rates.copy()
        self.vaccine_params = vaccine_params.copy()
        self.validation_results = {}
        self.control_results = None

    def validate_input_data(self) -> bool:
        required_cols = {
            'population_data': ['region_id', 'population_size'],
            'infection_rates': ['region_id', 'infection_rate'],
            'vaccine_params': ['region_id', 'vaccine_availability', 'vaccine_efficacy']
        }

        try:
            for df_name, cols in required_cols.items():
                df = getattr(self, df_name)
                if not all(col in df.columns for col in cols):
                    print(f"Missing required columns in {df_name}")
                    return False

            if not (0 <= self.infection_rates['infection_rate'].all() <= 1):
                print("Invalid infection rates detected")
                return False

            if not (0 <= self.vaccine_params['vaccine_efficacy'].all() <= 1):
                print("Invalid vaccine efficacy detected")
                return False

            return True
        except Exception as e:
            print(f"Validation error: {str(e)}")
            return False

    def allocate_vaccines(self, infection_rates: pd.DataFrame) -> pd.DataFrame:
        """Allocate vaccines based on infection rates"""
        population_data = self.population_data.copy()
        infection_rates_sorted = infection_rates.sort_values(
            by='infection_rate', ascending=False).reset_index(drop=True)

        population_data['vaccines_allocated'] = 0

        for idx, row in infection_rates_sorted.iterrows():
            region_id = row['region_id']
            population_size = population_data.loc[
                population_data['region_id'] == region_id, 'population_size'].values[0]
            vaccine_availability = self.vaccine_params.loc[
                self.vaccine_params['region_id'] == region_id, 'vaccine_availability'].values[0]

            allocation = int(min(population_size, vaccine_availability))  # Fixed: Convert to int
            population_data.loc[population_data['region_id'] == region_id,
                              'vaccines_allocated'] = allocation

        return population_data

    def calculate_infections_prevented(self, population_data: pd.DataFrame, infection_rates: pd.DataFrame):
        result_data = population_data.copy()
        result_data = pd.merge(result_data, infection_rates, on='region_id', suffixes=('_old', ''))
        result_data = pd.merge(result_data,
                               self.vaccine_params[['region_id', 'vaccine_efficacy']],
                               on='region_id')

        result_data['effective_vaccines'] = (result_data['vaccines_allocated'] *
                                             result_data['vaccine_efficacy'])
        result_data['infections_prevented'] = (result_data['effective_vaccines'] *
                                               result_data['infection_rate'])

        return result_data['infections_prevented'].sum(), result_data

    def run_single_simulation(self, uncertainty_factor: float = 0.1) -> Tuple[float, pd.DataFrame]:
        """Run a single iteration of the simulation"""
        # Apply uncertainty to infection rates
        modified_rates = self.infection_rates.copy()
        modified_rates['infection_rate'] = np.clip(
            modified_rates['infection_rate'] *
            np.random.uniform(1 - uncertainty_factor, 1 + uncertainty_factor,
                              size=len(modified_rates)),
            0.01, 0.99
        )

        # Run allocation and calculation
        allocated_data = self.allocate_vaccines(modified_rates)
        infections_prevented, result_data = self.calculate_infections_prevented(
            allocated_data, modified_rates)

        return infections_prevented, result_data

    def validate_simulation(self, n_runs: int = 100) -> bool:
        """Phase 2: Validate simulation behavior"""
        print("Running validation tests...")

        # Test statistical convergence
        results = []
        running_means = []

        for i in range(n_runs):
            infections_prevented, _ = self.run_single_simulation()
            results.append(infections_prevented)
            running_means.append(np.mean(results))

        # Check convergence
        if len(results) > 10:
            recent_mean = np.mean(running_means[-10:])
            recent_std = np.std(running_means[-10:])
            cv = recent_std / recent_mean if recent_mean != 0 else float('inf')

            self.validation_results = {
                'mean': np.mean(results),
                'std': np.std(results),
                'cv': cv,
                'converged': cv < 0.01
            }

            print(f"Validation Results:")
            print(f"Mean: {self.validation_results['mean']:.2f}")
            print(f"Standard Deviation: {self.validation_results['std']:.2f}")
            print(f"Coefficient of Variation: {self.validation_results['cv']:.4f}")
            print(f"Convergence Status: {'Passed' if self.validation_results['converged'] else 'Failed'}")

            # Plot convergence
            plt.figure(figsize=(10, 6))
            plt.plot(running_means)
            plt.title('Convergence of Mean Infections Prevented')
            plt.xlabel('Number of Simulations')
            plt.ylabel('Running Mean')
            plt.grid(True)
            plt.savefig('convergence_plot.png')
            plt.close()

            return self.validation_results['converged']

        return False

    def run_experiment(self, n_runs: int = 1000,
                       experiment_params: Dict = None) -> Dict:
        """Phase 3: Run experimental simulations"""
        if not self.validation_results.get('converged', False):
            print("Warning: Simulation not validated. Run validate_simulation() first.")

        # Store control results if not already done
        if self.control_results is None:
            control_results = []
            for _ in range(n_runs):
                infections_prevented, _ = self.run_single_simulation()
                control_results.append(infections_prevented)
            self.control_results = control_results

        # Run experimental condition
        if experiment_params:
            # Modify simulation parameters based on experimental condition
            original_params = self.vaccine_params.copy()
            for param, value in experiment_params.items():
                if param in self.vaccine_params.columns:
                    self.vaccine_params[param] = self.vaccine_params[param] * value

            # Run experimental simulations
            experimental_results = []
            for _ in range(n_runs):
                infections_prevented, _ = self.run_single_simulation()
                experimental_results.append(infections_prevented)

            # Restore original parameters
            self.vaccine_params = original_params

            # Calculate statistics
            t_stat, p_value = stats.ttest_ind(self.control_results, experimental_results)

            results = {
                'control_mean': np.mean(self.control_results),
                'experimental_mean': np.mean(experimental_results),
                'difference': np.mean(experimental_results) - np.mean(self.control_results),
                'p_value': p_value,
                'significant': p_value < 0.05
            }

            # Plot comparison
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            sns.histplot(data=self.control_results, label='Control', alpha=0.5)
            sns.histplot(data=experimental_results, label='Experimental', alpha=0.5)
            plt.title('Distribution Comparison')
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.boxplot([self.control_results, experimental_results],
                        labels=['Control', 'Experimental'])
            plt.title('Box Plot Comparison')
            plt.savefig('experiment_comparison.png')
            plt.close()

            return results

        return {'error': 'No experimental parameters provided'}


# Example usage
if __name__ == "__main__":
    # Create sample data
    population_data = pd.DataFrame({
        'region_id': range(1, 101),
        'population_size': [1000 + i * 1000 for i in range(100)]
    })

    infection_rates = pd.DataFrame({
        'region_id': range(1, 101),
        'infection_rate': [0.02 + (i % 10) * 0.01 for i in range(100)]
    })

    vaccine_params = pd.DataFrame({
        'region_id': range(1, 101),
        'vaccine_availability': [800 + i * 100 for i in range(100)],
        'vaccine_efficacy': [0.85 + (i % 5) * 0.01 for i in range(100)]
    })

    # Initialize and run simulation
    sim = VaccineSimulation(population_data, infection_rates, vaccine_params)

    # Phase 1: Validate input data
    if sim.validate_input_data():
        print("Input data validation passed")

        # Phase 2: Validate simulation behavior
        if sim.validate_simulation(n_runs=200):
            print("Simulation validation passed")

            # Phase 3: Run experiment
            experiment_params = {'vaccine_efficacy': 1.2}  # Test 20% improved vaccine efficacy
            results = sim.run_experiment(n_runs=1000, experiment_params=experiment_params)

            print("\nExperiment Results:")
            print(f"Control Mean: {results['control_mean']:.2f}")
            print(f"Experimental Mean: {results['experimental_mean']:.2f}")
            print(f"Difference: {results['difference']:.2f}")
            print(f"P-value: {results['p_value']:.4f}")
            print(f"Statistically Significant: {'Yes' if results['significant'] else 'No'}")
        else:
            print("Simulation validation failed")
    else:
        print("Input data validation failed")