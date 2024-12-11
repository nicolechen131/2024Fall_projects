import pandas as pd
from pathlib import Path


def load_simulation_data() -> tuple:
    """Load CSV files using relative paths"""
    current_dir = Path(__file__).parent.parent
    data_dir = current_dir / 'data'

    data_dir.mkdir(exist_ok=True)

    population_data = pd.read_csv(data_dir / 'population_data.csv')
    infection_rates = pd.read_csv(data_dir / 'infection_rates.csv')
    vaccine_params = pd.read_csv(data_dir / 'vaccine_params.csv')

    return population_data, infection_rates, vaccine_params