import numpy as np
from typing import Union

def calculate_stats(data: Union[np.ndarray, list]) -> tuple[float, float]:
    """
    Calculate mean and standard deviation of a list or numpy array
    
    Args:
        data: List or numpy array of numeric values
        
    Returns:
        tuple: (mean, std) as floats
    """
    # Convert to numpy array if it's a list
    if isinstance(data, list):
        data = np.array(data)
    
    # Calculate mean and standard deviation
    mean = np.mean(data)
    std = np.std(data)
    
    return mean, std

def change_rate_to_data(change_rates: Union[np.ndarray, list], initial_value: float = 1.0) -> np.ndarray:
    """
    Convert change rates to actual data values
    
    Args:
        change_rates: List or numpy array of change rates (as percentages)
        initial_value: Starting value for the data series
        
    Returns:
        numpy array: Actual data values
    """
    # Convert to numpy array if it's a list
    if isinstance(change_rates, list):
        change_rates = np.array(change_rates)
    
    # Convert percentage change rates to multipliers (e.g., 2.5% -> 1.025)
    multipliers = 1 + change_rates / 100
    
    # Calculate cumulative product to get the data values
    data = initial_value * np.cumprod(multipliers)
    
    return data


def data_to_change_rate(data: Union[np.ndarray, list]) -> np.ndarray:
    """
    Convert actual data values to change rates
    
    Args:
        data: List or numpy array of data values
        
    Returns:
        numpy array: Change rates as percentages
    """
    # Convert to numpy array if it's a list
    if isinstance(data, list):
        data = np.array(data)

    change_present = (data[1:] - data[:-1]) / data[:-1]
    
    # Calculate change rates as percentages
    return change_present * 100
