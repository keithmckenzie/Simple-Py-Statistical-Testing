#Keith Ngamphon McKenzie
#keith@mckenzie.page
#https://mckenzie.page
#Python Simple Statistical Tests

import re
from typing import List, Any, Optional
import numpy as np

def parse_comma_separated(data_str: str) -> List[float]:
    """
    Parse comma-separated string into list of floats
    
    Args:
        data_str: String containing comma-separated numbers
        
    Returns:
        List of float values
        
    Raises:
        ValueError: If parsing fails
    """
    # Remove extra whitespace and split by comma
    cleaned = re.sub(r'\s+', '', data_str)
    parts = cleaned.split(',')
    
    # Filter out empty parts
    parts = [p for p in parts if p]
    
    if not parts:
        raise ValueError("No valid numbers found")
    
    try:
        return [float(p) for p in parts]
    except ValueError as e:
        raise ValueError(f"Invalid number format: {e}")

def validate_numeric_data(data: List[Any]) -> bool:
    """
    Validate that data contains only numeric values
    
    Args:
        data: List to validate
        
    Returns:
        bool: True if all values are numeric
    """
    if not data:
        return False
    
    try:
        [float(x) for x in data]
        return True
    except (ValueError, TypeError):
        return False

def validate_minimum_sample_size(data: List[float], min_size: int) -> bool:
    """
    Check if data meets minimum sample size requirement
    
    Args:
        data: Dataset to check
        min_size: Minimum required size
        
    Returns:
        bool: True if sample size is adequate
    """
    return len(data) >= min_size

def validate_equal_sample_sizes(data1: List[float], data2: List[float]) -> bool:
    """
    Check if two datasets have equal sample sizes
    
    Args:
        data1: First dataset
        data2: Second dataset
        
    Returns:
        bool: True if sizes are equal
    """
    return len(data1) == len(data2)

def validate_normality_assumption(data: List[float], min_size: int = 30) -> tuple:
    """
    Basic check for normality assumption
    
    Args:
        data: Dataset to check
        min_size: Minimum size for Central Limit Theorem
        
    Returns:
        tuple: (is_valid, warning_message)
    """
    n = len(data)
    
    if n < min_size:
        return False, f"Sample size ({n}) is small. Consider normality testing."
    
    # Check for extreme outliers (beyond 3 standard deviations)
    if n > 1:
        mean = np.mean(data)
        std = np.std(data)
        outliers = [x for x in data if abs(x - mean) > 3 * std]
        
        if len(outliers) > n * 0.05:  # More than 5% outliers
            return False, f"Dataset has {len(outliers)} potential outliers. Check normality."
    
    return True, "Sample size adequate for normality assumption."

def validate_categorical_data(data: List[Any]) -> bool:
    """
    Validate data for categorical analysis
    
    Args:
        data: Data to validate
        
    Returns:
        bool: True if suitable for categorical analysis
    """
    if not data:
        return False
    
    # Check if all values are integers (counts/frequencies)
    try:
        int_data = [int(x) for x in data]
        return all(x >= 0 for x in int_data)  # Non-negative integers
    except (ValueError, TypeError):
        return False

def validate_contingency_table(table_str: str) -> tuple:
    """
    Validate and parse contingency table input
    
    Args:
        table_str: String representation of contingency table
        
    Returns:
        tuple: (is_valid, parsed_table, error_message)
    """
    try:
        # Expected format: rows separated by semicolons, values by commas
        # Example: "10,15,20;25,30,35"
        rows = table_str.split(';')
        
        if len(rows) < 2:
            return False, None, "Need at least 2 rows for contingency table"
        
        table = []
        expected_cols = None
        
        for row_str in rows:
            row = [float(x.strip()) for x in row_str.split(',')]
            
            if expected_cols is None:
                expected_cols = len(row)
            elif len(row) != expected_cols:
                return False, None, "All rows must have the same number of columns"
            
            if any(x < 0 for x in row):
                return False, None, "All values must be non-negative"
            
            table.append(row)
        
        return True, table, None
        
    except Exception as e:
        return False, None, f"Invalid table format: {e}"

def get_hypothesis_input(test_name: str) -> tuple:
    """
    Get null and alternative hypothesis from user
    
    Args:
        test_name: Name of the statistical test
        
    Returns:
        tuple: (null_hypothesis, alternative_hypothesis)
    """
    print(f"\nFor {test_name}, please state your hypotheses:")
    
    print("Null Hypothesis (H0):")
    null_hyp = input("> ").strip()
    
    print("Alternative Hypothesis (H1 or Ha):")
    alt_hyp = input("> ").strip()
    
    if not null_hyp:
        null_hyp = "No significant difference/effect"
    if not alt_hyp:
        alt_hyp = "Significant difference/effect exists"
    
    return null_hyp, alt_hyp

def validate_correlation_data(x_data: List[float], y_data: List[float]) -> tuple:
    """
    Validate data for correlation analysis
    
    Args:
        x_data: X variable data
        y_data: Y variable data
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if len(x_data) != len(y_data):
        return False, "X and Y datasets must have equal length"
    
    if len(x_data) < 3:
        return False, "Need at least 3 data points for correlation"
    
    # Check for constant variables
    if len(set(x_data)) == 1:
        return False, "X variable is constant (no variation)"
    
    if len(set(y_data)) == 1:
        return False, "Y variable is constant (no variation)"
    
    return True, None
