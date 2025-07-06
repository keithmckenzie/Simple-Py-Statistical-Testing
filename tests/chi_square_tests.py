#Keith Ngamphon McKenzie
#keith@mckenzie.page
#https://mckenzie.page
#Python Simple Statistical Tests

import numpy as np
import scipy.stats as stats
from typing import List, Dict, Any, Tuple
from utils.validators import (validate_categorical_data, validate_contingency_table,
                            get_hypothesis_input)
from utils.formatters import print_test_results, print_assumption_warnings

class ChiSquareTests:
    """Class containing chi-square statistical tests"""
    
    @staticmethod
    def chi_square_goodness_of_fit(observed: List[float], expected: List[float] = None,
                                  alpha: float = 0.05) -> Dict[str, Any]:
        """
        Chi-Square Goodness of Fit Test
        
        Args:
            observed: Observed frequencies
            expected: Expected frequencies (if None, assumes equal distribution)
            alpha: Significance level
            
        Returns:
            Dictionary with test results
        """
        test_name = "Chi-Square Goodness of Fit Test"
        
        # Validate data
        if not validate_categorical_data(observed):
            raise ValueError("Observed data must be non-negative integers (frequencies)")
        
        observed = [int(x) for x in observed]
        
        # Set expected frequencies if not provided
        if expected is None:
            total = sum(observed)
            expected = [total / len(observed)] * len(observed)
            print(f"Using equal expected frequencies: {expected[0]:.2f} for each category")
        else:
            if len(expected) != len(observed):
                raise ValueError("Observed and expected must have the same length")
            
            if not validate_categorical_data(expected):
                raise ValueError("Expected data must be non-negative numbers")
        
        # Get hypotheses
        hypotheses = get_hypothesis_input("Chi-Square Goodness of Fit Test")
        
        # Validate assumptions
        warnings = []
        
        if any(e < 5 for e in expected):
            warnings.append("Some expected frequencies < 5. Results may be unreliable.")
        
        if sum(observed) < 30:
            warnings.append("Total sample size < 30. Consider exact tests.")
        
        print_assumption_warnings(warnings)
        
        # Perform test
        try:
            statistic, p_value = stats.chisquare(observed, expected)
        except ValueError as e:
            return {
                'test_name': test_name,
                'error': f"Test failed: {e}"
            }
        
        # Calculate additional statistics
        df = len(observed) - 1
        total_observed = sum(observed)
        
        # Effect size (Cramér's V for goodness of fit)
        cramers_v = np.sqrt(statistic / (total_observed * df)) if df > 0 else 0
        
        # Calculate residuals
        residuals = [(obs - exp) / np.sqrt(exp) for obs, exp in zip(observed, expected)]
        
        results = {
            'test_name': test_name,
            'chi2_statistic': statistic,
            'p_value': p_value,
            'degrees_of_freedom': df,
            'observed_frequencies': observed,
            'expected_frequencies': expected,
            'cramers_v': cramers_v,
            'standardized_residuals': residuals,
            'interpretation': f"{'Reject' if p_value < alpha else 'Fail to reject'} H0 at α = {alpha}"
        }
        
        # Print frequency table
        print("\nFrequency Table:")
        print("-" * 50)
        print(f"{'Category':<10} {'Observed':<10} {'Expected':<10} {'Residual':<10}")
        print("-" * 50)
        for i, (obs, exp, res) in enumerate(zip(observed, expected, residuals), 1):
            print(f"Cat {i:<5} {obs:<10} {exp:<10.2f} {res:<10.2f}")
        print("-" * 50)
        
        print_test_results(results, hypotheses)
        
        return results
    
    @staticmethod
    def chi_square_association(contingency_table: List[List[float]], 
                              alpha: float = 0.05) -> Dict[str, Any]:
        """
        Chi-Square Test of Association (Independence)
        
        Args:
            contingency_table: 2D list representing contingency table
            alpha: Significance level
            
        Returns:
            Dictionary with test results
        """
        test_name = "Chi-Square Test of Association"
        
        # Convert to numpy array and validate
        table = np.array(contingency_table, dtype=float)
        
        if table.ndim != 2:
            raise ValueError("Contingency table must be 2-dimensional")
        
        if table.shape[0] < 2 or table.shape[1] < 2:
            raise ValueError("Contingency table must be at least 2x2")
        
        if np.any(table < 0):
            raise ValueError("All frequencies must be non-negative")
        
        # Get hypotheses
        hypotheses = get_hypothesis_input("Chi-Square Test of Association")
        
        # Validate assumptions
        warnings = []
        
        # Calculate expected frequencies
        row_totals = table.sum(axis=1)
        col_totals = table.sum(axis=0)
        total = table.sum()
        
        expected = np.outer(row_totals, col_totals) / total
        
        if np.any(expected < 5):
            warnings.append("Some expected frequencies < 5. Consider Fisher's exact test.")
        
        if total < 30:
            warnings.append("Total sample size < 30. Results may be unreliable.")
        
        print_assumption_warnings(warnings)
        
        # Perform test
        try:
            statistic, p_value, dof, expected_freq = stats.chi2_contingency(table)
        except ValueError as e:
            return {
                'test_name': test_name,
                'error': f"Test failed: {e}"
            }
        
        # Calculate additional statistics
        rows, cols = table.shape
        
        # Effect size measures
        # Cramér's V
        cramers_v = np.sqrt(statistic / (total * min(rows - 1, cols - 1)))
        
        # Phi coefficient (for 2x2 tables)
        phi = np.sqrt(statistic / total) if rows == 2 and cols == 2 else None
        
        # Contingency coefficient
        contingency_coeff = np.sqrt(statistic / (statistic + total))
        
        # Calculate standardized residuals
        std_residuals = (table - expected_freq) / np.sqrt(expected_freq)
        
        results = {
            'test_name': test_name,
            'chi2_statistic': statistic,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'cramers_v': cramers_v,
            'contingency_coefficient': contingency_coeff,
            'interpretation': f"{'Reject' if p_value < alpha else 'Fail to reject'} H0 at α = {alpha}"
        }
        
        if phi is not None:
            results['phi_coefficient'] = phi
        
        # Print contingency table with margins
        print("\nContingency Table:")
        print("-" * (cols * 12 + 15))
        
        # Header
        header = "Row\\Col".ljust(10)
        for j in range(cols):
            header += f"Col{j+1}".rjust(10)
        header += "Total".rjust(12)
        print(header)
        print("-" * (cols * 12 + 15))
        
        # Data rows
        for i in range(rows):
            row_str = f"Row{i+1}".ljust(10)
            for j in range(cols):
                row_str += f"{table[i,j]:.0f}".rjust(10)
            row_str += f"{row_totals[i]:.0f}".rjust(12)
            print(row_str)
        
        # Column totals
        total_str = "Total".ljust(10)
        for j in range(cols):
            total_str += f"{col_totals[j]:.0f}".rjust(10)
        total_str += f"{total:.0f}".rjust(12)
        print("-" * (cols * 12 + 15))
        print(total_str)
        print("-" * (cols * 12 + 15))
        
        # Print expected frequencies
        print("\nExpected Frequencies:")
        print("-" * (cols * 12 + 10))
        for i in range(rows):
            row_str = f"Row{i+1}".ljust(10)
            for j in range(cols):
                row_str += f"{expected_freq[i,j]:.2f}".rjust(10)
            print(row_str)
        print("-" * (cols * 12 + 10))
        
        print_test_results(results, hypotheses)
        
        return results
    
    @staticmethod
    def input_contingency_table() -> List[List[float]]:
        """
        Interactive input for contingency table
        
        Returns:
            2D list representing contingency table
        """
        print("\nEnter contingency table data:")
        print("Format: rows separated by semicolons, values separated by commas")
        print("Example for 2x3 table: '10,15,20;25,30,35'")
        
        while True:
            table_str = input("Enter table: ").strip()
            
            is_valid, table, error_msg = validate_contingency_table(table_str)
            
            if is_valid:
                return table
            else:
                print(f"Error: {error_msg}")
                print("Please try again.")
