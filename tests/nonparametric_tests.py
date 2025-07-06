#Keith Ngamphon McKenzie
#keith@mckenzie.page
#https://mckenzie.page
#Python Simple Statistical Tests

import numpy as np
import scipy.stats as stats
from typing import List, Dict, Any, Tuple, Optional
from utils.validators import (validate_minimum_sample_size, validate_equal_sample_sizes,
                            get_hypothesis_input)
from utils.formatters import print_test_results, print_assumption_warnings, print_data_summary

class NonParametricTests:
    """Class containing non-parametric statistical tests"""
    
    @staticmethod
    def wilcoxon_signed_rank_test(data1: List[float], data2: Optional[List[float]] = None,
                                 alpha: float = 0.05) -> Dict[str, Any]:
        """
        Wilcoxon Signed-Rank Test
        Can be used for one-sample or paired-sample testing
        
        Args:
            data1: First sample or differences (for one-sample test)
            data2: Second sample (for paired test) or None
            alpha: Significance level
            
        Returns:
            Dictionary with test results
        """
        if data2 is None:
            # One-sample test against median of 0
            test_name = "One-Sample Wilcoxon Signed-Rank Test"
            differences = data1
            
            # Get hypothesized median
            while True:
                try:
                    hyp_median = float(input("Enter hypothesized median (default 0): ") or "0")
                    break
                except ValueError:
                    print("Please enter a valid number.")
            
            differences = [x - hyp_median for x in data1]
            
        else:
            # Paired-sample test
            test_name = "Paired-Sample Wilcoxon Signed-Rank Test"
            
            if not validate_equal_sample_sizes(data1, data2):
                raise ValueError("Paired test requires equal sample sizes")
            
            differences = [x2 - x1 for x1, x2 in zip(data1, data2)]
        
        # Get hypotheses
        hypotheses = get_hypothesis_input("Wilcoxon Signed-Rank Test")
        
        # Validate assumptions
        warnings = []
        
        if not validate_minimum_sample_size(differences, 6):
            warnings.append("Small sample size. Consider exact p-values.")
        
        # Remove zero differences
        non_zero_diffs = [d for d in differences if d != 0]
        
        if len(non_zero_diffs) < len(differences):
            warnings.append(f"Removed {len(differences) - len(non_zero_diffs)} zero differences.")
        
        if len(non_zero_diffs) == 0:
            warnings.append("All differences are zero. Cannot perform test.")
            
        print_assumption_warnings(warnings)
        
        if len(non_zero_diffs) == 0:
            return {
                'test_name': test_name,
                'error': "Cannot perform test - all differences are zero"
            }
        
        # Perform test
        try:
            statistic, p_value = stats.wilcoxon(non_zero_diffs, alternative='two-sided')
        except ValueError as e:
            return {
                'test_name': test_name,
                'error': f"Test failed: {e}"
            }
        
        # Calculate additional statistics
        n = len(non_zero_diffs)
        median_diff = np.median(differences)
        
        # Effect size (r = Z / sqrt(N))
        if n > 10:
            z_score = (statistic - n*(n+1)/4) / np.sqrt(n*(n+1)*(2*n+1)/24)
            effect_size = abs(z_score) / np.sqrt(n)
        else:
            effect_size = None
        
        results = {
            'test_name': test_name,
            'statistic': statistic,
            'p_value': p_value,
            'n_pairs': n,
            'median_difference': median_diff,
            'interpretation': f"{'Reject' if p_value < alpha else 'Fail to reject'} H0 at α = {alpha}"
        }
        
        if effect_size is not None:
            results['effect_size'] = effect_size
        
        if data2 is None:
            print_data_summary(data1, "Sample")
        else:
            print_data_summary(data1, "Sample 1")
            print_data_summary(data2, "Sample 2")
        
        print_data_summary(differences, "Differences")
        print_test_results(results, hypotheses)
        
        return results
    
    @staticmethod  
    def one_sample_wilcoxon_test(data: List[float], hypothesized_median: float = 0,
                               alpha: float = 0.05) -> Dict[str, Any]:
        """
        One-Sample Wilcoxon Signed-Rank Test
        
        Args:
            data: Sample data
            hypothesized_median: Hypothesized population median
            alpha: Significance level
            
        Returns:
            Dictionary with test results
        """
        test_name = "One-Sample Wilcoxon Signed-Rank Test"
        
        # Get hypotheses
        hypotheses = get_hypothesis_input("One-Sample Wilcoxon Signed-Rank Test")
        
        # Calculate differences from hypothesized median
        differences = [x - hypothesized_median for x in data]
        
        # Validate assumptions
        warnings = []
        
        if not validate_minimum_sample_size(differences, 6):
            warnings.append("Small sample size. Consider exact p-values.")
        
        # Remove zero differences
        non_zero_diffs = [d for d in differences if d != 0]
        
        if len(non_zero_diffs) < len(differences):
            warnings.append(f"Removed {len(differences) - len(non_zero_diffs)} zero differences.")
        
        if len(non_zero_diffs) == 0:
            warnings.append("All differences are zero. Cannot perform test.")
            
        print_assumption_warnings(warnings)
        
        if len(non_zero_diffs) == 0:
            return {
                'test_name': test_name,
                'error': "Cannot perform test - all differences are zero"
            }
        
        # Perform test
        try:
            statistic, p_value = stats.wilcoxon(non_zero_diffs, alternative='two-sided')
        except ValueError as e:
            return {
                'test_name': test_name,
                'error': f"Test failed: {e}"
            }
        
        # Calculate additional statistics
        n = len(non_zero_diffs)
        median_diff = np.median(differences)
        sample_median = np.median(data)
        
        # Effect size (r = Z / sqrt(N))
        if n > 10:
            z_score = (statistic - n*(n+1)/4) / np.sqrt(n*(n+1)*(2*n+1)/24)
            effect_size = abs(z_score) / np.sqrt(n)
        else:
            effect_size = None
        
        results = {
            'test_name': test_name,
            'statistic': statistic,
            'p_value': p_value,
            'n_observations': n,
            'sample_median': sample_median,
            'hypothesized_median': hypothesized_median,
            'median_difference': median_diff,
            'interpretation': f"{'Reject' if p_value < alpha else 'Fail to reject'} H0 at α = {alpha}"
        }
        
        if effect_size is not None:
            results['effect_size'] = effect_size
        
        print_data_summary(data, "Sample")
        print_data_summary(differences, "Differences from hypothesized median")
        print_test_results(results, hypotheses)
        
        return results

    @staticmethod
    def mann_whitney_test(data1: List[float], data2: List[float], 
                         alpha: float = 0.05) -> Dict[str, Any]:
        """
        Mann-Whitney U Test (also known as Wilcoxon Rank-Sum Test)
        
        Args:
            data1: First independent sample
            data2: Second independent sample
            alpha: Significance level
            
        Returns:
            Dictionary with test results
        """
        test_name = "Mann-Whitney U Test"
        
        # Get hypotheses
        hypotheses = get_hypothesis_input("Mann-Whitney U Test")
        
        # Validate assumptions
        warnings = []
        
        if not validate_minimum_sample_size(data1, 3) or not validate_minimum_sample_size(data2, 3):
            warnings.append("Very small sample sizes. Results may be unreliable.")
        
        print_assumption_warnings(warnings)
        
        # Perform test
        try:
            statistic, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
        except ValueError as e:
            return {
                'test_name': test_name,
                'error': f"Test failed: {e}"
            }
        
        # Calculate additional statistics
        n1, n2 = len(data1), len(data2)
        median1, median2 = np.median(data1), np.median(data2)
        
        # Effect size (r = Z / sqrt(N))
        n_total = n1 + n2
        expected_u = n1 * n2 / 2
        std_u = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
        z_score = (statistic - expected_u) / std_u
        effect_size = abs(z_score) / np.sqrt(n_total)
        
        # Probability of superiority
        prob_superiority = statistic / (n1 * n2)
        
        results = {
            'test_name': test_name,
            'u_statistic': statistic,
            'p_value': p_value,
            'n1': n1,
            'n2': n2,
            'median_1': median1,
            'median_2': median2,
            'effect_size': effect_size,
            'probability_superiority': prob_superiority,
            'interpretation': f"{'Reject' if p_value < alpha else 'Fail to reject'} H0 at α = {alpha}"
        }
        
        print_data_summary(data1, "Sample 1")
        print_data_summary(data2, "Sample 2")
        print_test_results(results, hypotheses)
        
        return results
    
    @staticmethod
    def kruskal_wallis_test(*groups: List[float], alpha: float = 0.05) -> Dict[str, Any]:
        """
        Kruskal-Wallis H Test (non-parametric one-way ANOVA)
        
        Args:
            groups: Variable number of independent groups
            alpha: Significance level
            
        Returns:
            Dictionary with test results
        """
        test_name = "Kruskal-Wallis H Test"
        
        if len(groups) < 2:
            raise ValueError("Kruskal-Wallis test requires at least 2 groups")
        
        # Get hypotheses
        hypotheses = get_hypothesis_input("Kruskal-Wallis Test")
        
        # Validate assumptions
        warnings = []
        
        for i, group in enumerate(groups, 1):
            if not validate_minimum_sample_size(group, 5):
                warnings.append(f"Group {i} has small sample size.")
        
        print_assumption_warnings(warnings)
        
        # Perform test
        try:
            statistic, p_value = stats.kruskal(*groups)
        except ValueError as e:
            return {
                'test_name': test_name,
                'error': f"Test failed: {e}"
            }
        
        # Calculate additional statistics
        k = len(groups)  # number of groups
        n_total = sum(len(group) for group in groups)
        group_medians = [np.median(group) for group in groups]
        
        # Effect size (eta-squared approximation)
        eta_squared = (statistic - k + 1) / (n_total - k)
        eta_squared = max(0, eta_squared)  # Ensure non-negative
        
        results = {
            'test_name': test_name,
            'h_statistic': statistic,
            'p_value': p_value,
            'degrees_of_freedom': k - 1,
            'n_groups': k,
            'total_n': n_total,
            'group_medians': group_medians,
            'eta_squared': eta_squared,
            'interpretation': f"{'Reject' if p_value < alpha else 'Fail to reject'} H0 at α = {alpha}"
        }
        
        for i, group in enumerate(groups, 1):
            print_data_summary(group, f"Group {i}")
        
        print_test_results(results, hypotheses)
        
        return results
