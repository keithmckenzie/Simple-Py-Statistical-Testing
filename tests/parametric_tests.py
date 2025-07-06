#Keith Ngamphon McKenzie
#keith@mckenzie.page
#https://mckenzie.page
#Python Simple Statistical Tests

import numpy as np
import scipy.stats as stats
from typing import List, Dict, Any, Tuple
from utils.validators import (validate_minimum_sample_size, validate_equal_sample_sizes,
                            validate_normality_assumption, get_hypothesis_input)
from utils.formatters import print_test_results, print_assumption_warnings, print_data_summary

class ParametricTests:
    """Class containing parametric statistical tests"""
    
    @staticmethod
    def students_t_test(data: List[float], population_mean: float = 0.0, 
                       alpha: float = 0.05) -> Dict[str, Any]:
        """
        One-sample Student's t-test
        
        Args:
            data: Sample data
            population_mean: Hypothesized population mean
            alpha: Significance level
            
        Returns:
            Dictionary with test results
        """
        print_test_results.__name__ = "One-Sample Student's t-test"
        
        # Get population mean if not provided
        if population_mean == 0.0:
            while True:
                try:
                    population_mean = float(input("Enter hypothesized population mean (default 0): ") or "0")
                    break
                except ValueError:
                    print("Please enter a valid number.")
        
        # Get hypotheses
        hypotheses = get_hypothesis_input("One-Sample t-test")
        
        # Validate assumptions
        warnings = []
        
        if not validate_minimum_sample_size(data, 5):
            warnings.append("Very small sample size. Results may be unreliable.")
        
        is_normal, norm_msg = validate_normality_assumption(data)
        if not is_normal:
            warnings.append(norm_msg)
        
        print_assumption_warnings(warnings)
        
        # Perform test
        t_statistic, p_value = stats.ttest_1samp(data, population_mean)
        
        # Calculate additional statistics
        n = len(data)
        sample_mean = np.mean(data)
        sample_std = np.std(data, ddof=1)
        se = sample_std / np.sqrt(n)
        
        # Cohen's d (effect size)
        cohens_d = (sample_mean - population_mean) / sample_std
        
        # Confidence interval for mean
        df = n - 1
        t_critical = stats.t.ppf(1 - alpha/2, df)
        ci_lower = sample_mean - t_critical * se
        ci_upper = sample_mean + t_critical * se
        
        results = {
            'test_name': "One-Sample Student's t-test",
            'statistic': t_statistic,
            'p_value': p_value,
            'degrees_of_freedom': df,
            'sample_mean': sample_mean,
            'hypothesized_mean': population_mean,
            'effect_size': cohens_d,
            'confidence_interval': (ci_lower, ci_upper),
            'interpretation': f"{'Reject' if p_value < alpha else 'Fail to reject'} H0 at α = {alpha}"
        }
        
        print_data_summary(data, "Sample")
        print_test_results(results, hypotheses)
        
        return results
    
    @staticmethod
    def paired_t_test(data1: List[float], data2: List[float], 
                     alpha: float = 0.05) -> Dict[str, Any]:
        """
        Paired samples t-test
        
        Args:
            data1: First sample (e.g., pre-treatment)
            data2: Second sample (e.g., post-treatment)
            alpha: Significance level
            
        Returns:
            Dictionary with test results
        """
        print_test_results.__name__ = "Paired Samples t-test"
        
        # Validate equal sample sizes
        if not validate_equal_sample_sizes(data1, data2):
            raise ValueError("Paired t-test requires equal sample sizes")
        
        # Get hypotheses
        hypotheses = get_hypothesis_input("Paired t-test")
        
        # Calculate differences
        differences = [x2 - x1 for x1, x2 in zip(data1, data2)]
        
        # Validate assumptions
        warnings = []
        
        if not validate_minimum_sample_size(differences, 5):
            warnings.append("Very small sample size. Results may be unreliable.")
        
        is_normal, norm_msg = validate_normality_assumption(differences)
        if not is_normal:
            warnings.append(norm_msg)
        
        print_assumption_warnings(warnings)
        
        # Perform test
        t_statistic, p_value = stats.ttest_rel(data1, data2)
        
        # Calculate additional statistics
        n = len(differences)
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)
        se_diff = std_diff / np.sqrt(n)
        
        # Effect size (Cohen's d for paired samples)
        cohens_d = mean_diff / std_diff
        
        # Confidence interval for mean difference
        df = n - 1
        t_critical = stats.t.ppf(1 - alpha/2, df)
        ci_lower = mean_diff - t_critical * se_diff
        ci_upper = mean_diff + t_critical * se_diff
        
        results = {
            'test_name': "Paired Samples t-test",
            'statistic': t_statistic,
            'p_value': p_value,
            'degrees_of_freedom': df,
            'mean_difference': mean_diff,
            'effect_size': cohens_d,
            'confidence_interval': (ci_lower, ci_upper),
            'interpretation': f"{'Reject' if p_value < alpha else 'Fail to reject'} H0 at α = {alpha}"
        }
        
        print_data_summary(data1, "Sample 1")
        print_data_summary(data2, "Sample 2")
        print_data_summary(differences, "Differences")
        print_test_results(results, hypotheses)
        
        return results
    
    @staticmethod
    def independent_t_test(data1: List[float], data2: List[float], 
                          alpha: float = 0.05, equal_var: bool = True) -> Dict[str, Any]:
        """
        Independent samples t-test (unpaired two-sample t-test)
        
        Args:
            data1: First independent sample
            data2: Second independent sample
            alpha: Significance level
            equal_var: Whether to assume equal variances (None for automatic)
            
        Returns:
            Dictionary with test results
        """
        test_name = "Independent Samples t-test"
        
        # Get hypotheses
        hypotheses = get_hypothesis_input("Independent samples t-test")
        
        # Validate assumptions
        warnings = []
        
        if not validate_minimum_sample_size(data1, 5) or not validate_minimum_sample_size(data2, 5):
            warnings.append("Small sample sizes. Results may be unreliable.")
        
        is_normal1, norm_msg1 = validate_normality_assumption(data1)
        is_normal2, norm_msg2 = validate_normality_assumption(data2)
        
        if not is_normal1:
            warnings.append(f"Sample 1: {norm_msg1}")
        if not is_normal2:
            warnings.append(f"Sample 2: {norm_msg2}")
        
        # Test for equal variances automatically
        # Levene's test for equal variances
        _, levene_p = stats.levene(data1, data2)
        auto_equal_var = levene_p > 0.05  # Assume equal if p > 0.05
        
        if not auto_equal_var:
            warnings.append("Unequal variances detected. Using Welch's t-test.")
            equal_var = False
        
        print_assumption_warnings(warnings)
        
        # Perform test
        t_statistic, p_value = stats.ttest_ind(data1, data2, equal_var=equal_var)
        
        # Calculate additional statistics
        n1, n2 = len(data1), len(data2)
        mean1, mean2 = np.mean(data1), np.mean(data2)
        std1, std2 = np.std(data1, ddof=1), np.std(data2, ddof=1)
        
        # Pooled standard deviation (for equal variances)
        if equal_var:
            pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2))
            se_diff = pooled_std * np.sqrt(1/n1 + 1/n2)
            df = n1 + n2 - 2
        else:
            # Welch's t-test
            se1, se2 = std1/np.sqrt(n1), std2/np.sqrt(n2)
            se_diff = np.sqrt(se1**2 + se2**2)
            # Welch-Satterthwaite equation for degrees of freedom
            df = (se1**2 + se2**2)**2 / (se1**4/(n1-1) + se2**4/(n2-1))
            pooled_std = None
        
        # Effect size (Cohen's d)
        if equal_var and pooled_std is not None and pooled_std > 0:
            cohens_d = (mean1 - mean2) / pooled_std
        elif std1 > 0 and std2 > 0:
            # Use pooled standard deviation estimate
            cohens_d = (mean1 - mean2) / np.sqrt((std1**2 + std2**2) / 2)
        else:
            cohens_d = 0
        
        # Confidence interval for mean difference
        t_critical = stats.t.ppf(1 - alpha/2, df)
        mean_diff = mean1 - mean2
        ci_lower = mean_diff - t_critical * se_diff
        ci_upper = mean_diff + t_critical * se_diff
        
        results = {
            'test_name': test_name,
            'statistic': t_statistic,
            'p_value': p_value,
            'degrees_of_freedom': df,
            'mean_1': mean1,
            'mean_2': mean2,
            'mean_difference': mean_diff,
            'effect_size': cohens_d,
            'equal_variances_assumed': equal_var,
            'confidence_interval': (ci_lower, ci_upper),
            'interpretation': f"{'Reject' if p_value < alpha else 'Fail to reject'} H0 at α = {alpha}"
        }
        
        if pooled_std is not None:
            results['pooled_std'] = pooled_std
        
        print_data_summary(data1, "Sample 1")
        print_data_summary(data2, "Sample 2")
        print_test_results(results, hypotheses)
        
        return results
    
    @staticmethod
    def f_test(data1: List[float], data2: List[float], 
              alpha: float = 0.05) -> Dict[str, Any]:
        """
        F-test for equality of variances
        
        Args:
            data1: First sample
            data2: Second sample
            alpha: Significance level
            
        Returns:
            Dictionary with test results
        """
        print_test_results.__name__ = "F-test for Equality of Variances"
        
        # Get hypotheses
        hypotheses = get_hypothesis_input("F-test for equality of variances")
        
        # Validate assumptions
        warnings = []
        
        if not validate_minimum_sample_size(data1, 3) or not validate_minimum_sample_size(data2, 3):
            warnings.append("Very small sample sizes. Results may be unreliable.")
        
        is_normal1, norm_msg1 = validate_normality_assumption(data1)
        is_normal2, norm_msg2 = validate_normality_assumption(data2)
        
        if not is_normal1:
            warnings.append(f"Sample 1: {norm_msg1}")
        if not is_normal2:
            warnings.append(f"Sample 2: {norm_msg2}")
        
        print_assumption_warnings(warnings)
        
        # Calculate variances
        var1 = np.var(data1, ddof=1)
        var2 = np.var(data2, ddof=1)
        
        # F-statistic (larger variance in numerator)
        if var1 >= var2:
            f_statistic = var1 / var2
            df1 = len(data1) - 1
            df2 = len(data2) - 1
        else:
            f_statistic = var2 / var1
            df1 = len(data2) - 1
            df2 = len(data1) - 1
        
        # Calculate p-value (two-tailed)
        p_value = 2 * (1 - stats.f.cdf(f_statistic, df1, df2))
        
        # Confidence interval for variance ratio
        f_lower = stats.f.ppf(alpha/2, df1, df2)
        f_upper = stats.f.ppf(1 - alpha/2, df1, df2)
        
        if var1 >= var2:
            ci_lower = (var1/var2) / f_upper
            ci_upper = (var1/var2) / f_lower
        else:
            ci_lower = (var2/var1) / f_upper
            ci_upper = (var2/var1) / f_lower
        
        results = {
            'test_name': "F-test for Equality of Variances",
            'f_statistic': f_statistic,
            'p_value': p_value,
            'df1': df1,
            'df2': df2,
            'variance_1': var1,
            'variance_2': var2,
            'variance_ratio': var1/var2 if var1 >= var2 else var2/var1,
            'confidence_interval': (ci_lower, ci_upper),
            'interpretation': f"{'Reject' if p_value < alpha else 'Fail to reject'} H0 at α = {alpha}"
        }
        
        print_data_summary(data1, "Sample 1")
        print_data_summary(data2, "Sample 2")
        print_test_results(results, hypotheses)
        
        return results
    
    @staticmethod
    def one_way_anova(*groups: List[float], alpha: float = 0.05) -> Dict[str, Any]:
        """
        One-way ANOVA
        
        Args:
            groups: Variable number of group data
            alpha: Significance level
            
        Returns:
            Dictionary with test results
        """
        print_test_results.__name__ = "One-Way ANOVA"
        
        if len(groups) < 2:
            raise ValueError("ANOVA requires at least 2 groups")
        
        # Get hypotheses
        hypotheses = get_hypothesis_input("One-way ANOVA")
        
        # Validate assumptions
        warnings = []
        
        for i, group in enumerate(groups, 1):
            if not validate_minimum_sample_size(group, 3):
                warnings.append(f"Group {i} has very small sample size.")
            
            is_normal, norm_msg = validate_normality_assumption(group)
            if not is_normal:
                warnings.append(f"Group {i}: {norm_msg}")
        
        print_assumption_warnings(warnings)
        
        # Perform ANOVA
        f_statistic, p_value = stats.f_oneway(*groups)
        
        # Calculate additional statistics
        k = len(groups)  # number of groups
        n_total = sum(len(group) for group in groups)
        df_between = k - 1
        df_within = n_total - k
        
        # Calculate group means and overall mean
        group_means = [np.mean(group) for group in groups]
        overall_mean = np.mean([x for group in groups for x in group])
        
        # Calculate eta-squared (effect size)
        ss_between = sum(len(group) * (mean - overall_mean)**2 
                        for group, mean in zip(groups, group_means))
        ss_total = sum((x - overall_mean)**2 for group in groups for x in group)
        
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        
        results = {
            'test_name': "One-Way ANOVA",
            'f_statistic': f_statistic,
            'p_value': p_value,
            'df_between': df_between,
            'df_within': df_within,
            'eta_squared': eta_squared,
            'group_means': group_means,
            'overall_mean': overall_mean,
            'interpretation': f"{'Reject' if p_value < alpha else 'Fail to reject'} H0 at α = {alpha}"
        }
        
        for i, group in enumerate(groups, 1):
            print_data_summary(group, f"Group {i}")
        
        print_test_results(results, hypotheses)
        
        return results
