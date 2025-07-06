#Keith Ngamphon McKenzie
#keith@mckenzie.page
#https://mckenzie.page
#Python Simple Statistical Tests

import numpy as np
import scipy.stats as stats
from typing import List, Dict, Any, Tuple
from utils.validators import validate_correlation_data, get_hypothesis_input
from utils.formatters import (print_test_results, print_assumption_warnings, 
                            print_data_summary, format_regression_results)

class CorrelationTests:
    """Class containing correlation and regression tests"""
    
    @staticmethod
    def spearmans_rank_correlation(x_data: List[float], y_data: List[float],
                                  alpha: float = 0.05) -> Dict[str, Any]:
        """
        Spearman's Rank Correlation Test
        
        Args:
            x_data: X variable data
            y_data: Y variable data
            alpha: Significance level
            
        Returns:
            Dictionary with test results
        """
        test_name = "Spearman's Rank Correlation"
        
        # Validate data
        is_valid, error_msg = validate_correlation_data(x_data, y_data)
        if not is_valid:
            raise ValueError(error_msg)
        
        # Get hypotheses
        hypotheses = get_hypothesis_input("Spearman's Rank Correlation")
        
        # Validate assumptions
        warnings = []
        
        if len(x_data) < 10:
            warnings.append("Small sample size. Results may be unreliable.")
        
        # Check for ties
        x_ties = len(x_data) - len(set(x_data))
        y_ties = len(y_data) - len(set(y_data))
        
        if x_ties > len(x_data) * 0.1:
            warnings.append("Many ties in X variable. Consider alternative methods.")
        if y_ties > len(y_data) * 0.1:
            warnings.append("Many ties in Y variable. Consider alternative methods.")
        
        print_assumption_warnings(warnings)
        
        # Perform test
        try:
            correlation, p_value = stats.spearmanr(x_data, y_data)
        except ValueError as e:
            return {
                'test_name': test_name,
                'error': f"Test failed: {e}"
            }
        
        # Calculate additional statistics
        n = len(x_data)
        
        # Confidence interval for correlation (Fisher's z-transformation approximation)
        if abs(correlation) < 0.999:
            z_r = 0.5 * np.log((1 + correlation) / (1 - correlation))
            se_z = 1 / np.sqrt(n - 3)
            z_critical = stats.norm.ppf(1 - alpha/2)
            
            z_lower = z_r - z_critical * se_z
            z_upper = z_r + z_critical * se_z
            
            ci_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
            ci_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
        else:
            ci_lower, ci_upper = None, None
        
        # Interpretation of correlation strength
        abs_corr = abs(correlation)
        if abs_corr < 0.1:
            strength = "negligible"
        elif abs_corr < 0.3:
            strength = "weak"
        elif abs_corr < 0.5:
            strength = "moderate"
        elif abs_corr < 0.7:
            strength = "strong"
        else:
            strength = "very strong"
        
        results = {
            'test_name': test_name,
            'correlation_coefficient': correlation,
            'p_value': p_value,
            'n_pairs': n,
            'correlation_strength': strength,
            'interpretation': f"{'Reject' if p_value < alpha else 'Fail to reject'} H0 at α = {alpha}"
        }
        
        if ci_lower is not None and ci_upper is not None:
            results['confidence_interval'] = (ci_lower, ci_upper)
        
        print_data_summary(x_data, "X Variable")
        print_data_summary(y_data, "Y Variable")
        print_test_results(results, hypotheses)
        
        return results
    
    @staticmethod
    def coefficient_of_determination(x_data: List[float], y_data: List[float],
                                   alpha: float = 0.05) -> Dict[str, Any]:
        """
        Coefficient of Determination (R-squared) with significance test
        
        Args:
            x_data: X variable (predictor)
            y_data: Y variable (response)
            alpha: Significance level
            
        Returns:
            Dictionary with test results
        """
        test_name = "Coefficient of Determination"
        
        # Validate data
        is_valid, error_msg = validate_correlation_data(x_data, y_data)
        if not is_valid:
            raise ValueError(error_msg)
        
        # Get hypotheses
        hypotheses = get_hypothesis_input("Coefficient of Determination")
        
        # Convert to numpy arrays
        x = np.array(x_data)
        y = np.array(y_data)
        n = len(x)
        
        # Calculate linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Calculate R-squared
        r_squared = r_value ** 2
        
        # Calculate additional regression statistics
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)  # Sum of squares of residuals
        ss_tot = np.sum((y - np.mean(y)) ** 2)  # Total sum of squares
        
        # Mean squared error
        mse = ss_res / (n - 2)
        rmse = np.sqrt(mse)
        
        # F-statistic for overall model significance
        if ss_tot > 0:
            f_statistic = (ss_tot - ss_res) / (ss_res / (n - 2))
        else:
            f_statistic = 0
        
        # Adjusted R-squared
        adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - 2)
        
        # Confidence interval for R-squared (approximate)
        if r_squared > 0 and n > 10:
            # Fisher's z-transformation for correlation
            z_r = 0.5 * np.log((1 + abs(r_value)) / (1 - abs(r_value)))
            se_z = 1 / np.sqrt(n - 3)
            z_critical = stats.norm.ppf(1 - alpha/2)
            
            z_lower = z_r - z_critical * se_z
            z_upper = z_r + z_critical * se_z
            
            r_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
            r_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
            
            r2_lower = r_lower ** 2
            r2_upper = r_upper ** 2
        else:
            r2_lower, r2_upper = None, None
        
        results = {
            'test_name': test_name,
            'r_squared': r_squared,
            'adjusted_r_squared': adj_r_squared,
            'correlation_coefficient': r_value,
            'p_value': p_value,
            'f_statistic': f_statistic,
            'slope': slope,
            'intercept': intercept,
            'rmse': rmse,
            'n_observations': n,
            'interpretation': f"{'Reject' if p_value < alpha else 'Fail to reject'} H0 at α = {alpha}"
        }
        
        if r2_lower is not None and r2_upper is not None:
            results['r2_confidence_interval'] = (r2_lower, r2_upper)
        
        # Print regression equation and fit statistics
        print(f"\nRegression Analysis:")
        print(f"Equation: y = {slope:.4f}x + {intercept:.4f}")
        print(f"R-squared: {r_squared:.4f} ({r_squared*100:.2f}% of variance explained)")
        print(f"Adjusted R-squared: {adj_r_squared:.4f}")
        print(f"RMSE: {rmse:.4f}")
        
        print_data_summary(x_data, "X Variable (Predictor)")
        print_data_summary(y_data, "Y Variable (Response)")
        print_test_results(results, hypotheses)
        
        return results
    
    @staticmethod
    def linear_regression_tests(x_data: List[float], y_data: List[float],
                               alpha: float = 0.05) -> Dict[str, Any]:
        """
        Comprehensive Linear Regression Analysis with multiple tests
        
        Args:
            x_data: X variable (predictor)
            y_data: Y variable (response)
            alpha: Significance level
            
        Returns:
            Dictionary with comprehensive regression results
        """
        test_name = "Linear Regression Analysis"
        
        # Validate data
        is_valid, error_msg = validate_correlation_data(x_data, y_data)
        if not is_valid:
            raise ValueError(error_msg)
        
        # Get hypotheses
        hypotheses = get_hypothesis_input("Linear Regression Analysis")
        
        # Convert to numpy arrays
        x = np.array(x_data)
        y = np.array(y_data)
        n = len(x)
        
        # Validate assumptions
        warnings = []
        
        if n < 10:
            warnings.append("Small sample size. Results may be unreliable.")
        
        print_assumption_warnings(warnings)
        
        # Perform regression analysis
        slope, intercept, r_value, p_value_overall, std_err = stats.linregress(x, y)
        
        # Calculate comprehensive statistics
        y_pred = slope * x + intercept
        residuals = y - y_pred
        
        # Sums of squares
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        ss_reg = ss_tot - ss_res
        
        # Degrees of freedom
        df_reg = 1  # One predictor
        df_res = n - 2
        df_tot = n - 1
        
        # Mean squares
        ms_reg = ss_reg / df_reg
        ms_res = ss_res / df_res if df_res > 0 else 0
        
        # F-statistic for overall model
        f_statistic = ms_reg / ms_res if ms_res > 0 else 0
        f_p_value = 1 - stats.f.cdf(f_statistic, df_reg, df_res) if ms_res > 0 else 1
        
        # R-squared and adjusted R-squared
        r_squared = r_value ** 2
        adj_r_squared = 1 - (ms_res / (ss_tot / df_tot)) if ss_tot > 0 else 0
        
        # Standard error of regression
        se_regression = np.sqrt(ms_res)
        
        # Standard error of slope
        se_slope = se_regression / np.sqrt(np.sum((x - np.mean(x)) ** 2))
        
        # t-test for slope
        t_statistic = slope / se_slope if se_slope > 0 else 0
        t_p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), df_res))
        
        # Confidence interval for slope
        t_critical = stats.t.ppf(1 - alpha/2, df_res)
        slope_ci_lower = slope - t_critical * se_slope
        slope_ci_upper = slope + t_critical * se_slope
        
        # Prediction intervals (for mean response)
        x_mean = np.mean(x)
        se_pred_mean = se_regression * np.sqrt(1/n + (x - x_mean)**2 / np.sum((x - x_mean)**2))
        
        # Durbin-Watson test for autocorrelation (approximate)
        if n > 2:
            dw_statistic = np.sum(np.diff(residuals)**2) / np.sum(residuals**2)
        else:
            dw_statistic = None
        
        results = {
            'test_name': test_name,
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_squared,
            'adjusted_r_squared': adj_r_squared,
            'correlation_coefficient': r_value,
            'f_statistic': f_statistic,
            'f_p_value': f_p_value,
            't_statistic_slope': t_statistic,
            'slope_p_value': t_p_value,
            'slope_std_error': se_slope,
            'slope_confidence_interval': (slope_ci_lower, slope_ci_upper),
            'regression_std_error': se_regression,
            'n_observations': n,
            'degrees_of_freedom': df_res,
            'interpretation': f"{'Reject' if f_p_value < alpha else 'Fail to reject'} H0 at α = {alpha}"
        }
        
        if dw_statistic is not None:
            results['durbin_watson'] = dw_statistic
        
        # Print comprehensive results
        print(f"\nLinear Regression Analysis")
        print("=" * 50)
        print(f"Regression Equation: y = {slope:.4f}x + {intercept:.4f}")
        print(f"R-squared: {r_squared:.4f}")
        print(f"Adjusted R-squared: {adj_r_squared:.4f}")
        print(f"Standard Error: {se_regression:.4f}")
        print(f"F-statistic: {f_statistic:.4f} (p = {f_p_value:.4f})")
        print(f"Slope t-test: t = {t_statistic:.4f} (p = {t_p_value:.4f})")
        print(f"95% CI for slope: [{slope_ci_lower:.4f}, {slope_ci_upper:.4f}]")
        
        if dw_statistic is not None:
            print(f"Durbin-Watson: {dw_statistic:.4f}")
        
        # ANOVA table
        print(f"\nANOVA Table:")
        print("-" * 70)
        print(f"{'Source':<12} {'SS':<12} {'df':<6} {'MS':<12} {'F':<10} {'p-value':<10}")
        print("-" * 70)
        print(f"{'Regression':<12} {ss_reg:<12.4f} {df_reg:<6} {ms_reg:<12.4f} {f_statistic:<10.4f} {f_p_value:<10.4f}")
        print(f"{'Residual':<12} {ss_res:<12.4f} {df_res:<6} {ms_res:<12.4f}")
        print(f"{'Total':<12} {ss_tot:<12.4f} {df_tot:<6}")
        print("-" * 70)
        
        print_data_summary(x_data, "X Variable (Predictor)")
        print_data_summary(y_data, "Y Variable (Response)")
        print_test_results(results, hypotheses)
        
        return results
