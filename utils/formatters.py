#Keith Ngamphon McKenzie
#keith@mckenzie.page
#https://mckenzie.page
#Python Simple Statistical Tests

import math
from typing import Dict, Any, List, Optional

def print_header(title: str, width: int = 80):
    """Print a formatted header"""
    print("=" * width)
    print(f"{title:^{width}}")
    print("=" * width)

def print_separator(char: str = "-", width: int = 80):
    """Print a separator line"""
    print(char * width)

def print_test_header(test_name: str):
    """Print formatted test header"""
    print(f"\n{'='*60}")
    print(f"{test_name:^60}")
    print(f"{'='*60}")

def format_p_value(p_value: float) -> str:
    """
    Format p-value for display
    
    Args:
        p_value: P-value to format
        
    Returns:
        Formatted p-value string
    """
    if p_value < 0.001:
        return "p < 0.001"
    elif p_value < 0.01:
        return f"p = {p_value:.4f}"
    else:
        return f"p = {p_value:.3f}"

def format_statistic(stat_value: float, stat_name: str = "Test statistic") -> str:
    """
    Format test statistic for display
    
    Args:
        stat_value: Statistical value
        stat_name: Name of the statistic
        
    Returns:
        Formatted statistic string
    """
    return f"{stat_name}: {stat_value:.4f}"

def interpret_p_value(p_value: float, alpha: float = 0.05) -> str:
    """
    Provide interpretation of p-value
    
    Args:
        p_value: P-value from test
        alpha: Significance level
        
    Returns:
        Interpretation string
    """
    if p_value < alpha:
        return f"Significant at α = {alpha} level (reject H0)"
    else:
        return f"Not significant at α = {alpha} level (fail to reject H0)"

def format_confidence_interval(ci_lower: float, ci_upper: float, 
                             confidence: float = 95) -> str:
    """
    Format confidence interval
    
    Args:
        ci_lower: Lower bound
        ci_upper: Upper bound
        confidence: Confidence level percentage
        
    Returns:
        Formatted CI string
    """
    return f"{confidence}% CI: [{ci_lower:.4f}, {ci_upper:.4f}]"

def print_test_results(results: Dict[str, Any], hypotheses: Optional[tuple] = None):
    """
    Print formatted test results
    
    Args:
        results: Dictionary containing test results
        hypotheses: Tuple of (null_hypothesis, alternative_hypothesis)
    """
    print("\nTest Results:")
    print("-" * 50)
    
    # Print hypotheses if provided
    if hypotheses:
        null_hyp, alt_hyp = hypotheses
        print(f"H0: {null_hyp}")
        print(f"H1: {alt_hyp}")
        print("-" * 50)
    
    # Print main results
    for key, value in results.items():
        if key == 'p_value':
            print(f"P-value: {format_p_value(value)}")
        elif key == 'statistic':
            print(f"Test statistic: {value:.4f}")
        elif key.endswith('_statistic'):
            stat_name = key.replace('_', ' ').title()
            print(f"{stat_name}: {value:.4f}")
        elif key == 'interpretation':
            print(f"Interpretation: {value}")
        elif key == 'effect_size':
            print(f"Effect size: {value:.4f}")
        elif key == 'confidence_interval':
            if isinstance(value, tuple) and len(value) == 2:
                print(f"95% CI: [{value[0]:.4f}, {value[1]:.4f}]")
        elif isinstance(value, (int, float)):
            key_formatted = key.replace('_', ' ').title()
            if isinstance(value, float):
                print(f"{key_formatted}: {value:.4f}")
            else:
                print(f"{key_formatted}: {value}")
        else:
            key_formatted = key.replace('_', ' ').title()
            print(f"{key_formatted}: {value}")

def print_assumption_warnings(warnings: List[str]):
    """
    Print assumption warnings
    
    Args:
        warnings: List of warning messages
    """
    if warnings:
        print("\n⚠️  Assumption Warnings:")
        print("-" * 30)
        for warning in warnings:
            print(f"• {warning}")

def print_data_summary(data: List[float], name: str = "Data"):
    """
    Print summary statistics for dataset
    
    Args:
        data: Dataset
        name: Name of the dataset
    """
    n = len(data)
    mean_val = sum(data) / n
    
    if n > 1:
        variance = sum((x - mean_val) ** 2 for x in data) / (n - 1)
        std_dev = math.sqrt(variance)
    else:
        std_dev = 0
    
    print(f"\n{name} Summary:")
    print(f"  n = {n}")
    print(f"  Mean = {mean_val:.4f}")
    print(f"  Std Dev = {std_dev:.4f}")
    print(f"  Range = [{min(data):.4f}, {max(data):.4f}]")

def format_regression_results(results: Dict[str, Any]) -> str:
    """
    Format regression analysis results
    
    Args:
        results: Regression results dictionary
        
    Returns:
        Formatted results string
    """
    output = []
    
    if 'slope' in results and 'intercept' in results:
        output.append(f"Regression equation: y = {results['slope']:.4f}x + {results['intercept']:.4f}")
    
    if 'r_squared' in results:
        output.append(f"R-squared: {results['r_squared']:.4f}")
        output.append(f"Variance explained: {results['r_squared']*100:.2f}%")
    
    if 'slope_se' in results:
        output.append(f"Slope standard error: {results['slope_se']:.4f}")
    
    return "\n".join(output)

def get_significance_stars(p_value: float) -> str:
    """
    Get significance stars based on p-value
    
    Args:
        p_value: P-value
        
    Returns:
        Significance stars string
    """
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    elif p_value < 0.1:
        return "."
    else:
        return " "

def print_significance_legend():
    """Print significance level legend"""
    print("\nSignificance levels: *** p<0.001, ** p<0.01, * p<0.05, . p<0.1")
