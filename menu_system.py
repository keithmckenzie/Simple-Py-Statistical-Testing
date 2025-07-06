#Keith Ngamphon McKenzie
#keith@mckenzie.page
#https://mckenzie.page
#Python Simple Statistical Tests

import sys
from typing import Dict, Any, Callable
from data_manager import DataManager
from tests.parametric_tests import ParametricTests
from tests.nonparametric_tests import NonParametricTests
from tests.chi_square_tests import ChiSquareTests
from tests.correlation_tests import CorrelationTests
from utils.formatters import print_header, print_separator

class MenuSystem:
    """Interactive menu system for statistical tests"""
    
    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager
        self.parametric_tests = ParametricTests()
        self.nonparametric_tests = NonParametricTests()
        self.chi_square_tests = ChiSquareTests()
        self.correlation_tests = CorrelationTests()
        
        # Menu structure
        self.test_menu = {
            '1': ('Wilcoxon Signed-Rank Test', self._wilcoxon_menu),
            '2': ("Student's T-Test (One Sample)", self._students_t_menu),
            '3': ("Student's T-Test (Two Sample)", self._independent_t_menu),
            '4': ('Paired T-Test', self._paired_t_menu),
            '5': ('Mann-Whitney U Test', self._mann_whitney_menu),
            '6': ('Chi-Square Goodness of Fit', self._chi_square_gof_menu),
            '7': ('Chi-Square Test of Association', self._chi_square_assoc_menu),
            '8': ('Coefficient of Determination', self._coefficient_determination_menu),
            '9': ('F-Test (Equality of Variances)', self._f_test_menu),
            '10': ('One-Way ANOVA', self._anova_menu),
            '11': ('Kruskal-Wallis Test', self._kruskal_wallis_menu),
            '12': ("Spearman's Rank Correlation", self._spearman_menu),
            '13': ('Linear Regression Analysis', self._linear_regression_menu),
        }
    
    def run(self):
        """Main menu loop"""
        while True:
            self._display_main_menu()
            choice = input("\nEnter your choice: ").strip()
            
            if choice == '0':
                break
            elif choice == 'd':
                self._data_management_menu()
            elif choice in self.test_menu:
                self._run_test(choice)
            else:
                print("Invalid choice. Please try again.")
                input("Press Enter to continue...")
    
    def _display_main_menu(self):
        """Display the main menu"""
        print_header("Simple Py Statistical Testing - Main Menu")
        print("Keith Ngamphon McKenzie (keith@mckenzie.page)")
        print("https://mckenzie.page")
        print_separator("-", 60)
        print("Available Statistical Tests:")
        print_separator("-", 60)
        
        for key, (name, _) in self.test_menu.items():
            print(f"{key:>2}. {name}")
        
        print_separator("-", 60)
        print(" d. Data Management")
        print(" 0. Exit")
        print_separator("-", 60)
        
        # Show current datasets
        datasets = self.data_manager.list_datasets()
        if datasets:
            print(f"Current datasets: {', '.join(datasets)}")
        else:
            print("No datasets currently stored.")
    
    def _data_management_menu(self):
        """Data management submenu"""
        while True:
            print_header("Data Management")
            print("1. Add new dataset")
            print("2. View datasets")
            print("3. Remove dataset")
            print("0. Back to main menu")
            print_separator()
            
            choice = input("Enter choice: ").strip()
            
            if choice == '0':
                break
            elif choice == '1':
                self._add_dataset()
            elif choice == '2':
                self._view_datasets()
            elif choice == '3':
                self._remove_dataset()
            else:
                print("Invalid choice.")
            
            input("Press Enter to continue...")
    
    def _add_dataset(self):
        """Add a new dataset"""
        print("\nAdd New Dataset")
        print_separator("-", 30)
        
        name = input("Enter dataset name: ").strip()
        if not name:
            print("Dataset name cannot be empty.")
            return
        
        if name in self.data_manager.list_datasets():
            overwrite = input(f"Dataset '{name}' already exists. Overwrite? (y/n): ").strip().lower()
            if overwrite != 'y':
                print("Operation cancelled.")
                return
        
        print("Enter data as comma-separated numbers:")
        data_str = input("> ").strip()
        
        if self.data_manager.add_dataset(name, data_str):
            print(f"Dataset '{name}' added successfully!")
            info = self.data_manager.get_dataset_info(name)
            if info:
                print(f"Summary: n={info['count']}, mean={info['mean']:.3f}, std={info['std_dev']:.3f}")
            else:
                print("Error getting dataset summary.")
        else:
            print("Failed to add dataset. Please check your data format.")
    
    def _view_datasets(self):
        """View all stored datasets"""
        print("\nStored Datasets")
        print_separator("-", 30)
        self.data_manager.display_datasets()
    
    def _remove_dataset(self):
        """Remove a dataset"""
        datasets = self.data_manager.list_datasets()
        if not datasets:
            print("No datasets to remove.")
            return
        
        print("\nRemove Dataset")
        print_separator("-", 30)
        
        for i, name in enumerate(datasets, 1):
            print(f"{i}. {name}")
        
        try:
            choice = int(input("Enter dataset number to remove: "))
            if 1 <= choice <= len(datasets):
                name = datasets[choice - 1]
                confirm = input(f"Remove '{name}'? (y/n): ").strip().lower()
                if confirm == 'y':
                    self.data_manager.remove_dataset(name)
                    print(f"Dataset '{name}' removed.")
                else:
                    print("Operation cancelled.")
            else:
                print("Invalid dataset number.")
        except ValueError:
            print("Please enter a valid number.")
    
    def _run_test(self, test_key: str):
        """Run the selected statistical test"""
        test_name, test_function = self.test_menu[test_key]
        
        try:
            print_header(f"Running: {test_name}")
            test_function()
        except Exception as e:
            print(f"\nError running test: {e}")
        finally:
            input("\nPress Enter to continue...")
    
    # Individual test menu functions
    def _wilcoxon_menu(self):
        """Wilcoxon Signed-Rank Test menu"""
        print("Wilcoxon Signed-Rank Test Options:")
        print("1. One-sample test (compare sample against hypothesized median)")
        print("2. Two-sample paired test (compare two related samples)")
        
        choice = input("Select option (1-2): ").strip()
        
        if choice == '1':
            # One-sample test
            _, data = self.data_manager.select_dataset("Select dataset for one-sample test")
            # Get hypothesized median
            while True:
                try:
                    hyp_median = float(input("Enter hypothesized median (default 0): ") or "0")
                    break
                except ValueError:
                    print("Please enter a valid number.")
            self.nonparametric_tests.one_sample_wilcoxon_test(data, hyp_median)
        elif choice == '2':
            # Two-sample paired test
            _, data1 = self.data_manager.select_dataset("Select first paired sample (e.g., before)")
            _, data2 = self.data_manager.select_dataset("Select second paired sample (e.g., after)")
            self.nonparametric_tests.wilcoxon_signed_rank_test(data1, data2)
        else:
            print("Invalid option.")
    
    def _students_t_menu(self):
        """Student's T-Test menu"""
        _, data = self.data_manager.select_dataset("Select dataset for one-sample t-test")
        self.parametric_tests.students_t_test(data)
    
    def _independent_t_menu(self):
        """Independent T-Test menu"""
        _, data1 = self.data_manager.select_dataset("Select first independent sample")
        _, data2 = self.data_manager.select_dataset("Select second independent sample")
        self.parametric_tests.independent_t_test(data1, data2)
    
    def _paired_t_menu(self):
        """Paired T-Test menu"""
        _, data1 = self.data_manager.select_dataset("Select first dataset (e.g., pre-treatment)")
        _, data2 = self.data_manager.select_dataset("Select second dataset (e.g., post-treatment)")
        self.parametric_tests.paired_t_test(data1, data2)
    
    def _mann_whitney_menu(self):
        """Mann-Whitney U Test menu"""
        _, data1 = self.data_manager.select_dataset("Select first independent sample")
        _, data2 = self.data_manager.select_dataset("Select second independent sample")
        self.nonparametric_tests.mann_whitney_test(data1, data2)
    
    def _chi_square_gof_menu(self):
        """Chi-Square Goodness of Fit menu"""
        print("Enter observed frequencies (comma-separated):")
        observed_str = input("> ").strip()
        
        try:
            observed = [float(x.strip()) for x in observed_str.split(',')]
        except ValueError:
            print("Invalid data format.")
            return
        
        print("Enter expected frequencies (comma-separated, or press Enter for equal distribution):")
        expected_str = input("> ").strip()
        
        expected = None
        if expected_str:
            try:
                expected = [float(x.strip()) for x in expected_str.split(',')]
            except ValueError:
                print("Invalid expected frequencies format.")
                return
        
        if expected:
            self.chi_square_tests.chi_square_goodness_of_fit(observed, expected)
        else:
            self.chi_square_tests.chi_square_goodness_of_fit(observed)
    
    def _chi_square_assoc_menu(self):
        """Chi-Square Test of Association menu"""
        contingency_table = self.chi_square_tests.input_contingency_table()
        self.chi_square_tests.chi_square_association(contingency_table)
    
    def _coefficient_determination_menu(self):
        """Coefficient of Determination menu"""
        (_, x_data), (_, y_data) = self.data_manager.select_two_datasets(
            "Select X variable (predictor) dataset", 
            "Select Y variable (response) dataset"
        )
        self.correlation_tests.coefficient_of_determination(x_data, y_data)
    
    def _f_test_menu(self):
        """F-Test menu"""
        _, data1 = self.data_manager.select_dataset("Select first sample")
        _, data2 = self.data_manager.select_dataset("Select second sample")
        self.parametric_tests.f_test(data1, data2)
    
    def _anova_menu(self):
        """One-Way ANOVA menu"""
        print("One-Way ANOVA - Enter groups one by one")
        groups = []
        group_num = 1
        
        while True:
            print(f"\nGroup {group_num} (press Enter without data to finish):")
            _, data = self.data_manager.select_dataset(f"Select dataset for Group {group_num}")
            
            if data:
                groups.append(data)
                group_num += 1
                
                if len(groups) >= 2:
                    continue_input = input("Add another group? (y/n): ").strip().lower()
                    if continue_input != 'y':
                        break
            else:
                break
        
        if len(groups) >= 2:
            self.parametric_tests.one_way_anova(*groups)
        else:
            print("ANOVA requires at least 2 groups.")
    
    def _kruskal_wallis_menu(self):
        """Kruskal-Wallis Test menu"""
        print("Kruskal-Wallis Test - Enter groups one by one")
        groups = []
        group_num = 1
        
        while True:
            print(f"\nGroup {group_num} (press Enter without data to finish):")
            _, data = self.data_manager.select_dataset(f"Select dataset for Group {group_num}")
            
            if data:
                groups.append(data)
                group_num += 1
                
                if len(groups) >= 2:
                    continue_input = input("Add another group? (y/n): ").strip().lower()
                    if continue_input != 'y':
                        break
            else:
                break
        
        if len(groups) >= 2:
            self.nonparametric_tests.kruskal_wallis_test(*groups)
        else:
            print("Kruskal-Wallis test requires at least 2 groups.")
    
    def _spearman_menu(self):
        """Spearman's Rank Correlation menu"""
        (_, x_data), (_, y_data) = self.data_manager.select_two_datasets(
            "Select X variable dataset", 
            "Select Y variable dataset"
        )
        self.correlation_tests.spearmans_rank_correlation(x_data, y_data)
    
    def _linear_regression_menu(self):
        """Linear Regression Analysis menu"""
        (_, x_data), (_, y_data) = self.data_manager.select_two_datasets(
            "Select X variable (predictor) dataset", 
            "Select Y variable (response) dataset"
        )
        self.correlation_tests.linear_regression_tests(x_data, y_data)
