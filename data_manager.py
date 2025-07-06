#Keith Ngamphon McKenzie
#keith@mckenzie.page
#https://mckenzie.page
#Python Simple Statistical Tests

from typing import Dict, List, Optional, Tuple
import statistics
from utils.validators import validate_numeric_data, parse_comma_separated

class DataManager:
    """Manages datasets for statistical testing"""
    
    def __init__(self):
        self.datasets: Dict[str, List[float]] = {}
    
    def add_dataset(self, name: str, data_str: str) -> bool:
        """
        Add a new dataset from comma-separated string
        
        Args:
            name: Name for the dataset
            data_str: Comma-separated string of numbers
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            data = parse_comma_separated(data_str)
            if validate_numeric_data(data):
                self.datasets[name] = data
                return True
            return False
        except Exception as e:
            print(f"Error adding dataset: {e}")
            return False
    
    def get_dataset(self, name: str) -> Optional[List[float]]:
        """Get dataset by name"""
        return self.datasets.get(name)
    
    def list_datasets(self) -> List[str]:
        """Get list of all dataset names"""
        return list(self.datasets.keys())
    
    def remove_dataset(self, name: str) -> bool:
        """Remove a dataset"""
        if name in self.datasets:
            del self.datasets[name]
            return True
        return False
    
    def get_dataset_info(self, name: str) -> Optional[dict]:
        """Get basic statistics about a dataset"""
        data = self.get_dataset(name)
        if data is None:
            return None
        
        return {
            'name': name,
            'count': len(data),
            'mean': statistics.mean(data),
            'median': statistics.median(data),
            'std_dev': statistics.stdev(data) if len(data) > 1 else 0,
            'min': min(data),
            'max': max(data)
        }
    
    def display_datasets(self):
        """Display all datasets with basic info"""
        if not self.datasets:
            print("No datasets currently stored.")
            return
        
        print("\nStored Datasets:")
        print("-" * 80)
        print(f"{'Name':<15} {'Count':<8} {'Mean':<12} {'Std Dev':<12} {'Range':<15}")
        print("-" * 80)
        
        for name in self.datasets:
            info = self.get_dataset_info(name)
            if info:
                range_str = f"{info['min']:.2f} - {info['max']:.2f}"
                print(f"{name:<15} {info['count']:<8} {info['mean']:<12.3f} "
                      f"{info['std_dev']:<12.3f} {range_str:<15}")
        print("-" * 80)
    
    def input_single_dataset(self, prompt: str = "Enter data", allow_naming: bool = True) -> List[float]:
        """
        Interactive input for a single dataset
        
        Args:
            prompt: Custom prompt message
            allow_naming: Whether to prompt for dataset naming after input
            
        Returns:
            List of float values
        """
        while True:
            try:
                print(f"\n{prompt} (comma-separated numbers):")
                data_str = input("> ").strip()
                
                if not data_str:
                    print("Please enter some data.")
                    continue
                
                data = parse_comma_separated(data_str)
                if validate_numeric_data(data):
                    # Offer to name and store the dataset
                    if allow_naming:
                        name_choice = input("Would you like to name and store this dataset? (y/n): ").strip().lower()
                        if name_choice in ['y', 'yes']:
                            while True:
                                dataset_name = input("Enter a name for this dataset: ").strip()
                                if dataset_name:
                                    if self.add_dataset(dataset_name, data_str):
                                        print(f"Dataset '{dataset_name}' saved successfully!")
                                    else:
                                        print(f"Dataset '{dataset_name}' already exists. Choose a different name.")
                                        continue
                                    break
                                else:
                                    print("Please enter a valid name.")
                    return data
                else:
                    print("Invalid data format. Please enter numeric values only.")
                    
            except Exception as e:
                print(f"Error parsing data: {e}")
                print("Please try again with comma-separated numbers.")
    
    def select_dataset(self, prompt: str = "Select dataset") -> Tuple[str, List[float]]:
        """
        Interactive dataset selection
        
        Args:
            prompt: Custom prompt message
            
        Returns:
            Tuple of (dataset_name, dataset_values)
        """
        if not self.datasets:
            print("No datasets available. Please enter data directly.")
            data = self.input_single_dataset("Enter your data")
            return "direct_input", data
        
        print(f"\n{prompt}:")
        print("0. Enter new data directly")
        
        dataset_names = self.list_datasets()
        for i, name in enumerate(dataset_names, 1):
            info = self.get_dataset_info(name)
            if info:
                print(f"{i}. {name} (n={info['count']}, mean={info['mean']:.3f})")
            else:
                print(f"{i}. {name} (no info available)")
        
        while True:
            try:
                choice = input("\nEnter choice number: ").strip()
                choice_num = int(choice)
                
                if choice_num == 0:
                    data = self.input_single_dataset("Enter your data")
                    return "direct_input", data
                elif 1 <= choice_num <= len(dataset_names):
                    name = dataset_names[choice_num - 1]
                    return name, self.datasets[name]
                else:
                    print(f"Please enter a number between 0 and {len(dataset_names)}")
                    
            except ValueError:
                print("Please enter a valid number.")
    
    def select_two_datasets(self, x_prompt: str = "Select X variable dataset", 
                           y_prompt: str = "Select Y variable dataset") -> Tuple[Tuple[str, List[float]], Tuple[str, List[float]]]:
        """
        Interactive selection for two datasets with custom prompts for X and Y variables
        
        Args:
            x_prompt: Prompt for X variable
            y_prompt: Prompt for Y variable
            
        Returns:
            Tuple of two (dataset_name, dataset_values) tuples
        """
        if not self.datasets:
            print("No datasets available. Please enter data directly.")
            x_data = self.input_single_dataset("Enter your data (X)")
            y_data = self.input_single_dataset("Enter your data (Y)")
            return ("direct_input_X", x_data), ("direct_input_Y", y_data)
        
        # Select X dataset
        x_name, x_data = self.select_dataset(x_prompt)
        
        # Select Y dataset  
        y_name, y_data = self.select_dataset(y_prompt)
        
        return (x_name, x_data), (y_name, y_data)
