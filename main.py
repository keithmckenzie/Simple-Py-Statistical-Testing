#!/usr/bin/env python3
#Keith Ngamphon McKenzie
#keith@mckenzie.page
#https://mckenzie.page
#Python Simple Statistical Tests

import sys
import os

# Add current directory to path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from menu_system import MenuSystem
from data_manager import DataManager
from utils.formatters import print_header, print_separator

def main():
    """Main application entry point"""
    print_header("Simple Py Statistical Testing")
    print("Keith Ngamphon McKenzie (keith@mckenzie.page)")
    print("https://mckenzie.page")
    print()
    print("Welcome to the comprehensive statistical testing toolkit!")
    print("This application supports 13 different statistical tests with flexible data input.")
    print_separator()
    
    # Initialize core components
    data_manager = DataManager()
    menu_system = MenuSystem(data_manager)
    
    try:
        # Start the main menu loop
        menu_system.run()
    except KeyboardInterrupt:
        print("\n\nApplication terminated by user.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        print("Please restart the application.")
    finally:
        print("\nThank you for using the Statistical Testing Application!")

if __name__ == "__main__":
    main()
