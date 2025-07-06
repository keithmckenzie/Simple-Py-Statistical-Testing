# Simple Py Statistical Testing

**Author:** Keith Ngamphon McKenzie\
**Email:** [keith\@mckenzie.page](mailto:keith@mckenzie.page)\
**Website:** <https://mckenzie.page>

## Overview

A comprehensive terminal-based statistical testing application with 13
different statistical tests.

## What it's about

-   **13 Statistical Tests:**
    -   Wilcoxon Signed-Rank Test (one-sample and two-sample paired)
    -   Student's T-Tests (one-sample and two-sample independent)
    -   Paired T-Test
    -   Mann-Whitney Test (independent samples)
    -   Chi-Square Tests (Goodness of Fit and Association)
    -   F-Test for Equality of Variances
    -   One-Way ANOVA
    -   Kruskal-Wallis Test
    -   Spearman's Rank Correlation
    -   Coefficient of Determination
    -   Linear Regression Analysis
-   **User-Friendly Interface:**
    -   Interactive terminal menu system
    -   Flexible comma-separated data input
    -   Dataset naming and storage capabilities
    -   Comprehensive result formatting with p-values
    -   Assumption checking and warnings
-   **Technical Features:**
    -   Modular architecture with separated concerns
    -   Comprehensive input validation
    -   Statistical assumption verification

## Usage

You may use the included executable, which is not compiled. This executable was made with PyInstaller.
This means that it will take time to unpack.

You can also just Build the application:

``` bash
build_and_run.bat
```

Run the Python application (after first build):

``` bash
python main.py
```

Follow the interactive menu to:

1\. Add datasets using comma-separated values

2\. Select from 13 available statistical tests

3\. View comprehensive results with p-values and interpretations

## Dependencies

If you have Python 3.8+, the build batch file will install
the dependencies for you.

-   Python 3.8+ - https://www.python.org/downloads/
-   NumPy \>= 1.20.0 - https://numpy.org/install/
-   SciPy \>= 1.7.0 - https://scipy.org/

## About Me

**Keith Ngamphon McKenzie**

Email: [keith\@mckenzie.page](mailto:keith@mckenzie.page)

Website: <https://mckenzie.page>

## License

MIT License

Copyright (c) 2025 Keith McKenzie

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.