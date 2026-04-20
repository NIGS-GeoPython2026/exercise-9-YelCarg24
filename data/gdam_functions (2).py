import numpy as np

def linregress(x, y):
    # N is the number of data points
    N = len(x)
    
    # Calculate Delta (the denominator)
    # Formula: Delta = N * sum(x^2) - (sum(x))^2
    delta = N * np.sum(x**2) - (np.sum(x))**2
    
    # Calculate A (y-intercept)
    # Formula: A = [sum(x^2) * sum(y) - sum(x) * sum(xy)] / Delta
    a = (np.sum(x**2) * np.sum(y) - np.sum(x) * np.sum(x * y)) / delta
    
    # Calculate B (slope)
    # Formula: B = [N * sum(xy) - sum(x) * sum(y)] / Delta
    b = (N * np.sum(x * y) - np.sum(x) * np.sum(y)) / delta
    
    return a, b


def pearson(x, y):
    """Calculates the Pearson correlation coefficient."""
    # We use numpy's built-in tool to get the correlation matrix
    selection = np.corrcoef(x, y)
    # The result is a table; we want the value at row 0, column 1
    r = selection[0, 1]
    return r

def chi_squared(observed, expected, std_dev):
    """
    Calculates the Reduced Chi-Squared value using N-2 degrees of freedom.
    """
    import numpy as np
    
    # 1. Calculate degrees of freedom (N - 2)
    # N is the length of the observed data array
    df = len(observed) - 2
    
    # 2. Calculate the chi-squared sum
    # Formula: sum( ((obs - exp) / std)**2 )
    chi_sq = np.sum(((observed - expected) / std_dev)**2)
    
    # 3. Divide by degrees of freedom
    reduced_chi_sq = chi_sq / df
    
    return reduced_chi_sq