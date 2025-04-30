import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf
import ruptures as rpt
from scipy import signal
import pandas as pd
# ----------------- nicu hr attributes -------------------------------------------
def successive_increases(y):
    """
    Input: y - numeric array/list of measurements
    Output: dictionary containing counts and proportions of successive increases
    """
    # Check length
    N = len(y)
    if N < 5:
        print("Warning: Time series too short")
        return None
    
    # Get binary sequence (1 for increase, 0 for decrease/same)
    y_diff = np.diff(y)
    y_bin = (y_diff > 0).astype(int)
    
    # Initialize output dictionary
    out = {}
    
    # Single increases (length 1)
    out['u'] = np.mean(y_bin == 1)  # proportion of increases
    out['d'] = np.mean(y_bin == 0)  # proportion of decreases/same
    
    # Two consecutive (length 2)
    n2 = len(y_bin) - 1
    out['uu'] = np.mean((y_bin[:-1] == 1) & (y_bin[1:] == 1))  # up,up
    out['ud'] = np.mean((y_bin[:-1] == 1) & (y_bin[1:] == 0))  # up,down
    out['du'] = np.mean((y_bin[:-1] == 0) & (y_bin[1:] == 1))  # down,up
    out['dd'] = np.mean((y_bin[:-1] == 0) & (y_bin[1:] == 0))  # down,down
    
    # Three consecutive (length 3)
    n3 = len(y_bin) - 2
    out['uuu'] = np.mean((y_bin[:-2] == 1) & (y_bin[1:-1] == 1) & (y_bin[2:] == 1))
    
    # Calculate entropy for each length
    def calc_entropy(p):
        # Remove zeros and calculate entropy
        p = p[p > 0]
        return -np.sum(p * np.log(p))
    
    # Entropy calculations
    out['h'] = calc_entropy(np.array([out['u'], out['d']]))  # entropy of length 1
    out['h2'] = calc_entropy(np.array([out['uu'], out['ud'], out['du'], out['dd']]))  # entropy of length 2
    
    return out

# Example usage:
# y = [1, 2, 1, 3, 2, 4, 5]
# result = successive_increases(y)

def successive_unchanges(y):
    """
    Input: y - numeric array/list of measurements
    Output: dictionary containing counts and proportions of successive unchanges
    """
    # Check length
    N = len(y)
    if N < 5:
        print("Warning: Time series too short")
        return None
    
    # Get binary sequence (1 for no change, 0 otherwise)
    y_diff = np.diff(y)
    y_bin = (y_diff == 0).astype(int)
    
    # Initialize output dictionary
    out = {}
    
    # Single unchanges (length 1)
    out['u'] = np.mean(y_bin == 1)  # proportion of unchanges
    
    # Two consecutive unchanges (length 2)
    n2 = len(y_bin) - 1
    out['uu'] = np.mean((y_bin[:-1] == 1) & (y_bin[1:] == 1))  # unchange,unchange
    
    # Three consecutive (length 3)
    n3 = len(y_bin) - 2
    out['uuu'] = np.mean((y_bin[:-2] == 1) & (y_bin[1:-1] == 1) & (y_bin[2:] == 1))
    
    return out

# Example usage:
# y = [1, 1, 1, 2, 2, 3, 3, 3]
# result = successive_unchanges(y)


# ----------------- synthetic attributes -------------------------------------------
# --- 1. Trend slope ---
def trend_slope(x):
    """OLS slope beta (trend strength & sign)."""
    t = np.arange(len(x))
    return sm.OLS(x, sm.add_constant(t)).fit().params[1]

# --- 2. Seasonality strength ---
def seasonality_strength_acf(series, nlags=50):
    """Evaluate seasonality strength using ACF peaks"""
    acf_values = acf(series, nlags=nlags)
    # Look for significant peaks at regular intervals
    peaks = np.where(acf_values > 2/np.sqrt(len(series)))[0]
    if len(peaks) > 1:
        # Calculate average peak height
        return np.mean(acf_values[peaks])
    return 0

def seasonality_strength_spectral(series):
    """Evaluate seasonality strength using power spectrum"""
    f, Pxx = signal.periodogram(series)
    # Look for significant peaks in the spectrum
    peaks, _ = signal.find_peaks(Pxx, height=np.mean(Pxx))
    if len(peaks) > 0:
        return np.max(Pxx[peaks]) / np.mean(Pxx)
    return 0

# --- 3. Shift of mean ---
def cusum_range(x):
    """CUSUM range and direction: returns (range, direction) where direction is 1 for positive shift, -1 for negative"""
    cumsum = np.cumsum(x - x.mean())
    max_idx = np.argmax(cumsum)
    min_idx = np.argmin(cumsum)
    
    # Determine direction based on which comes first
    if max_idx > min_idx:
        direction = 1  # Positive shift
    else:
        direction = -1  # Negative shift
    
    return np.ptp(cumsum)*direction/len(x)

def rolling_mean_shift(x, window=10):
    """Detect shifts using rolling mean differences, returns (magnitude, direction)"""
    rolling_mean = pd.Series(x).rolling(window).mean()
    diff = rolling_mean.diff()
    max_diff_idx = np.argmax(np.abs(diff))
    magnitude = np.abs(diff).max()
    direction = np.sign(diff.iloc[max_diff_idx])
    
    return magnitude*direction


# --- 4. Step shift (abrupt) ---
def step_shift(x, pen=1):
    # pen is the penalty parameter for the PELT algorithm
    # low value = more sensitive to changes (1) Good for detecting small shifts
    # high value = less sensitive to changes (10) Only detects large, significant shifts

    """Detect step shifts using PELT algorithm"""
    cp = rpt.Pelt(model="l2").fit(x).predict(pen=pen)
    if len(cp) <= 1:          # no change-point found
        return 0.0
    i = cp[0]
    shift = x[i:].mean() - x[:i].mean()
    return abs(shift) * np.sign(shift)

def cusum_step(x, threshold=2):
    """Detect steps using CUSUM"""
    cumsum = np.cumsum(x - x.mean())
    max_idx = np.argmax(np.abs(cumsum))
    if np.abs(cumsum[max_idx]) > threshold:
        shift = -(x[max_idx] - x[max_idx-1])
        return abs(shift)*np.sign(shift)
    return 0.0

# --- 5. Variability ---------------------------------------------------------
def coeff_variation(x):
    """Coefficient of variation."""
    return np.std(x, ddof=1) / abs(x.mean())
