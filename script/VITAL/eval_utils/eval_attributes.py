import numpy as np
import statsmodels.api as sm
# seasonality
from scipy.signal import periodogram
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.seasonal import STL

import ruptures as rpt
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

def trend_slope_moving_window(x, length=20, step=5):
    """Calculate trend slope using moving windows and robust statistics.
    
    Args:
        x (array-like): Input time series
        length (int): Length of moving window (default: 20)
        step (int): Step size between windows (default: 5)
    
    Returns:
        float: Robust estimate of overall trend slope
    """
    # Convert input to numpy array
    x = np.asarray(x, dtype=float)
    
    # Input validation
    if len(x) < length:
        return trend_slope(x)  # fallback to simple slope if series too short
    
    # Calculate slopes for each window
    slopes = []
    for i in range(0, len(x) - length + 1, step):
        window = x[i:i+length]
        try:
            slope = trend_slope(window)
            if not np.isnan(slope) and not np.isinf(slope):
                slopes.append(slope)
        except:
            continue
    
    if not slopes:
        return 0.0
    
    # Remove outliers using IQR method
    slopes = np.array(slopes)
    q1, q3 = np.percentile(slopes, [25, 75])
    iqr = q3 - q1
    mask = (slopes >= q1 - 1.5*iqr) & (slopes <= q3 + 1.5*iqr)
    slopes_clean = slopes[mask]
    
    # Return robust central tendency
    return float(np.median(slopes_clean) if len(slopes_clean) > 0 else np.median(slopes))
   

# --- 2. Seasonality strength ---
# ------------------------------------------------------------------
# helper – estimate fundamental period (integer ≥ 2)
# ------------------------------------------------------------------
def _estimate_period(x, fs=1.0):
    """
    Return integer period m̂ from the strongest spectral peak.
    """
    x = np.asarray(x, dtype=float) - np.mean(x)
    f, P = periodogram(x, fs=fs, window='hann', detrend='linear')
    f, P = f[1:], P[1:]                     # drop DC
    if len(P) == 0:
        return 1                           # degenerate fallback
    m_hat = int(round(1.0 / f[np.argmax(P)]))
    # keep in a reasonable range
    return max(2, min(m_hat, len(x) // 2))

# ------------------------------------------------------------------
# 1. ACF-based strength  (average of harmonics m,2m,3m,…)
# ------------------------------------------------------------------
def seasonality_strength_acf(series, m=None, fs=1.0, nlags=None):
    """
    Strength = mean positive PACF values at lags k·m, k = 1,2,...
    If m is None it is auto-estimated.
    """
    x = np.asarray(series, dtype=float) - np.mean(series)
    if m is None:
        m = _estimate_period(x, fs)
    if nlags is None:
        nlags = min(4 * m, len(x) - 1)  # Ensure nlags doesn't exceed series length
    r = acf(x, nlags=nlags, fft=True)
    ks = np.arange(1, nlags // m + 1)
    if len(ks) == 0:  # Handle case where nlags < m
        return 0.0
    indices = ks * m
    indices = indices[indices < len(r)]  # Ensure indices are within bounds
    if len(indices) == 0:
        return 0.0
    vals = r[indices]
    vals = vals[vals > 0]  # ignore negative correlations
    return float(vals.mean()) if vals.size else 0.0


# ------------------------------------------------------------------
# 2. STL / X-11 strength (Hyndman & Athanasopoulos)
# ------------------------------------------------------------------
def seasonality_strength_stl(series, m=None, fs=1.0):
    """
    S = 1 – Var(R) / Var(R+S)   where R = remainder, S = seasonal component.
    If m is None it is auto-estimated.
    """
    x = np.asarray(series, dtype=float)
    if m is None:
        m = _estimate_period(x, fs)
    res = STL(x, period=m, robust=True).fit()
    s, r = res.seasonal, res.resid
    denom = np.var(r + s, ddof=1)
    return 0.0 if denom == 0 else 1.0 - np.var(r, ddof=1) / denom

# ------------------------------------------------------------------
# 3. Spectral–power-ratio (periodogram-based) seasonality 
# (Pereira et al., Information, 2022)
# ------------------------------------------------------------------
def seasonality_strength_spectrum(series, fs=1.0, m_hint=None, bw=1):
    """
    Fraction of spectral power in the dominant seasonal peak.
    If m_hint is given (length of season) we integrate a small band
    around that frequency; else we take the highest non-DC peak.
    """
    f, P = periodogram(series, fs=fs, window='hann', detrend='linear',
                       scaling='spectrum')
    # Drop DC
    f, P = f[1:], P[1:]
    if m_hint is not None:
        target = 1.0 / m_hint
        idx = np.argmin(np.abs(f - target))
    else:
        idx = np.argmax(P)
    # integrate ±bw bins
    lo, hi = max(idx-bw, 0), min(idx+bw+1, len(P))
    return P[lo:hi].sum() / P.sum()

# ------------------------------------------------------------------
# 4. spectral entropy
# (Martínez-Granados et al., Entropy, 2020) https://www.mdpi.com/1099-4300/22/1/89?utm_source=chatgpt.com
# ------------------------------------------------------------------
def seasonality_spectral_entropy(series, fs=1.0):
    """
    Strength = 1 – normalized spectral entropy.
    0: white noise, 1: single pure tone.
    """
    _, P = periodogram(series, fs=fs, window='hann', detrend='linear')
    P = P[1:]                       # drop DC
    P /= P.sum()
    ent = -np.sum(P * np.log(P + 1e-12)) / np.log(len(P))
    return 1.0 - ent                # map to [0,1]

def get_seasonality_scores(series, fs=1.0):
    """
    Calculate all four seasonality strength scores for a time series.
    
    Returns:
        dict: Dictionary containing four seasonality scores:
            - acf: ACF-based strength (0-1)
            - stl: STL-based strength (0-1)
            - spectrum: Spectral power ratio (0-1)
            - entropy: Spectral entropy-based strength (0-1)
    """
    m = _estimate_period(series, fs)
    return {
        'acf': np.round(seasonality_strength_acf(series, m=m, fs=fs), 4),
        'stl': np.round(seasonality_strength_stl(series, m=m, fs=fs), 4),
        'spectrum': np.round(seasonality_strength_spectrum(series, fs=fs, m_hint=m), 4),
        'entropy': np.round(seasonality_spectral_entropy(series, fs=fs), 4)
    }


# --- 3. Shift of mean ---
def shift_cusum(x):
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

def shift_rolling_mean(x, window=10):
    """Detect shifts using rolling mean differences, returns (magnitude, direction)"""
    rolling_mean = pd.Series(x).rolling(window).mean()
    diff = rolling_mean.diff()
    max_diff_idx = np.argmax(np.abs(diff))
    magnitude = np.abs(diff).max()
    direction = np.sign(diff.iloc[max_diff_idx])
    
    return magnitude*direction

def shift_pelt(x, pen=1):
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


# --- 4. Step shift (abrupt) ---
def step_cusum(x, threshold=2):
    """Detect steps using CUSUM"""
    cumsum = np.cumsum(x - x.mean())
    max_idx = np.argmax(np.abs(cumsum))
    if np.abs(cumsum[max_idx]) > threshold:
        shift = -(x[max_idx] - x[max_idx-1])
        return abs(shift)*np.sign(shift)
    return 0.0

def step_pelt(
    x
) -> float:
    """
    Detect a *temporary* step-like bump (up-then-down or down-then-up).

    Returns
    -------
    float
        +H  : positive bump (level goes up, stays ≥ min_len samples, then down)
        -H  : negative bump (level goes down, then up)
         0  : no bump, or only a permanent mean shift.
    """
    jump_score_all = []
    for pen in np.logspace(1.0, 2.0, 50):  
        x = np.asarray(x, dtype=float)

        # 1 — change-points with PELT (last cp returned == len(x))
        cps = rpt.Pelt(model="l2").fit(x).predict(pen=pen)
        cps = [0] + cps                                    # prepend start idx

        jump_score = 0 
        # 2 — scan triples of consecutive segments for the bump pattern
        for left, middle, right in zip(cps, cps[1:], cps[2:]):
            plateau_len = right - middle
            if plateau_len < 2:
                continue

            mean_left   = x[left:middle].mean()
            mean_mid    = x[middle:right].mean()
            mean_right  = x[right:].mean()

            jump_1 = mean_mid  - mean_left   # first step edge
            jump_2 = mean_right - mean_mid   # second edge

            # need opposite signs and similar magnitude (within 25 %)
            if np.sign(jump_1) == -np.sign(jump_2) and \
            abs(abs(jump_1) - abs(jump_2)) < 0.25 * abs(jump_1):

                noise_std = (
                    np.sqrt(
                        np.var(x[left:middle], ddof=1) +
                        np.var(x[right:],      ddof=1)
                    ) / 2
                ) or 1.0

                if abs(jump_1) / noise_std >= 0.5:
                    jump_score = float(jump_1)      # positive ⇒ up-then-down, negative ⇒ down-then-up
        jump_score_all.append(jump_score)

    # return median of all jump scores
    return float(np.mean(jump_score_all))



# --- 5. Variability ---
def coeff_variation(x):
    """Coefficient of variation."""
    return np.std(x, ddof=1) / abs(x.mean())



def get_all_properties(x, step = False):
    """Calculate math properties of a given timeseries"""
    res = {
        'trend': np.round(trend_slope_moving_window(x), 4),
        'seasonality': get_seasonality_scores(x),
        # 'seasonality': np.round(seasonality_spectral_entropy(x), 4),  # Using spectrum as the main seasonality measure
        'shift': np.round(shift_pelt(x), 4),
        'variability': np.round(coeff_variation(x), 4)
    }
    if step:
        res['step'] = np.round(step_pelt(x), 4)
    return res


