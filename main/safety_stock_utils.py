import pandas as pd
import numpy as np

from scipy import stats

from datetime import datetime
import glob
pd.options.mode.chained_assignment = None


def fit_discrete_kde(d_x, bw_method='scott', cut=3, support_min=0):
    """
    Fit Gaussian KDE to data d_x, evaluate on integer grid, and return a discrete PMF.
    Probability mass below `support_min` is merged into the first valid bin.
    """
    d_x = np.asarray(d_x, dtype=float)

    # KDE
    kde = stats.gaussian_kde(d_x, bw_method=bw_method)

    # Bandwidth and range (inclusive)
    bandwidth = kde.factor * np.std(d_x, ddof=1)
    lower = int(np.floor(d_x.min() - cut * bandwidth))
    upper = int(np.ceil(d_x.max() + cut * bandwidth))
    if upper < lower:
        lower, upper = upper, lower
    x = np.arange(lower, upper + 1, dtype=int)

    # Evaluate and normalize to PMF (Δx = 1)
    pdf_vals = kde.pdf(x.astype(float))
    total = pdf_vals.sum()
    if total == 0:
        # Degenerate fallback: put all mass at nearest integer to mean
        xi = int(np.rint(d_x.mean()))
        return np.array([xi], dtype=int), np.array([1.0], dtype=float)
    pmf = pdf_vals / total

    # Truncate below support_min and merge spill into first valid bin
    if support_min is not None:
        mask = x >= support_min
        if not mask.any():
            # Everything is below support_min → put all mass at support_min
            return np.array([int(support_min)], dtype=int), np.array([1.0], dtype=float)

        spill = pmf[~mask].sum()
        first_idx = np.where(mask)[0][0]          # index in original arrays
        pmf[first_idx] += spill                   # add to original pmf
        # Now actually truncate arrays
        pmf = pmf[mask]
        x = x[mask]

    # Final renormalization (protect against FP drift)
    pmf = pmf / pmf.sum()

    return x.astype(int), pmf

### function to get distribution attributes

def attributes(pmf,x): 
    """
    Takes discrete demand series and corresponding probability mass function and returns discrete distribution attributes

    Parameters:
    pmf (numpy array): corresponding probability mass function
    x (numpy array): discrete demand

    Returns:
    tuple: tuple of (mu: expected mean of the distribution, std: standard deviation of the distribution)
    """
    mu = sum(pmf*x) 
    var = sum(((x-mu)**2)*pmf)
    std = np.sqrt(var)
    return mu, std 


### Safety Stock Over Risk Period (L + R): Fitting Gaussian Kernel Density, simulation
def simulate_safety_custom(
    d_x, d_pmf, L=4, R=1, alpha=0.95, time=200, pu_price=None, holding_rate=1, seed=111
):
    """
    Simulate inventory performance under stochastic demand using an empirical 
    safety stock estimate derived from a discrete demand distribution.

    Parameters
    ----------
    d_x : array-like
        Discrete demand values (support of the demand distribution).
    d_pmf : array-like
        Probability mass function corresponding to `d_x`.
    L : int, default=4
        Lead time (in periods) for replenishment.
    R : int, default=1
        Review period (in periods) between orders.
    alpha : float, default=0.95
        Service level target (quantile for empirical safety stock computation).
    time : int, default=200
        Number of simulation periods.
    pu_price : float, optional
        Unit purchase price used to estimate the holding cost value of safety stock.
    holding_rate : float, default=1
        Holding cost rate applied to the safety stock value.
    seed : int, default=111
        Random seed for reproducibility.

    Returns
    -------
    tuple
        (target_service_level, simulated_cycle_service_level, 
         simulated_period_service_level, safety_stock_units, safety_stock_value)
         
        where:
        - target_service_level : float — target α service level (in percentage).
        - simulated_cycle_service_level : float — achieved service level per cycle.
        - simulated_period_service_level : float — achieved service level per period.
        - safety_stock_units : int — computed empirical safety stock quantity.
        - safety_stock_value : float — cost value of safety stock.

    Notes
    -----
    - The empirical safety stock is estimated as the α-quantile of the total demand 
      over the combined lead and review period minus its mean.
    - The simulation tracks on-hand and in-transit inventories and identifies 
      stockout events across periods and cycles.
    - A DataFrame of inventory trajectories is generated internally but not returned.

    """
    if seed:
        np.random.seed(seed)
        
    # --- Demand distribution attributes ---
    d_mu, d_std = attributes(d_pmf, d_x)
    # print(f"mean: {d_mu} and std: {d_std}")
    d = np.random.choice(d_x, size=time, p=d_pmf)
    # print(f"demand: {d}")
    # --- Empirical safety stock based on KDE quantile ---
    period_demand_samples = np.random.choice(d_x, size=(1000, L+R), p=d_pmf)
    # print(f"sample: {period_demand_samples}")
    total_demand_LR = np.sum(period_demand_samples, axis=1)
    ss_empirical = np.quantile(total_demand_LR, alpha) - np.mean(total_demand_LR)
    ss_empirical = max(0.0, ss_empirical)
    Ss = np.round(ss_empirical).astype(int)
    # print(f"total_demand: {total_demand_LR}, ss_empirical: {ss_empirical}, ss: {Ss}")

    # --- Stock components ---
    Cs = 0.5 * d_mu * R
    Is = d_mu * L
    S = Ss + 2*Cs + Is

    # --- Cost impact ---
    S_value = round(pu_price * holding_rate * Ss,4) if pu_price else 0.0000

    # --- Inventory simulation ---
    hand = np.zeros(time)
    transit = np.zeros((time, L+1))
    stockout_period = np.full(time, False, dtype=bool)
    stockout_cycle = []

    hand[0] = S - d[0]
    transit[0, -1] = d[0]

    for t in range(1, time):
        if transit[t-1, 0] > 0:
            stockout_cycle.append(stockout_period[t-1])
        hand[t] = hand[t-1] - d[t] + transit[t-1, 0]
        stockout_period[t] = hand[t] < 0
        hand[t] = max(0, hand[t])
        transit[t, :-1] = transit[t-1, 1:]
        if t % R == 0:
            net = hand[t] + transit[t].sum()
            transit[t, L] = S - net

    df = pd.DataFrame({'Demand': d, 'On-hand': hand, 'In-transit': list(transit)})
    df = df.iloc[L+R:, :]

    # SL_cycle = round((1 - np.mean(stockout_cycle)) * 100, 1)
    # SL_period = round((1 - np.mean(stockout_period)) * 100, 1)
    SL_cycle = round((1 - np.mean(stockout_cycle)), 4)
    SL_period = round((1 - np.mean(stockout_period)), 4)
    
    return round(alpha * 100, 1), SL_cycle, SL_period, Ss, S_value


def simulate_safety_custom_norm(
    d_mu, d_std, L=4, R=1, alpha=0.95, time=200, pu_price=None, holding_rate=1, seed=111
):
    """
    Simulate inventory performance using a normal demand distribution and 
    compute safety stock based on the normal quantile method.

    Parameters
    ----------
    d_mu : float
        Mean of the normal demand distribution.
    d_std : float
        Standard deviation of the normal demand distribution.
    L : int, default=4
        Lead time (in periods) for replenishment.
    R : int, default=1
        Review period (in periods) between orders.
    alpha : float, default=0.95
        Target service level (used for safety stock quantile).
    time : int, default=200
        Number of simulation periods.
    pu_price : float, optional
        Unit price for estimating holding cost value of safety stock.
    holding_rate : float, default=1
        Holding cost rate applied to the safety stock.
    seed : int, default=111
        Random seed for reproducibility.

    Returns
    -------
    tuple
        (target_service_level, simulated_cycle_service_level, 
         simulated_period_service_level, safety_stock_units, safety_stock_value)
         
        where:
        - target_service_level : float — target α service level (in percentage).
        - simulated_cycle_service_level : float — achieved service level per cycle.
        - simulated_period_service_level : float — achieved service level per period.
        - safety_stock_units : int — computed safety stock quantity.
        - safety_stock_value : float — estimated cost of safety stock.

    Notes
    -----
    - Demand is generated from a normal distribution truncated at zero.
    - Safety stock is computed as z * σ√(L+R), where z is the α-quantile.
    - The simulation tracks on-hand, in-transit inventory, and stockouts.
    """
    if seed:
        np.random.seed(seed)
        
    d = np.maximum(np.random.normal(d_mu,d_std,time).round(0).astype(int),0)
    z = stats.norm.ppf(alpha)
    if z < 0:
        z = 0.0
    x_std = np.sqrt(L+R)*d_std
    Ss = np.round(x_std*z).astype(int)
    Ss = max(0.0, Ss)
    Cs = 1/2 * d_mu * R
    Is = d_mu * L
    S = Ss + 2*Cs + Is

    # --- Cost impact ---
    S_value = round(pu_price * holding_rate * Ss,4) if pu_price else 0

    # --- Inventory simulation ---
    hand = np.zeros(time)
    transit = np.zeros((time, L+1))
    stockout_period = np.full(time, False, dtype=bool)
    stockout_cycle = []

    hand[0] = S - d[0]
    transit[0, -1] = d[0]

    for t in range(1, time):
        if transit[t-1, 0] > 0:
            stockout_cycle.append(stockout_period[t-1])
        hand[t] = hand[t-1] - d[t] + transit[t-1, 0]
        stockout_period[t] = hand[t] < 0
        hand[t] = max(0, hand[t])
        transit[t, :-1] = transit[t-1, 1:]
        if t % R == 0:
            net = hand[t] + transit[t].sum()
            transit[t, L] = S - net

    df = pd.DataFrame({'Demand': d, 'On-hand': hand, 'In-transit': list(transit)})
    df = df.iloc[L+R:, :]

    # SL_cycle = round((1 - np.mean(stockout_cycle)) * 100, 1)
    # SL_period = round((1 - np.mean(stockout_period)) * 100, 1)
    SL_cycle = round((1 - np.mean(stockout_cycle)), 4)
    SL_period = round((1 - np.mean(stockout_period)), 4)
    
    return round(alpha * 100, 1), SL_cycle, SL_period, Ss, S_value
    

