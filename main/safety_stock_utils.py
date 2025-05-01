import pandas as pd
import numpy as np

from scipy import stats

from datetime import datetime
import glob
pd.options.mode.chained_assignment = None


def fit_discrete_kde(d_x):
    """
    Takes numpy array or pandas series, of demand and returns numpy array of discrete distribution following Gaussian KDE

    Parameters:
    d_x (numpy array or pandas series): periodic demand data (weekly, monthly etc)

    Returns:
    tuple: tuple of (x:discrete demand series, pmf: corresponding probability mass function)
    """
    #fit continous gaussian kernel density with scott bandwidth
    kde_dist = stats.gaussian_kde(d_x, bw_method='scott')

    #compute bandwith for upper and lower intervals for discrete distribution
    bandwidth = np.std(d_x, ddof=0) / (len(d_x)**(1/5))

    #set upper and lower intervals with a heuristic of 3 * bandwidth
    lower = np.floor(np.min(d_x) - 3 * bandwidth) 
    upper = np.ceil(np.max(d_x) + 3 * bandwidth)

    #range of discrete values from lower to upper intervals
    x = np.arange(lower,upper,step=1)

    #get probability densities given for x
    pmf = kde_dist.pdf(x) 

    #scale the pmf to equal to 1
    pmf = pmf/sum(pmf)

    ### deal with negative PMFs
    zero = x == 0 
    negative = x < 0 
    #Update the PMF at 0 
    pmf[zero] = pmf[zero] + pmf[negative].sum() 
    #Remove negative values 
    zero_arg = np.argmax(x==0) 
    pmf = pmf[zero_arg:] 
    x = x[zero_arg:].astype(int)
    return x, pmf

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
def simulate_safety_custom(d_x, d_pmf, L=4, R=1, alpha=0.95, time=200,pu_price = None):
    """
    Performs K timestep simulation over a given review period and lead time, Risk Period (L + R), 
    to set safety stock and evaluate Safety Stock metrics like achieved service level alpha, period service level alpha etc

    Parameters:
    d_x (numpy array): discrete demand
    d_pmf (numpy array): corresponding probability mass function
    L (int): Lead time (in weeks). Default is 4 weeks
    R (int): Review period. Default is 1 week
    alpha (float): desired service level. Default of 95%
    time (int): K number of time steps to run simulation. Default is 200
    pu_price (float): per unit price of item. Default is None

    Returns:
    tuple: tuple of (alpha: desired service level, SL_alpha: achieved service level from simulation, Ss: safety stock, S_value: safety stock value)
    """
    d_mu, d_std = attributes(d_pmf, d_x)
    d = random_values = np.random.choice(d_x, size=time, p=d_pmf)
    z = stats.norm.ppf(alpha)
    if z < 0:
        z = 0.0
    x_std = np.sqrt(L+R)*d_std
    Ss = np.round(x_std*z).astype(int)
    Cs = 1/2 * d_mu * R
    Is = d_mu * L
    S = Ss + 2*Cs + Is
    if pu_price:
        S_value = pu_price*0.05*Ss
    else:
        S_value = 0
    hand = np.zeros(time)
    transit = np.zeros((time,L+1))
    hand[0] = S - d[0]
    transit[0,-1] = d[0]
    stockout_period = np.full(time,False,dtype=bool)
    stockout_cycle = []
    for t in range(1,time):
        if transit[t-1,0]>0:
            stockout_cycle.append(stockout_period[t-1])
        hand[t] = hand[t-1] - d[t] + transit[t-1,0]
        stockout_period[t] = hand[t] < 0
        hand[t] = max(0,hand[t]) #Uncomment if excess demand result in lost sales rather than backorders
        transit[t,:-1] = transit[t-1,1:]
        if 0==t%R:
            net = hand[t] + transit[t].sum()
            transit[t,L] = S - net
    df = pd.DataFrame(data= {'Demand':d,'On-hand':hand,'In-transit':list(transit)})
    df = df.iloc[R+L:,:] #Remove initialization periods
    alpha = round(alpha * 100,1)
    if len(stockout_cycle) != 0:
        SL_alpha = round((1-sum(stockout_cycle)/len(stockout_cycle))*100,1)
    else:
        SL_alpha = round((1-sum(stockout_cycle))*100,1)
    if len(stockout_period) != 0:
        SL_period = round((1-sum(stockout_period)/time)*100,1)
    else:
        SL_period = round((1-sum(stockout_period))*100,1)
    return alpha, SL_alpha, SL_period,Ss, S_value

