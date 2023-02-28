import numpy as np

def exponential_cost(start, end, cumm_data):
    """
    Creates an array of segment costs for a time series with an
    exponential distribution with changing mean
    
    Args:
        start: int or array if ints
            start indeces for segments
        end: int or array if ints
            end indeces for segments
        cumm_data: array
            cumulative sum of the original time series data
    
    """
    return -1*(start-end)*(np.log(start-end)-np.log(cumm_data[start]-cumm_data[end]))         
    


def pelt(data, pen=None):
    """ PELT algorithm to compute changepoints in time series with exponential distribution
    Ported from:
        https://github.com/STOR-i/Changepoints.jl
        https://github.com/rkillick/changepoint/
    Reference:
        Killick R, Fearnhead P, Eckley IA (2012) Optimal detection
            of changepoints with a linear computational cost, JASA
            107(500), 1590-1598
    Args:
        data: np.array
            time series data to segment
        pen: float (optional)
            defaults to log(n)
    Returns:
        (:obj:`list` of int): List with the indexes of changepoints
    """
    
    length = len(data)
    data = np.hstack(([0.0], np.array(data)))
    cumm = np.cumsum(data)
    
    if pen is None:
        pen = np.log(length)

    # F[t] is optimal cost of segmentation upto time t
    F = np.zeros(length + 1)
    F[0] = -pen
    
    # last changepoint prior to time t
    R = np.array([0], dtype=np.int)

    # vector of candidate changepoints at t
    candidates = np.zeros(length + 1, dtype=np.int)
    
    for tstar in range(2, length + 1):
        cpt_cands = R
        seg_costs = exponential_cost(tstar, cpt_cands, cumm)
        F_cost = F[cpt_cands] + seg_costs
        F[tstar], tau = min(F_cost) + pen, np.argmin(F_cost)
        candidates[tstar] = cpt_cands[tau]

        # pruning step
        ineq_prune =  F_cost < F[tstar] 
        R = list(cpt_cands[ineq_prune]) 
        R.append(tstar - 1)
        R = np.array(R, dtype=np.int)  
             
    # get changepoints
    last = candidates[-1]
    changepoints = [last]
    while last > 0:
        last = candidates[last]
        changepoints.append(last)

    return sorted(changepoints)