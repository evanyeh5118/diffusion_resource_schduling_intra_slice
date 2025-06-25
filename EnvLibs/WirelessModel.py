import numpy as np

def successfulPacketCDF(r, a=0.5):
    # CDF of exponential distribution: F(r) = 1 - exp(-a*r)
    return 1 - np.exp(-a*r)