import numpy as np
from scipy.signal import convolve
from math import ceil

class WirelessModel():
    def __init__(self, params):
        self.r_min = 0
        self.r_max = params['B']
        self.r_list = np.arange(self.r_min, self.r_max).astype(int)
        self.N_r = len(self.r_list)
        self.packetTransmissionCDF = None
        self.initialize()
        
    def initialize(self):
        self.analyticalModel = WirelessAnalyticalModel()
        RB_elements, RB_distribute = self.analyticalModel.getRbPmfAnalytical()
        self.computeCDF(RB_elements, RB_distribute)

    def successfulPacketCDF(self, r):
        r_idx = np.clip(np.searchsorted(self.r_list, r), 0, self.N_r-1)
        return self.packetTransmissionCDF[r_idx]

    def computeCDF(self, RB_elements, RB_distribute):
        self.packetTransmissionCDF = np.zeros((self.N_r, ))
        for r in self.r_list:
            self.packetTransmissionCDF[r] = np.sum(RB_distribute[r >= RB_elements])

class WirelessAnalyticalModel:
    def __init__(self, param=None, D_range=None, N_D_points=None):
        self.params = param
        self.D_range = D_range
        self.N_D_points = N_D_points
        self.RB_elements = None
        self.RB_distribute = None
        self.RB_elements_analy = None
        self.RB_distribute_analy = None
        self.initialize()

    def initialize(self):
        if self.params is None:
            self.params = {}
            self.setDefaultParams()
        if self.D_range is None:
            self.D_range=(1e-20, 1.0)
        if  self.N_D_points is None:
            self.N_D_points=1000

    def setDefaultParams(self):
        # Set 1
        self.params["alpha"] = 10 ** 13.6       # Large path loss exponent
        self.params["beta"] = 3.6               # Path loss exponent
        self.params["G"] = 10 ** 0.3            # Antenna gain
        self.params["L"] = 10 ** 0.7            # Losses
        self.params["PRBsize"] = 180000         # 180 kHz
        self.params["noise_dB"] = -174 + 10 * np.log10(self.params["PRBsize"]) + 6  # +6 dB for NoiseRise
        self.params["bruit"] = 10 ** (self.params["noise_dB"] / 10) / 1000           # Noise + interference in watts
        self.params["Pmax"] = 0.5               # Maximum power
        
        # Set 2
        self.params["packet_size_bits"] = 100 * 8   # 0.1 KB in bits
        self.params["slot_length"] = 0.001          # 1 ms
        self.params["K"] = self.params["packet_size_bits"] / (self.params["PRBsize"]* self.params["slot_length"])  # Transmission efficiency factor
        self.params["a"] = 0.9
        self.params["E_max"] = 6      # dB
        self.params["E_min"] = 0.01  # dB
        self.params["lambda_"] = 1 / 2  # Parameter of the fast fading exponential distribution

    #===========================================================================
    #=========================Analytical Result=================================
    #===========================================================================
    def getRbPmfAnalytical(self):
        """
        Computes the PMF of rho when summing over N packets, averaging across
        distances in [D_range[0], D_range[1]] split into N_D_points.
        """
        D_values = np.linspace(self.D_range[0], self.D_range[1], self.N_D_points)
        RB_distribute = None

        for D in D_values:
            d = np.sqrt(D)
            pmf_single, RB_elements, _ = self.singlePacketPmfGivenD(d)
            if RB_distribute is None:
                RB_distribute = pmf_single
            else:
                RB_distribute += pmf_single

        RB_distribute /= self.N_D_points
        return RB_elements, RB_distribute
    
    def fromSingleToMultiPackagePmf(self, x_pmf, N):
        """
        Computes the PMF of S = X_1 + X_2 + ... + X_N (sum of N i.i.d. random variables),
        where each X_i has PMF x_pmf.
        """
        pmf_values = x_pmf.copy()
        for _ in range(1, N):
            pmf_values = convolve(pmf_values, x_pmf, mode='full')

        pmf_values /= pmf_values.sum()
        new_range = np.arange(len(pmf_values))
        return pmf_values, new_range

    def singlePacketPmfGivenD(self, D):
        """
        Computes the PMF of the number of resource units (rho) for a single packet,
        given distance D.
        """
        K, a, E_min, E_max, lam = (
            self.params["K"], self.params["a"],  self.params["E_min"],  self.params["E_max"],  self.params["lambda_"])

        cD = self.cFunc(D)
        overlineF_th = (2.0**(E_max) - 1.0) / cD
        underlineF_th = (2.0**(E_min) - 1.0) / cD

        rho_min = ceil(K / (a * E_max))   # underline{rho}
        rho_max = ceil(K / (a * E_min))   # overline{rho}

        F_min = (2.0**(K/(a*rho_max)) - 1.0) / cD
        F_max = (2.0**(K/(a*rho_min)) - 1.0) / cD

        rho_range = np.arange(rho_max + 1)
        p = np.zeros_like(rho_range, dtype=float)

        for i, r in enumerate(rho_range):
            if r < rho_min:
                p[i] = 0
            elif r == rho_min:
                p[i] = np.exp(-lam * min(F_max, overlineF_th))
            elif r == rho_max:
                p[i] = 1.0 - np.exp(-lam * underlineF_th)
            else:
                term1 = np.exp(-lam * ((2.0**(K/(a*r)) - 1.0) / cD))
                term2 = np.exp(-lam * ((2.0**(K/(a*(r - 1))) - 1.0) / cD))
                p[i] = term1 - term2

        p /= p.sum()

        return p, rho_range, (F_min, overlineF_th, F_max, underlineF_th)
    
    def cFunc(self, D):
        """
        Computes the quantity c(D) = G*Pmax / (alpha * D^beta * L * bruit).
        """
        return (
            (self.params["G"] * self.params["Pmax"]) / 
            (self.params["alpha"] * (D ** self.params["beta"]) * self.params["L"] * self.params["bruit"])
        )