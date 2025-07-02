import numpy as np


class WirelessModel():
    def __init__(self, params):
        self.r_min = 0
        self.r_max = params['B']
        self.r_list = np.arange(self.r_min, self.r_max).astype(int)
        self.N_r = len(self.r_list)
        self.MAX_arrival_packet = params['LEN_window']+1
        self.packetTransmissionCDF = None
        self.initialize()
        
    def initialize(self):
        RB_conso = RbRequirementExp(self.MAX_arrival_packet)
        RB_elements, RB_distribute = computeDistributionFromExp(RB_conso)
        self.computeCDF(RB_elements, RB_distribute)

    def successfulPacketCDF(self, r):
        r_idx = np.clip(np.searchsorted(self.r_list, r),0,self.N_r-1)
        return self.packetTransmissionCDF[r_idx]

    def computeCDF(self, RB_elements, RB_distribute):
        self.packetTransmissionCDF = np.zeros((self.N_r, ))
        for r in self.r_list:
            self.packetTransmissionCDF[r] = ComputeTailProb(r, 1, RB_elements, RB_distribute)

def ComputeTailProb(r, n, RB_elements, RB_distribute):
    elements = RB_elements[n]
    distribute = RB_distribute[n]
    return np.sum(distribute[r >= elements])

def computeDistributionFromExp(RB_conso):
    def CountFrequency(arr):
        unique_elements, counts = np.unique(arr, return_counts=True)
        return unique_elements, counts
    
    RB_distribute = []
    RB_elements = []
    for i in range(RB_conso.shape[1]):
        unique_elements, counts = CountFrequency(RB_conso[:,i])
        probs = counts / np.sum(counts)
        RB_elements.append(unique_elements)
        RB_distribute.append(probs)

    return RB_elements, RB_distribute

def RbRequirementExp(MAX_arrival_packet):
    # Constants
    alpha = 10 ** 13.6  # Coefficient of the path loss
    beta = 3.6          # Coefficient of the path loss
    _lambda = 1 / 2     # Parameter of the fast fading exponential distribution
    sigma = 3.5         # Parameter of the slow fading log-normal distribution
    G = 10 ** 0.3       # Antenna gain
    L = 10 ** 0.7       # Losses due to equipment imperfections
    PRBsize = 180000    # Bandwidth in Hz

    # Noise calculations
    bruitdB = -174 + 10 * np.log10(PRBsize)  # Noise in dBm
    NoiseRise = 6                            # To model interference in the uplink, add 6 dB on the noise
    bruit = 10 ** ((bruitdB + NoiseRise) / 10) / 1000  # Noise + interference in Watt

    # Power and efficiency parameters
    Pmax = 0.5  # Power in Watt
    a = 0.5
    b = 6

    # Simulation parameters
    K = 10000          # Number of simulations
    N = MAX_arrival_packet           # Number of packets
    Rayon = 1         # Cell radius
    packet_size = 100 * 8  # 0.1 KByte in bits
    RB_size = 180000       # Hz
    slot_length = 0.001    # 1 ms

    # Resource block consumption initialization
    RB_conso = np.zeros((K, N))

    # Simulation loop
    for k in range(K):
        distance = Rayon * np.sqrt(np.random.rand())
        phi = 2 * np.pi * np.random.rand()
        x = distance * np.cos(phi)
        y = distance * np.sin(phi)
        d = np.sqrt(x ** 2 + y ** 2)  # Distance to the base station

        # For each k (a experiment), we fix user's position
        #shadow = 10 ** (np.random.normal(0, sigma) / 10)
        shadow = 1.0
        Old_conso = 0
        for packet in range(N):
            path_loss = alpha * d ** beta / G * L / shadow / np.random.exponential(1 / _lambda)
            SINR = Pmax / path_loss / bruit
            Efficiency = a * min(np.log2(1 + SINR), b)
            Volume_per_RB = Efficiency * slot_length * RB_size  # How many bits can be sent per RB
            RB_per_packet = np.ceil(packet_size / Volume_per_RB)  # How many RBs are consumed by the packet

            RB_conso[k, packet] = Old_conso + RB_per_packet
            Old_conso += RB_per_packet
    return RB_conso
