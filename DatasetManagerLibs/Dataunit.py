from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from .DeadbandReduction import DataReductionForDataUnit

class DataUnit:
    def __init__(self):
        self.name = []
        self.Ts = []
        self.timestamps = []
        self.contextData = []
        self.contextDataDpDr = []
        self.contextDataPorcessed = []
        self.transmitionFlags = []
        self.dimFeatures = []
        self.dataLength = []
        self.compressionRate = []

    def __getitem__(self, key):
        dataUnitCopy = DataUnit()

        dataUnitCopy.contextData = self.contextData[key]
        dataUnitCopy.contextDataDpDr = self.contextDataDpDr[key]
        dataUnitCopy.contextDataPorcessed = self.contextDataPorcessed[key]
        dataUnitCopy.transmitionFlags = self.transmitionFlags[key]

        dataUnitCopy.name = self.name
        dataUnitCopy.Ts = self.Ts
        dataUnitCopy.dataLength = dataUnitCopy.contextData.shape[0]
        dataUnitCopy.compressionRate = np.sum(dataUnitCopy.transmitionFlags) / len(dataUnitCopy.transmitionFlags)
        dataUnitCopy.dimFeatures = self.dimFeatures
        dataUnitCopy.timestamps = self.timestamps[key] - self.timestamps[0]
        return dataUnitCopy

    def setContextData(self, contextData):
        self.contextData = contextData
        self.dataLength = contextData.shape[0]
        self.dimFeatures = contextData.shape[1]

    def getContextDataProcessed(self):
        data = self.contextDataPorcessed.copy()
        min_values = data.min(axis=0)
        max_values = data.max(axis=0)
        normalizedData = (data - min_values) / (max_values - min_values)
        if normalizedData.ndim == 1:
            normalizedData  = normalizedData[..., np.newaxis]
        return normalizedData
    
    def getContextDataProcessedAndSmoothed(self, fc, order): #self.contextDataPorcessed -> #self.contextDataPorcessedSmoothed
        smoothData = smoothDataByFiltfilt(self.contextDataPorcessed, fc, 1/self.Ts, order)
        min_values = smoothData.min(axis=0)
        max_values = smoothData.max(axis=0)
        normalizedData = (smoothData - min_values) / (max_values - min_values)
        if normalizedData.ndim == 1:
            normalizedData  = normalizedData[..., np.newaxis]
        return normalizedData

    def getTransmissionFlags(self):
        return self.transmitionFlags.copy()

    def display(self):
        print(f"Name: {self.name}, Ts:{self.Ts}, Data length:{self.dataLength}, Dim of context:{self.dimFeatures}, Compression rate:{self.compressionRate}")

    def generateTrafficPattern(self, lenWindow):
        traffic_state = []
        N_slot = int(np.floor(len(self.transmitionFlags)/lenWindow))
        for i in range(N_slot):
            traffic_state.append(np.sum(self.transmitionFlags[i*lenWindow:(i+1)*lenWindow]))
        return np.array(traffic_state)
    
    def interpolateCotextAfterDpDr(self): #self.contextDataDpDr -> #self.contextDataPorcessed
        self.contextDataPorcessed = interpolationData(
            np.asarray(self.transmitionFlags).astype(int), 
            np.asarray(self.contextDataDpDr, dtype=np.float64), 
            np.asarray(self.timestamps))
        
    def applyDpDr(self, dbParameter=0.01, alpha=0.01, mode="fixed"): #self.contextData -> #self.contextDataDpDr
        contextDataDpDr, transmitionFlags = DataReductionForDataUnit(self, dbParameter=dbParameter, alpha=alpha, mode=mode)
        self.contextDataDpDr = contextDataDpDr
        self.transmitionFlags = transmitionFlags
        self.compressionRate = np.sum(self.transmitionFlags) / self.transmitionFlags.shape[0]

    def resampleContextData(self): #self.contextData -> self.contextData
        self.Ts = round(np.mean(self.timestamps[1:]-self.timestamps[0:-1]), 2)
        (self.timestamps, contextData) = resampleData(self.timestamps, self.contextData)
        self.setContextData(contextData)

    def upsampleData(self, K):
        (self.timestamps, contextData) = upsampleData(self.contextData, self.Ts, K, method='linear')
        self.Ts = self.Ts / K
        self.setContextData(contextData)


def interpolationData(flags, data, time):
    if data.ndim == 1:
        data = data[:, np.newaxis]  # Convert (T,) -> (T, 1)
    
    T, N = data.shape  

    valid_indices = flags == 1  # Where flags are 0 (valid data)
    flagged_indices = flags == 0  # Where flags are 1 (to be interpolated)

    valid_time = time[valid_indices]
    valid_data = data[valid_indices]  # Shape: (num_valid, N)
    interpolated_data = np.copy(data)
    for i in range(N):
        interp_func = interp1d(valid_time, valid_data[:, i], kind='linear', fill_value="extrapolate")
        interpolated_data[flagged_indices, i] = interp_func(time[flagged_indices])

    if interpolated_data.shape[1] == 1:
        interpolated_data = interpolated_data.flatten()

    return interpolated_data


def resampleData(timestamps, data):
    timestamps = np.array(timestamps)
    data = np.array(data)
    
    # Ensure timestamps and data have matching lengths
    if len(timestamps) != len(data):
        raise ValueError("timestamps and data must have the same length.")
    
    # Define target number of samples to match input size
    num_samples = len(timestamps)
    
    # Define the new timestamps ensuring the same length as input
    start_time, end_time = timestamps[0], timestamps[-1]
    new_timestamps = np.linspace(start_time, end_time, num_samples)
    
    # Ensure data is at least 2D
    if data.ndim == 1:
        data = data[:, np.newaxis]
    
    # Interpolate each dimension separately
    new_data = np.column_stack([
        interp1d(timestamps, data[:, i], kind='linear', fill_value='extrapolate', assume_sorted=False)(new_timestamps)
        for i in range(data.shape[1])
    ])
    
    return new_timestamps, new_data

def upsampleData(context_data, Ts, K, method='linear'):
    n, d = context_data.shape
    original_time = np.arange(n) * Ts
    new_time = np.arange(0, (n - 1) * Ts + Ts / K, Ts / K)

    upsampled_data = np.zeros((len(new_time), d))
    for i in range(d):
        interp_func = interp1d(original_time, context_data[:, i], kind=method)
        upsampled_data[:, i] = interp_func(new_time)
    
    return new_time, upsampled_data


def smoothDataByFiltfilt(x, fc, fs, order):
    """
    Applies a Butterworth low-pass filter to a NumPy array.

    Parameters:
    x (np.ndarray): Input data of shape (len_data, dim).
    fc (float): Cutoff frequency of the filter.
    fs (float): Sampling frequency.
    order (int): Order of the Butterworth filter.

    Returns:
    np.ndarray: Filtered data of the same shape as x.
    """
    if not isinstance(x, np.ndarray):
        raise ValueError("Input must be a NumPy array.")
    if x.ndim != 2:
        x = x.reshape(-1, 1)
        #raise ValueError("Input array must be 2D with shape (len_data, dim).")
    
    # Butterworth filter design
    nyquist = 0.5 * fs  # Nyquist frequency
    normalized_cutoff = fc / nyquist
    b, a = butter(order, normalized_cutoff, btype='low', analog=False)
    
    # Apply the filter along axis 0 (time dimension)
    filtered_data = np.apply_along_axis(lambda col: filtfilt(b, a, col), axis=0, arr=x)
    
    return filtered_data