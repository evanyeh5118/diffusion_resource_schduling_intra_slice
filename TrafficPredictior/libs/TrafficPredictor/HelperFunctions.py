from scipy.signal import butter, filtfilt
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

def SmoothFilter(df, fc, fs, order):
    # Ensure the input is a DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a Pandas DataFrame.")
    # Butterworth filter design
    nyquist = 0.5 * fs  # Nyquist frequency
    normalized_cutoff = fc / nyquist
    b, a = butter(order, normalized_cutoff, btype='low', analog=False)
    # Apply the filter to each column
    filtered_data = df.apply(lambda col: filtfilt(b, a, col), axis=0)
    # Return the filtered data as a DataFrame with the same index and columns
    return pd.DataFrame(filtered_data, index=df.index, columns=df.columns)

def DiscretizedTraffic(data):    
    outputs = []
    for d in data:
        # Ensure the value is finite and convert to integer
        if np.isfinite(d):
            outputs.append(int(round(float(d))))
        else:
            outputs.append(0)  # Default value for invalid data
    return np.array(outputs, dtype=np.int64)

def FindLastTransmissionIdx(transmission, current_idx):
    while current_idx-1 >= 0:
        current_idx = current_idx-1
        if transmission[current_idx] == 1:
            return current_idx
    return 0

def createDataLoaders(batch_size, dataset, shuffle=True):
    # Convert all input data into tensors and stack them
    tensor_list = [torch.stack([torch.from_numpy(d).float() for d in data]) for data in dataset]
    
    num_samples = tensor_list[0].shape[0]
    assert all(t.shape[0] == num_samples for t in tensor_list), "All input tensors must have the same number of samples."
    
    # Create a dataset
    dataset = TensorDataset(*tensor_list)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)

    return dataloader

# Calculate total parameters
def countModelParameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def moving_average_smoothing_optimized(data, window_size=5):
    """
    Applies a moving average smoothing to a 3D tensor of shape [len, batch_size, dim]
    without using a for-loop.
    """
    # Ensure data is 3D: [len, batch_size, dim]
    assert data.ndim == 3, "Input data must be of shape [len, batch_size, dim]"

    # Calculate padding size
    pad_size = window_size // 2
    
    # Manually create reflection padding along the time axis (dim=0)
    data_padded = torch.cat(
        [data[:pad_size].flip(0), data, data[-pad_size:].flip(0)], dim=0
    )
    
    # Reshape for convolution: [batch_size * dim, 1, len]
    data_padded_reshaped = data_padded.permute(1, 2, 0).reshape(-1, 1, data_padded.size(0))
    
    # Applying the moving average using 1D convolution
    kernel = torch.ones(1, 1, window_size, device=data.device) / window_size
    smoothed_reshaped = F.conv1d(data_padded_reshaped, kernel, padding=0)
    
    # Reshape back to original format [len, batch_size, dim]
    smoothed_data = smoothed_reshaped.view(data.size(1), data.size(2), -1).permute(2, 0, 1)
    
    # Truncate to match the original length
    smoothed_data = smoothed_data[:data.size(0)]
    
    return smoothed_data

def exponential_moving_average_smoothing(data, alpha=0.1):
    """
    Applies exponential moving average smoothing to a 3D tensor of shape [len, batch_size, dim].
    """
    weights = torch.pow(1 - alpha, torch.arange(data.size(0), device=data.device)).unsqueeze(1).unsqueeze(2)
    weighted_data = data * weights
    cumulative_sum = torch.cumsum(weighted_data, dim=0)
    
    normalizer = torch.cumsum(weights, dim=0)
    smoothed_data = cumulative_sum / normalizer
    
    return smoothed_data
'''
def exponential_moving_average_smoothing(data, alpha=0.1):
    """
    Applies exponential moving average smoothing to a 3D tensor of shape [len, batch_size, dim].
    """
    smoothed_data = torch.zeros_like(data)
    smoothed_data[0] = data[0]

    for t in range(1, data.size(0)):
        smoothed_data[t] = alpha * data[t] + (1 - alpha) * smoothed_data[t - 1]

    return smoothed_data
'''