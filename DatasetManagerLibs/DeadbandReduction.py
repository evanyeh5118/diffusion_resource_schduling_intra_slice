import numpy as np
import math
import pandas as pd

#from .DatasetProcessing import DataUnit 

class DeadbandDataReduction:
    def __init__(self, dbParameter, dim_feature):
        self.DeadbandParameter = dbParameter
        self.previousSample = np.zeros(dim_feature)

    def update(self, currentSample):
        updated_sample = np.zeros_like(currentSample)
        # Compute the difference between the recently transmitted signal and the current signal
        difference_mag = math.sqrt(np.sum((currentSample-self.previousSample)**2))
        previous_mag = math.sqrt(np.sum((self.previousSample)**2))

        # Check whether the difference is above the perceptual threshold
        if difference_mag >= self.DeadbandParameter*previous_mag:
            # Transmit the current signal
            transmit_flag = True
            updated_sample = currentSample
            self.previousSample = currentSample
        else:
            # Do not transmit the current signal
            transmit_flag = False
            updated_sample = self.previousSample
        return (updated_sample, transmit_flag)
    
class DeadbandDataReductionAaptive:
    def __init__(self, dbParameter, dim_feature, alpha):
        self.DeadbandParameter = dbParameter
        self.previousSample = np.zeros(dim_feature)
        self.lastSample = np.zeros(dim_feature)
        self.alpha = alpha

    def update(self, currentSample):
        updated_sample = np.zeros_like(currentSample)
        # Compute the difference between the recently transmitted signal and the current signal
        difference_mag = math.sqrt(np.sum((currentSample-self.previousSample)**2))
        delta_x = math.sqrt(np.sum((currentSample-self.lastSample)**2))
        previous_mag = math.sqrt(np.sum((self.previousSample)**2))

        DP_parameter = (self.DeadbandParameter + self.alpha*delta_x)
        # Check whether the difference is above the perceptual threshold
        if difference_mag >= DP_parameter*previous_mag:
            # Transmit the current signal
            transmit_flag = True
            updated_sample = currentSample
            self.previousSample = currentSample
        else:
            # Do not transmit the current signal
            transmit_flag = False
            updated_sample = self.previousSample

        self.lastSample = currentSample    
        return (updated_sample, transmit_flag)

def DataReductionForDataUnit(dataUnit, dbParameter=0.01, alpha=0.01, mode="fixed"):
    if mode == "fixed":
        dataReductor = DeadbandDataReduction(
            dbParameter=dbParameter, dim_feature=dataUnit.dimFeatures)
    else:
        dataReductor = DeadbandDataReductionAaptive(
            dbParameter=dbParameter, dim_feature=dataUnit.dimFeatures, alpha=alpha
        )

    contextDataDpDr = np.zeros((dataUnit.dataLength, dataUnit.dimFeatures))
    transmitionFlags = np.zeros((dataUnit.dataLength, ))

    for i in range(dataUnit.dataLength):
        (
            contextDataDpDr[i,:], transmitionFlags[i]
        ) = dataReductor.update(dataUnit.contextData[i])

    return contextDataDpDr, transmitionFlags

def ApplyZohToMissingData(df_source, data_idxs, transmission):
    # Concatenate target indices for setting up df_target
    output = np.zeros((df_source.shape[0], len(data_idxs)))
    
    # Iterate through each row
    for i in range(df_source.shape[0]):
        # Copy data if transmission flag is 1
        if transmission[i] == 1:
            output[i, :] = df_source.iloc[i, data_idxs].to_numpy()
        else:
            # Fill with previous row values if transmission flag is 0 (and if not the first row)
            if i > 0:
                output[i, :] = output[i-1, :]
            else:
                output[i, :] = df_source.iloc[i, data_idxs].to_numpy()
    df_output = pd.DataFrame(output, columns=df_source.columns[data_idxs])
    df_output.insert(0, value=df_source['Time'], column='Time')
    df_output['transmission'] = transmission
    return df_output, df_source.columns[data_idxs]

def ApplyInterpToMissingData(df_source, data_idxs, transmission):
    df_interp = df_source.iloc[:,data_idxs].copy()
    df_interp.loc[transmission==False, :] = np.nan
    df_interp = df_interp.interpolate(method='linear')
    df_interp.insert(0, column='Time', value=df_source['Time'])
    df_interp['transmission'] = transmission
    return df_interp, df_source.columns[data_idxs]