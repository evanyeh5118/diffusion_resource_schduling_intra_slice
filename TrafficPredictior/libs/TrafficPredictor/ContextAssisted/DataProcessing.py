import numpy as np
import pandas as pd
import math
import random

#import torch

from ..HelperFunctions import FindLastTransmissionIdx, DiscretizedTraffic


def PreparingDatasetHelper(dataUnit, params):
    #numWindow = params['numWindow']
    lenSource = params['lenSource']
    lenTarget = params['lenTarget']
    dataAugment = params['dataAugment'] # True;False
    smoothFc = params['smoothFc']
    smoothOrder = params['smoothOrder']
    contextDownsample = params['contextDownsample']
    newLenSource = None
    newLenTarget = None

    lenDataset = dataUnit.dataLength
    # ====== Preprocess ContextData ======
    contextData = dataUnit.getContextDataProcessedAndSmoothed(smoothFc, smoothOrder)
    contextDataNoSmooth = dataUnit.getContextDataProcessed()
    #=====================================
    transmissionFlags = dataUnit.getTransmissionFlags()

    sources, targets, lastTranmittedContext, transmissionsVector, trafficStatesSource, trafficStatesTarget, sourcesNoSmooth = [], [], [], [], [], [], []
    if dataAugment == True:
        idxs = [(i, FindLastTransmissionIdx(transmissionFlags, i)) 
                for i in range(lenSource, lenDataset - lenTarget)]
    else:
        idxs = [(i * lenTarget, FindLastTransmissionIdx(transmissionFlags, i * lenTarget)) 
                for i in range(int(lenSource/lenTarget), int(np.floor(lenDataset / lenTarget)))]
        
    for i, last_transmission_idx in idxs:
        sources.append(contextData[i-lenSource:i])
        targets.append(contextData[i:i+lenTarget])
        sourcesNoSmooth.append(contextDataNoSmooth[i-lenSource:i])
        transmissionsVector.append(transmissionFlags[i:i+lenTarget])
        trafficStatesSource.append(np.sum(transmissionFlags[i-lenSource:i]))
        trafficStatesTarget.append(np.sum(transmissionFlags[i:i+lenTarget]))
        lastTranmittedContext.append(contextData[last_transmission_idx:last_transmission_idx+1])

    if contextDownsample is not None:
        sources = [source[::contextDownsample] for source in sources]
        targets = [target[::contextDownsample] for target in targets]
        sourcesNoSmooth = [sourceNoSmooth[::contextDownsample] for sourceNoSmooth in sourcesNoSmooth]
        transmissionsVector = [transmissionFlag[::contextDownsample] for transmissionFlag in transmissionsVector]
        newLenSource = len(sources[0])
        newLenTarget = len(targets[0])
        # -----------------------------------------
        # Interpolate traffic states to match the downsampled length
        trafficStatesSource = [np.interp(
            trafficState, 
            np.linspace(0, lenSource+1, num=lenSource+1), 
            np.linspace(0.0,  newLenSource+1, num = lenSource+1)
        ) for trafficState in trafficStatesSource]
        trafficStatesTarget = [np.interp(
            trafficState, 
            np.linspace(0, lenTarget+1, num=lenTarget+1), 
            np.linspace(0.0, newLenTarget+1, num = lenTarget+1)
        ) for trafficState in trafficStatesTarget]

        
       
    trafficClassesTarget = DiscretizedTraffic(trafficStatesTarget) #[0 ~ L]
    return (
        (
            np.array(sources), 
            np.array(targets),
            np.array(lastTranmittedContext),
            np.array(trafficStatesSource).reshape(-1,1),
            np.array(trafficStatesTarget).reshape(-1,1),
            np.array(trafficClassesTarget).reshape(-1,1),
            np.array(transmissionsVector),
            np.array(sourcesNoSmooth)
        ),
        (newLenSource, newLenTarget)
    )

def PreparingDataset(dataUnit, parameters, verbose=True):
    trainRatio = parameters['trainRatio']

    train_size = int(trainRatio*dataUnit.dataLength)
    dataUnitTrain = dataUnit[:train_size]
    dataUnitTest = dataUnit[train_size:]

    if verbose == True:
        print(f"Train size: {dataUnitTrain.dataLength}, Test size: {dataUnitTest.dataLength}")

    datasetTrain, (newLenSource, newLenTarget) = PreparingDatasetHelper(dataUnitTrain, parameters)
    datasetTest, _ = PreparingDatasetHelper(dataUnitTest, parameters)
    return (
       datasetTrain,
       datasetTest,
       (newLenSource, newLenTarget)
    )





