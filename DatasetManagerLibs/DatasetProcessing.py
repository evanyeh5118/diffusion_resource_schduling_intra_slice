import pandas as pd
import numpy as np

from .DatasetReader import DatasetReader
from .DeadbandReduction import DataReductionForDataUnit
from .Dataunit import DataUnit

class DatasetConvertor:
    def __init__(self):
        self.fingerDataUnits = {}
        self.idxsContextForward = {}
        self.idxsContextBackward = {}
        self.initialize()

    def initialize(self):
        self.configuration()
        
    def configuration(self, idxsContextForward=None, idxsContextBackward=None):
        if idxsContextForward is None:
            self.idxsContextForward = {
                "thumb": [1, 2, 3],
                "index": [5, 6, 7],
                "middle": [9, 10, 11],
                #"palm": [13, 14, 15],
            }
        if idxsContextBackward is None:
            self.idxsContextBackward = {
                "thumb": [4],
                "index": [8],
                "middle": [12],
            }

    def getDataUnit(self, unitName):
        return self.fingerDataUnits[unitName]

    def generateTrafficByDpdr(self, dbParameter=0.01, alpha=0.01, mode="fixed", direction="forward", upsampleK=None):
        for fingerName in ['thumb', 'index', 'middle']:
            print(f"========== {fingerName} ============")
            if direction == "forward":
                self.fingerDataUnits[f"{fingerName}_fr"].resampleContextData()
                if upsampleK is not None:
                    self.fingerDataUnits[f"{fingerName}_fr"].upsampleData(upsampleK)
                self.fingerDataUnits[f"{fingerName}_fr"].applyDpDr(dbParameter=dbParameter, alpha=alpha, mode=mode)
                self.fingerDataUnits[f"{fingerName}_fr"].interpolateCotextAfterDpDr()
                compressRate = self.fingerDataUnits[f"{fingerName}_fr"].compressionRate
                print(f"Forward: Compression rate:{compressRate}")
            else:
                self.fingerDataUnits[f"{fingerName}_bk"].resampleContextData()
                if upsampleK is not None:
                    self.fingerDataUnits[f"{fingerName}_bk"].upsampleData(upsampleK)
                self.fingerDataUnits[f"{fingerName}_bk"].applyDpDr(dbParameter=dbParameter, alpha=alpha, mode=mode)
                self.fingerDataUnits[f"{fingerName}_bk"].interpolateCotextAfterDpDr()
                compressRate = self.fingerDataUnits[f"{fingerName}_bk"].compressionRate
                print(f"Backward: Compression rate:{compressRate}")
            
    def registerDataUnit(self, dfInput):
        self.fingerDataUnits = {}
        for fingerName, idxsContext in self.idxsContextForward.items():
            dataUnit = DataUnit()
            dataUnit.name = fingerName
            dataUnit.setContextData(dfInput.iloc[:, idxsContext].to_numpy())
            dataUnit.timestamps = dfInput.iloc[:, 0].to_numpy()
            self.fingerDataUnits[f"{fingerName}_fr"] = dataUnit

        for fingerName, idxsContext in self.idxsContextBackward.items():
            dataUnit = DataUnit()
            dataUnit.name = fingerName
            dataUnit.setContextData(dfInput.iloc[:, idxsContext].to_numpy())
            dataUnit.timestamps = dfInput.iloc[:, 0].to_numpy()
            self.fingerDataUnits[f"{fingerName}_bk"] = dataUnit

  