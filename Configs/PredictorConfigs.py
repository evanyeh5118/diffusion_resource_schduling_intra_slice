

def getPredictorConfig(configIdx):
    if configIdx == 0:
        return {
            'LEN_window': 200,
            'upsampleK': 10,
            'dataflow': "thumb_fr",
            'dbParameter': 0.001,
            'alpha': 0.01,
            'mode': "fixed",
            'direction': "forward",
            'train_ratio': 0.6,
            'trainDataAugment': False,
            'smoothFc': 1.5,
            'smoothOrder': 3,
        }
    elif configIdx == 1:
        return {
            'LEN_window': 200,
            'upsampleK': 10,
            'dataflow': "thumb_bk",
            'dbParameter': 0.012,
            'alpha': 0.01,
            'mode': "fixed",
            'direction': "backward",
            'train_ratio': 0.6,
            'trainDataAugment': False,
            'smoothFc': 1.5,
            'smoothOrder': 3,
        }
    elif configIdx == 2:
        return {
            'LEN_window': 200,
            'upsampleK': 10,
            'dataflow': "thumb_bk",
            'dbParameter': 0.012,
            'alpha': 0.01,
            'mode': "fixed",
            'direction': "backward",
            'train_ratio': 0.6,
            'trainDataAugment': False,
            'smoothFc': 1.5,
            'smoothOrder': 3,
        }
    else:
        raise ValueError(f"Invalid configIdx: {configIdx}")


def visualizePredictorConfig(simParams):
    print(f"{'='*50}")
    print(f"Predictor Configuration")
    print(f"{'='*50}")
    print(f"Window Length:          {simParams['LEN_window']}")
    print(f"Upsample K:             {simParams['upsampleK']}")
    print(f"Dataflow:               {simParams['dataflow']}")
    print(f"DB Parameter:           {simParams['dbParameter']}")
    print(f"Alpha:                  {simParams['alpha']}")
    print(f"Mode:                   {simParams['mode']}")
    print(f"Direction:              {simParams['direction']}")
    print(f"Train Ratio:            {simParams['train_ratio']}")
    print(f"Train Data Augment:     {simParams['trainDataAugment']}")
    print(f"Smooth Fc:              {simParams['smoothFc']}")
    print(f"Smooth Order:           {simParams['smoothOrder']}")
    print(f"{'='*50}")