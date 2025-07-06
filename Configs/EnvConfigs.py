

def getEnvConfig(configIdx):
    if configIdx == 0:
        return {
            'N_user': 4,
            'LEN_window': 200,
            'dataflow': 'thumb_fr',
            'r_bar': 5,
            'B': 100,
            'M_list': [3,4,5],
            'randomSeed': 999,
            'alpha_range': (0.01, 1.0),
            'discrete_alpha_steps': 10,
            'N_aggregation': 4,
        }
    elif configIdx == 1:
        return {
            'N_user': 4,
            'LEN_window': 200,
            'dataflow': 'thumb_bk',
            'r_bar': 5,
            'B': 100,
            'M_list': [3,4,5],
            'randomSeed': 999,
            'alpha_range': (0.01, 1.0),
            'discrete_alpha_steps': 10,
            'N_aggregation': 4,
        }
    elif configIdx == 2:
        return {
            'N_user': 3,
            'LEN_window': 200,
            'dataflow': 'thumb_bk',
            'r_bar': 5,
            'B': 100,
            'M_list': [3,4,5],
            'randomSeed': 999,
            'alpha_range': (0.01, 1.0),
            'discrete_alpha_steps': 2,
            'N_aggregation': 2,
        }
    else:
        raise ValueError(f"Invalid configIdx: {configIdx}")


def visualizeEnvConfig(simParams):
    print(f"{'='*50}")
    print(f"Environment Configuration")
    print(f"{'='*50}")
    print(f"Number of Users:        {simParams['N_user']}")
    print(f"Window Length:          {simParams['LEN_window']}")
    print(f"Dataflow:               {simParams['dataflow']}")
    print(f"N_aggregation:          {simParams['N_aggregation']}")
    print(f"Resource Bar:           {simParams['r_bar']}")
    print(f"Bandwidth:              {simParams['B']}")
    print(f"M List:                 {simParams['M_list']}")
    print(f"Random Seed:            {simParams['randomSeed']}")
    print(f"Alpha Range:            {simParams['alpha_range']}")
    print(f"Discrete Alpha Steps:   {simParams['discrete_alpha_steps']}")
    print(f"{'='*50}")