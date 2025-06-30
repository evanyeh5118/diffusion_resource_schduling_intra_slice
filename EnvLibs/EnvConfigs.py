

def getEnvConfig(configIdx):
    if configIdx == 0:
        return {
            'N_user': 4,
            'LEN_window': 20,
            'r_bar': 4,
            'B': 40,
            'M_list': [2,3],
            'randomSeed': 999,
            'alpha_range': (0.01, 1.0),
            'discrete_alpha_steps': 10,
            'N_aggregation': 4,
            'dataflow': "thumb_fr"
        }
    elif configIdx == 1:
        return {
            'N_user': 4,
            'LEN_window': 20,
            'r_bar': 4,
            'B': 40,
            'M_list': [2,3],
            'randomSeed': 999,
            'alpha_range': (0.01, 1.0),
            'discrete_alpha_steps': 10,
            'N_aggregation': 4,
            'dataflow': "thumb_bk"
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
    print(f"Resource Bar:           {simParams['r_bar']}")
    print(f"Bandwidth:              {simParams['B']}")
    print(f"M List:                 {simParams['M_list']}")
    print(f"Random Seed:            {simParams['randomSeed']}")
    print(f"Alpha Range:            {simParams['alpha_range']}")
    print(f"Discrete Alpha Steps:   {simParams['discrete_alpha_steps']}")
    print(f"{'='*50}")