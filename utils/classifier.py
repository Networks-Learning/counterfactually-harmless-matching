from .common import reset_seed, LOCATIONS

import argparse
import random
import numpy as np
import pandas as pd

def get_biased_location_probs(
    location_probs,
    beta,
    seed=0,
):
    reset_seed(seed)
    
    location_num = location_probs.shape[1]
    overs = random.sample(range(location_num), location_num // 2)
    
    biased_location_probs = []
    for location in range(location_num):
        lp = location_probs[:, location]
        if location in overs:
            clp = lp * (1 + beta)
        else:
            clp = lp * (1 - beta)
        
        biased_location_probs.append(clp)

    biased_location_probs = np.stack(biased_location_probs, axis=1)
    biased_location_probs = np.clip(biased_location_probs, a_min=1e-6, a_max=1-1e-6)
    
    return biased_location_probs

def lp2df(
    lp,
    ref_id,
):
    score_df = pd.DataFrame(lp, columns=LOCATIONS)
    score_df['refugee_id'] = ref_id
    score_df = score_df[['refugee_id'] + LOCATIONS]
    
    return score_df