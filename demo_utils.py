import os
import dill
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from functools import reduce

def dill_dump(something, filename, make_dirs=False):
    # If make_folders is True, then any directories
    # in the filename that dont exist will be created
    # before `something` is saved to disk
    if make_dirs:
        dirs = filename.split('/')[:-1]
        for i in range(len(dirs)):
            save_dir = '/'.join(dirs[:i+1])
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
    with open(filename, "wb") as f:
        dill.dump(something, f)

def dill_load(filename):
    with open(filename, "rb") as f:
        return dill.load(f)


def get_mpf_rho(mpf_statevecs, mpf_coeffs):
    statevecs = [deepcopy(sv) for sv in mpf_statevecs]
    mpf_rho = reduce(lambda x,y : x+y, 
                     [sv.to_operator()*coeff 
                          for sv, coeff 
                          in zip(statevecs, mpf_coeffs)
                     ]
                    )
    return mpf_rho

def frob_distance(rho1, rho2):
    return np.linalg.norm(rho1 - rho2)

def get_pbar(N, label):
    """
    This method returns a progress bar.
    Provide N, the number of ticks the progress bar should have
    as well as a label for it.

    It's on you to call pbar.update(num_ticks) to make the progress bar
    track progress. And you must call pbar.close() when you're done.

    Returns: pbar a tqdm progress bar object.
    """
    return tqdm(    
        ncols=100,
        total=N,
        unit_scale=1,
        position=0,
        dynamic_ncols=True,
        bar_format="{l_bar}{bar:35}{r_bar}{bar:-10b}",
        desc=label,
    )