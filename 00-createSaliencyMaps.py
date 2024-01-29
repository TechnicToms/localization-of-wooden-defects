import torch
import torchvision

import os
import tqdm
import numpy as np

from helpers.terminalColor import terminalColor as tc
from metrics.saliency import salience_metric


DATA_ROOT: str = "/home/tsa/data/NaturalOak/train/good" 


def get_files(path: str) -> list:
    """Returns a list of all images, that are inside of the folder

    Args:
        path (str): path to files

    Returns:
        list: files
    """
    AllFiles = []
    for file in os.listdir(path):
        if file.endswith(('.png', '.jpg')):
            AllFiles.append(file)
            
    return AllFiles

def generatePrior(files: list):
    """Generates the priors and stores them in a separate directory ``priors``.

    Args:
        files (list): List of files to process
    """
    # Create Prior Generator Class
    sm = salience_metric(K=64)
    
    new_sm_root = os.path.join(DATA_ROOT, "prior/")
    if not os.path.exists(new_sm_root):
        # Create directory if non exists 
        os.makedirs(new_sm_root)
    else:
        # Else: Check, if there are already finished priors
        #       Remove those from the computing list, that already exists
        alreadyFinishedPriors = get_files(new_sm_root)
        for alFiles in alreadyFinishedPriors:
            files.remove(alFiles)
    
    for file in tqdm.tqdm(files, desc=tc.info + "Processing saliency maps"):
        # Load image 
        filepath = os.path.join(DATA_ROOT, file)
        img = torchvision.io.read_image(filepath, mode=torchvision.io.ImageReadMode.RGB) / 255.0
        
        # Convert dtype and compute prior
        img_numpy = np.array(img.permute((1, 2, 0)), dtype=np.float32)
        prior_numpy = sm.calculate_defectmap_parallel(img_numpy)
        prior_torch = torch.from_numpy(prior_numpy)
        
        # Save image in prior directory
        save_path = os.path.join(new_sm_root, file)
        torchvision.io.write_png((prior_torch[None, ...]*255.0).type(torch.uint8), save_path)
   

if __name__ == '__main__':
    files = get_files(DATA_ROOT)
    generatePrior(files=files)        
    print(tc.success + 'finished!')