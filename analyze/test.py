#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 15:31:48 2020

@author: avelinojaver
"""

from pathlib import Path
import pandas as pd
from tqdm import tqdm
import numpy as np
import pickle
import matplotlib.pylab as plt
import seaborn as sns


images_dir = '/Users/avelinojaver/OneDrive - Nexus365/heba/WoundHealing/QualityControl/artifacts/'
coords_dir = '/Users/avelinojaver/OneDrive - Nexus365/heba/WoundHealing/QualityControl/artifacts_coords/'

images_dir = Path(images_dir)
coords_dir = Path(coords_dir)

fnames = [x for x in images_dir.rglob('*.tif')]
#%%
coord_f = {}
for fname in coords_dir.glob('*.csv'):
    bn = fname.stem[:-len('__coords')].replace('_CONTROL', '')
    bn = bn.rpartition('_')[0]
    coord_f[bn] = fname
  #%%  

for fname in fnames:
    
    img = plt.imread(str(fname), -1)
    bn = fname.stem
    
    coord_fname = coord_f[bn]
    df = pd.read_csv(coord_fname)
    
    
    loc_intensities = img[df['x'], df['y']]
    
    med = np.median(loc_intensities)
    mad = np.median(np.abs(loc_intensities - med))
    
    int_limits =  (med - 6*mad), (med + 6*mad)
    
    invalid_mask = (img<int_limits[0]) | (img>int_limits[1])
    
    
    
    fig, axs = plt.subplots(2, 1, figsize = (30, 15), sharex = True, sharey = True)
    axs[0].imshow(img, cmap = 'gray')
    axs[0].plot(df['y'], df['x'], '.r')
    #axs[1].imshow(invalid_mask)
    axs[1].imshow(img > 4090)