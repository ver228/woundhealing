#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 16:38:57 2019

@author: avelinojaver
"""
from pathlib import Path
from localize_wound import img2woundcnt
import cv2
import matplotlib.pylab as plt
import numpy as np
from cell_localization.collect import save_data
from tqdm import tqdm

_debug = False
if __name__ == '__main__': 

    imgs_root_dir = Path('/Users/avelinojaver/OneDrive - Nexus365/heba/WoundHealing/raw')
    img_files = {x.stem : x  for x in imgs_root_dir.rglob('*.tif') if not x.name.startswith('.')}

    selected_dir = Path('/Users/avelinojaver/OneDrive - Nexus365/heba/WoundHealing/wound_contour_selectedv2')
    imgs2check = [(x.parent.name, x.stem[4:]) for x in selected_dir.rglob('*.tif') if not x.name.startswith('.')]
    
    save_name = Path('/Users/avelinojaver/OneDrive - Nexus365/heba/WoundHealing/wound_area_masksv2.hdf5')
    
    #%%
    src_files = []
    images = []
    masks = []
    contours = []
    
    imgs2check = sorted(imgs2check)
    for fname_id, (set_type, bn) in enumerate(tqdm(imgs2check)):
        fname = img_files[bn]
        
        img = cv2.imread(str(fname), -1)
        
        largest_cnt = img2woundcnt(img, set_type)
        cnt_rows = [(fname_id, 1, *x) for x in largest_cnt.tolist()]
        
        mask = np.zeros(img.shape, np.uint8)
        cv2.drawContours(mask, [largest_cnt], 0, 1, -1)
        
        src_files.append((fname_id, bn))
        images.append(img)
        masks.append(mask)
        contours += cnt_rows
        
        if _debug:
            figsize = (15, 5)
            plt.figure(figsize = figsize)
            plt.imshow(img)
            plt.plot(largest_cnt[:, 0], largest_cnt[:, 1], 'r')
            plt.axis('off')
            plt.title(bn)
        
    save_data(save_name, src_files, images, masks = masks, contours = contours)