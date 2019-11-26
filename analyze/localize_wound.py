#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 11:01:08 2019

@author: avelinojaver
"""
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pylab as plt
from tqdm import tqdm

from pybasic.shading_correction import BaSiC


try:
    from skimage.filters import threshold_multiotsu
except ImportError:
    pass #this function is not available in scikit-image <0.17 I cannot install it in the server but i don't really need it
    

def _nuclei_seg_low(img):
    th = 0
    #img_g = cv2.GaussianBlur(img_p1, (11,11), 0)
    kernel_size = 7
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    img_g = cv2.dilate(img, kernel, iterations = 1)
    img_g = cv2.morphologyEx(img_g, cv2.MORPH_CLOSE, kernel, iterations = 3)
    
    
    bw = cv2.adaptiveThreshold(img_g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 251, th)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations = 3)
    
    return img_g, bw

def _nuclei_seg_hi(img):
    
    #img_g = cv2.GaussianBlur(img_p1, (11,11), 0)
    
    kernel_size = 5
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    img_g = cv2.medianBlur(img, kernel_size)
    img_g = cv2.dilate(img_g, kernel, iterations = 3)
    img_g = cv2.erode(img_g, kernel, iterations = 3)
    #img_g = cv2.morphologyEx(img_g, cv2.MORPH_CLOSE, kernel, iterations = 3)
    
    
    
    thresholds = threshold_multiotsu(img_g, classes=3)
    
    th = thresholds[0]
    bw = (img_g > th ).astype(np.uint8)*255
    
    return img_g, bw

#%%
def membrane_seg(img):
    th = 2
    kernel_size = 7
    
    img_g = img
    for kk in range(3):
        img_g = cv2.medianBlur(img_g, kernel_size)
        
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    bw = cv2.adaptiveThreshold(img_g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 251, th)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations = 1)
    
    return img_g, bw

def BaSiC_correction(img):
    mid = img.shape[1]//2
    dat = np.array([img[:, :mid], img[:, mid:]])
    
    optimizer = BaSiC('')
    optimizer.prepare(dat)
    optimizer.run()
    
    img_p1 = [optimizer.normalize(x) for x in dat]
    img_p1 = np.concatenate(img_p1, axis=1)
    return img_p1


def img2woundcnt(img_o, set_type, _debug = False):
    #%%
    bot, top = img_o.min(), img_o.max()
    img = (img_o - bot) / (top - bot)
    img = (img*255).astype(np.uint8)
    
    
    if set_type == 'nuclei':
        med = np.percentile(img, [50, 99])
        if med[1] - med[0] > 100:
            img_g, bw =  _nuclei_seg_hi(img)
        else:
            img_corr = BaSiC_correction(img)
            img_g, bw = _nuclei_seg_low(img_corr)
           
    elif set_type == 'membrane':
        img_corr = BaSiC_correction(img)
        img_g, bw = membrane_seg(img_corr)
    else:
        raise ValueError(f'Not implemented `{set_type}`.')
    
    contours, _ = cv2.findContours(~bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_cnt = max(contours, key = cv2.contourArea)
    largest_cnt = largest_cnt.squeeze(1)
    
    if _debug:
        fig, axs = plt.subplots(3, 1, figsize = (10, 10), sharex = True, sharey = True)
        axs[0].imshow(img)
        axs[0].plot(largest_cnt[:, 0], largest_cnt[:, 1], 'r')
        axs[1].imshow(img_g)
        axs[2].imshow(bw)
    #%%
    return largest_cnt

if __name__ == '__main__':
    
    save_dir = Path('/Users/avelinojaver/OneDrive - Nexus365/heba/WoundHealing/wound_contour_res/')
    save_dir.mkdir(exist_ok = True, parents = True)
    
    #root_dir = Path('/Users/avelinojaver/OneDrive - Nexus365/heba/WoundHealing/raw/membrane')
    root_dir = Path('/Users/avelinojaver/OneDrive - Nexus365/heba/WoundHealing/raw/nuclei')
    
    set_type = root_dir.name
    
    img_files = [x for x in root_dir.rglob('*.tif') if not x.name.startswith('.')]
    
    
    
    all_halfs = []
    
    #img_files = ['/Users/avelinojaver/OneDrive - Nexus365/heba/WoundHealing/raw/nuclei/110523SWLEC-11224C-098B_t24_Well_A12.tif']
    #img_files = ['/Users/avelinojaver/OneDrive - Nexus365/heba/WoundHealing/raw/mix/110905SWLEC-11149C-171B_t24_Well_F06.tif']
    #%%
    
    meds = []
    for fname in tqdm(img_files):
        img_o = cv2.imread(str(fname), -1)
        #%%
        largest_cnt = img2woundcnt(img_o, set_type, _debug = False)
        #%%
        figsize = (15, 5)
        plt.figure(figsize = figsize)
        plt.imshow(img_o)
        plt.plot(largest_cnt[:, 0], largest_cnt[:, 1], 'r')
        plt.axis('off')
        
        
        #plt.title(np.diff(np.percentile(img, [50, 99])))
        
        save_name = save_dir / f'RES_{fname.name}'
        plt.savefig(save_name, bbox_inches = 'tight')
        