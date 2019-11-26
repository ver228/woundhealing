#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 01:30:42 2019

@author: avelinojaver
"""
from cell_localization.models import get_mapping_network, model_types
from cell_localization.utils import get_device
from pathlib import Path
import torch
import cv2
import numpy as np
import matplotlib.pylab as plt
from tqdm import tqdm

if __name__ == '__main__':
    cuda_id = 0
    device = get_device(cuda_id)
    
    save_dir = Path.home() / 'workspace/segmentation/predictions/woundhealing-contour/'
    root_dir = Path.home() / 'workspace/localization/data/woundhealing/raw'
    
    
    #bn = 'woundhealing-contourfold-1-5+Fwoundhealing-contour+roi256_unet-simple-bn_BCE_20191107_173604_adam_lr0.00064_wd0.0_batch16'
    bn = 'woundhealing-contourfold-1-5+Fwoundhealing-contour+roi256_unet-simple-bn_BCE_20191107_202508_adam_lr0.00064_wd0.0_batch16'
    model_path = Path.home() / 'workspace/segmentation/results/woundhealing-contour/woundhealing-contour' / bn / 'model_best.pth.tar'
    
    
    n_ch_in, n_ch_out = 1, 1
    model_name = 'unet-simple-bn'
    
    model = get_mapping_network(n_ch_in, n_ch_out, **model_types[model_name], output_activation = 'sigmoid')
    state = torch.load(model_path, map_location = 'cpu')
    model.load_state_dict(state['state_dict'])
    model.eval()
    model = model.to(device)
    
    #%%
    fnames = [x for x in root_dir.rglob('*.tif') if not x.name.startswith('.')]
    
    for fname in tqdm(fnames):
    
        img = cv2.imread(str(fname), -1)
        
        img_norm = img.astype(np.float32)/4095
        with torch.no_grad():
            X = torch.from_numpy(img_norm[None, None])
            X = X.to(device)
            Xout = model(X)
            xout = Xout[0,0].detach().cpu().numpy()
            
        #%%
        th = 0.5
        mask = (xout > th).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_cnt = max(contours, key = cv2.contourArea)
            largest_cnt = largest_cnt.squeeze(1)
        else:
            largest_cnt = None
        
        
        
        fig, axs = plt.subplots(2, 1, sharex = True, sharey = True)
        axs[0].imshow(img)
        
        if largest_cnt is not None:
            axs[0].plot(largest_cnt[:, 0], largest_cnt[:, 1], 'r')
        axs[1].imshow(xout, vmin = 0., vmax = 1.)
        for ax in axs:
            ax.axis('off')
        
        save_name = save_dir / 'contour_plots' / (fname.stem + '.jpg')
        save_name.parent.mkdir(exist_ok = True, parents = True)
        plt.savefig(save_name, bbox_inches = 'tight')
        plt.close()
        
        
        
        
        