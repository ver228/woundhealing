#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 11:12:10 2018

@author: avelinojaver
"""
import sys
from pathlib import Path

dname = Path(__file__).resolve().parents[2]
sys.path.append(str(dname))


from cell_localization.utils import get_device
from cell_localization.models import get_model
import cv2
import torch
import numpy as np
import pandas as pd
import tables
import tqdm


filters = tables.Filters(complevel=0, 
                          complib='blosc', 
                          shuffle=True, 
                          bitshuffle=True, 
                          fletcher32=True
                          )

if __name__ == '__main__':
    is_plot = False
    cuda_id = 0
    scale_int = (0, 4095)
    img_src_dir = Path.home() / 'workspace/localization/data/woundhealing/raw/'
    save_dir_root = Path.home() / 'workspace/localization/predictions/woundhealing'
    
    model_loc_dir = Path.home() / 'workspace/localization/results/locmax_detection/woundhealing-v2/woundhealing-v2-mix/different_losses'
    bn = 'woundhealing-v2-mix+Fwoundhealing+roi96_unet-simple_l2-G2.5_20190802_161150_adam_lr0.000128_wd0.0_batch128'
    model_path = model_loc_dir / bn / 'model_best.pth.tar'
    nms_threshold_abs = 0.0
    nms_threshold_rel = 0.05
    loss_type = 'l2-G2.5'
    model_type = 'unet-simple'  
    
    
#    model_loc_dir = Path.home() / 'workspace/localization/results/locmax_detection/woundhealing-v2/woundhealing-v2-mix/different_losses_complete/roi48/'
#    bn = 'woundhealing-v2-mix+Fwoundhealing+roi48_unet-simple_maxlikelihood_20190719_021908_adam_lr0.000256_wd0.0_batch256'
#    model_path = model_loc_dir / bn / 'model_best.pth.tar'
#    nms_threshold_abs = 0.0
#    nms_threshold_rel = 0.05
#    loss_type = 'maxlikelihood'
#    model_type = 'unet-simple'  
    
    
    assert model_path.exists()
    
    
    save_dir = save_dir_root / bn
    
      
    n_ch_in = 1
    n_ch_out = 1
    

    
    
    
    device = get_device(cuda_id)
    model = get_model(model_type, 
                      n_ch_in, 
                      n_ch_out, 
                      loss_type,
                         nms_threshold_abs = nms_threshold_abs,
                         nms_threshold_rel = nms_threshold_rel
                         )
    
    state = torch.load(model_path, map_location = 'cpu')
    model.load_state_dict(state['state_dict'])
    model.eval()
    model = model.to(device)
    
    #%%
    img_paths = img_src_dir.rglob('*.tif')
    img_paths = list(img_paths)
    for img_path in tqdm.tqdm(img_paths):
        img = cv2.imread(str(img_path), -1)
        
        x = img.astype(np.float32)
        x = (x - scale_int[0])/(scale_int[1] - scale_int[0])
        
        with torch.no_grad():
            X = torch.from_numpy(x[None, None])
            X = X.to(device)
            
            predictions = model(X)
            
            
            predictions = [{k:v.detach().cpu().numpy() for k,v in p.items()} for p in predictions]
            
            
        predictions = predictions[0]
        
        df = pd.DataFrame({'cx':predictions['coordinates'][:, 0], 
                            'cy':predictions['coordinates'][:, 1],
                            'scores_abs':predictions['scores_abs'],
                            'scores_rel':predictions['scores_rel']
                            })
        
    
        
        base_dir = str(img_path.parent).replace(str(img_src_dir), str(save_dir))
        save_name = Path(base_dir) / f'{img_path.name}_preds.csv'
        save_name.parent.mkdir(parents = True, exist_ok = True)
        
        
        df.to_csv(save_name, index=False)
        