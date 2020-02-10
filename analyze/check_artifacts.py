#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 15:00:35 2020

@author: avelinojaver
"""

from pathlib import Path


if __name__ == '__main__':
    artifacts_dir = Path('/Users/avelinojaver/OneDrive - Nexus365/heba/WoundHealing/QualityControl/artifacts/')
    predictions_dir = Path('/Users/avelinojaver/OneDrive - Nexus365/heba/WoundHealing/predictions/woundhealing-v2-mix+Fwoundhealing+roi48_unet-simple_maxlikelihood_20190719_021908_adam_lr0.000256_wd0.0_batch256/membrane/')
    
    artifacts_files = list(artifacts_dir.rglob('*.tif'))
    predictions_files = list(predictions_dir.rglob('*.csv'))
    predictions_files = {x.name[:-len('.tif_preds.csv')] : x for x in predictions_files}
    
    
    for fname in artifacts_files:
        if not fname.stem in predictions_files:
            print(fname.stem)
            continue
        pred_file = predictions_files[fname.stem]
        
    