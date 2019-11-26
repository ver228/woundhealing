#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 10:29:14 2019

@author: avelinojaver
"""
from pathlib import Path

flow_types = {
        'woundhealing' : {
            'scale_int' : (0, 4095),
            'zoom_range' : (0.90, 1.1),
            'prob_unseeded_patch' : 0.2,
            'int_aug_offset' : (-0.2, 0.2),
            'int_aug_expansion' : (0.5, 1.3)
            },
        
        'woundhealing-merged' : {
                'scale_int' : (0, 1.),
                'zoom_range' : (0.90, 1.1),
                'prob_unseeded_patch' : 0.2,
                'int_aug_offset' : (-0.2, 0.2),
                'int_aug_expansion' : (0.5, 1.3)
            }
        
        }

data_types = {
        'woundhealing-v2-mix': {
        'root_data_dir' : Path.home() / 'workspace/localization/data/woundhealing/annotated/v2/mix',
        'log_prefix' : 'woundhealing-v2',
        'dflt_flow_type' : 'woundhealing',
         'n_ch_in'  : 1,
         'n_ch_out' : 1
        },
                
        'woundhealing-v2-nuclei': {
        'root_data_dir' : Path.home() / 'workspace/localization/data/woundhealing/annotated/v2/nuclei',
        'log_prefix' : 'woundhealing-v2',
        'dflt_flow_type' : 'woundhealing',
         'n_ch_in'  : 1,
         'n_ch_out' : 1
        },
        
        'woundhealing-F0.5-merged': {
        'root_data_dir' : Path.home() / 'workspace/localization/data/woundhealing/annotated/splitted/F0.5x/',
        'log_prefix' : 'woundhealing-F0.5-merged',
        'dflt_flow_type' : 'woundhealing-merged',
         'n_ch_in'  : 1,
         'n_ch_out' : 1
        }
    }