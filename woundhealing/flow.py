#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 14:27:27 2019

@author: avelinojaver
"""

from cell_localization.flow import CoordFlow

import tables
import random
import numpy as np
import tqdm


class CoordFlowMerged(CoordFlow):
    def __init__(self, *args, **argkws):
        super().__init__(*args, **argkws)
        
        #I am duplicating the indexes so i have a good way to iterate over on the validation
        self.data_indexes = self.data_indexes + self.data_indexes
        
    
    def load_data(self, root_dir, padded_roi_size, is_preloaded = True):
        data = {} 
        fnames = [x for x in root_dir.rglob('*.hdf5') if not x.name.startswith('.')]
        
        header = 'Preloading Data' if is_preloaded else 'Reading Data'
        for fname in tqdm.tqdm(fnames, header):
            with tables.File(str(fname), 'r') as fid:
                img1 = fid.get_node('/img1')[:]
                img2 = fid.get_node('/img2')[:]
                rec = self._read_coords(fid)
                
                
                x2add = (img1[:], img2[:], rec)
                
            type_ids = set(np.unique(rec['type_id']).tolist())
            
            k = fname.parent.name
            for tid in type_ids:
                if tid not in data:
                    data[tid] = {}
                if k not in data[tid]:
                    data[tid][k] = []
                
                data[tid][k].append(x2add)
                
        return data
    
    def read_key(self, _type, _group, _img_id, is_full = False):
        input_ = self.data[_type][_group][_img_id]
        img1, img2, coords_rec = input_
        
        labels = np.array([self.types2label[x] for x in coords_rec['type_id']])
        target = dict(
                labels = labels,
                coordinates = np.array((coords_rec['cx'], coords_rec['cy'])).T
                )
        
        img1 = img1/img1.max()
        img2 = img2/img2.max()
            
        if not is_full:
            #I am randomly merging this image. This might bring a bit of problems 
            #on the validation, but for the moment this should work...
            
            if random.random() < 0.5:
                p = np.random.uniform(0., 1.)
                img = p*img1 + (1-p)*img2
            else:
                img = img1 if random.random() < 0.5 else img2 
            
            return img, target
        else:
            return img1, img2, target
    
    
    
    def read_full(self, ind):
        (_type, _group, _img_id) = self.data_indexes[ind]
        img1, img2, target = self.read_key(_type, _group, _img_id, is_full = True)
        
        #here i am expecting the indexes to be duplicated. I will assing the first half to one image while the second to the other
        N = len(self.data_indexes) // 2
        img = img1 if (ind // N) == 0 else img2
        
        img, target = self.transforms_full(img, target)
        
        return img, target
