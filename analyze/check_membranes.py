#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 14:37:00 2019

@author: avelinojaver
"""

import cv2
import matplotlib.pylab as plt



if __name__ == '__main__':
    fname = '/Users/avelinojaver/OneDrive - Nexus365/heba/WoundHealing/raw/membrane/111031SWLEC-11212C-258A_t24_Well_H01.tif'
    
    img = cv2.imread(fname, -1)
    
    
    plt.figure()
    plt.imshow(img)


