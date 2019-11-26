#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 14:34:57 2019

@author: avelinojaver
"""
from pathlib import Path
import pandas as pd
import numpy as np
import cv2

import matplotlib.pylab as plt

def coords2img(coords, n_photons = None, pix_size = 0.2, im_size = None):
    '''
    Create a reconstruted images from a group of coordinates.

    if n_photons is None the function will return the blink counts
    if im_size is None the function will return an image of the minimum size to fit the binned counts
    '''
    
    if coords.size == 0:
        return np.zeros(im_size)
    
    coords_i = coords / pix_size
    coords_i = np.floor(coords_i).astype(np.int)

    #center the coords according to the blink limits
    corner = coords_i.min(axis = 0)
    coords_i -= corner[None, :]


    if im_size is None:
        im_size = coords_i.max(axis = 0) + 1 #here I got the maximum index, but i want the largest dimension so i add one
    else:
        #make sure the coordinates are within the limits
        _valid = (coords_i[:, 0] < im_size[0]) & (coords_i[:, 1] < im_size[1])
        coords_i = coords_i[_valid]

    #change coordiantes to indexes
    inds = coords_i[:, 0]*im_size[1] + coords_i[:, 1]

    counts = np.bincount(inds, weights = n_photons, minlength = im_size[0]*im_size[1])

    counts = counts.reshape(im_size)

    return counts

def get_wound_mask(coords, 
                   image_shape,
                   density_kernel_size,
                   opening_kernel_size,
                   empty_th,
                   _debug = False
                   ):
    cell_counts = coords2img(coords, pix_size = 1., im_size = image_shape)
    cell_density = cv2.blur(cell_counts.astype(np.float32), (density_kernel_size, density_kernel_size))
    _, wound_mask = cv2.threshold(cell_density, empty_th, 255, cv2.THRESH_BINARY_INV)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (opening_kernel_size,opening_kernel_size))
    wound_mask = cv2.morphologyEx(wound_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    
    if _debug:
        contours, _ = cv2.findContours(wound_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        fig, axs = plt.subplots(2,1, sharex = True, sharey = True)
        axs[0].imshow(img, cmap = 'gray')
        axs[0].set_title('Raw Image')
        for cnt in contours:
            cnt = cnt.squeeze(1)
            axs[0].plot(cnt[:, 0], cnt[:, 1], 'r')
        
        axs[1].imshow(cell_density)
        axs[1].set_title('Cell density Map')
        
        for ax in axs:
            ax.axis('off')
    
    return cell_density, wound_mask


def get_distance_from_wound(cell_density, wound_mask, dist_bin_size, bin_min_area = 1000, bins_length = None, _debug = False):
    #%%
    dist_from_wound = cv2.distanceTransform(~wound_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    
    valid = wound_mask==0
    dists = dist_from_wound[valid]
    vals = cell_density[valid]
    
    #data binned
    
    dists_digitized = (np.floor(dists/dist_bin_size)).astype(np.int)
    
    
    if bins_length is None:
        minlength = 0
    else:
        minlength = bins_length
        dists_digitized[dists_digitized >= bins_length] = bins_length - 1
        
        
    area_counts = np.bincount(dists_digitized, minlength = minlength)
    weighted_counts = np.bincount(dists_digitized, weights = vals, minlength = minlength)
    
    avg_density_from_wound = (weighted_counts/area_counts)
    avg_density_from_wound[area_counts < bin_min_area] = np.nan
    #%%
    if bins_length is not None:
         assert avg_density_from_wound.size == bins_length
             
    
    
    if _debug:
        plt.figure()
        plt.plot(dist_bins, avg_density_from_wound)
        plt.xlabel('Distance from the wound (pixels)')
        plt.ylabel('Average Cell Density (cells/pixels^2)')
    
    return avg_density_from_wound, area_counts

if __name__ == '__main__':
    from tqdm import tqdm
    
    coords_root_dir = '/Users/avelinojaver/OneDrive - Nexus365/heba/WoundHealing/predictions/woundhealing-v2-mix+Fwoundhealing+roi48_unet-simple_maxlikelihood_20190719_021908_adam_lr0.000256_wd0.0_batch256/'
    images_root_dir = '/Users/avelinojaver/OneDrive - Nexus365/heba/WoundHealing/raw/'
    
    coords_root_dir = Path(coords_root_dir)
    images_root_dir = Path(images_root_dir)
    

    density_kernel_size = 13
    opening_kernel_size = 35
    dist_bin_size = 10
    bin_min_area = 1000
    image_shape = (512, 1392)
    empty_th = 0#1/(density_kernel_size**2)


    #%%
    files2check = list(coords_root_dir.rglob('*.csv'))
    for fname in tqdm(files2check[:10]):
        img_name = str(fname).replace(str(coords_root_dir), str(images_root_dir))[:-len('_preds.csv')]
        img = cv2.imread(img_name, -1)
        
        df = pd.read_csv(fname)
        coords = df[['cy', 'cx']].values
        
        cell_density, wound_mask = get_wound_mask(coords, 
                                                   image_shape,
                                                   density_kernel_size,
                                                   opening_kernel_size,
                                                   empty_th,
                                                   _debug = True
                                                   )
        
        avg_density_from_wound, area_counts = get_distance_from_wound(cell_density, wound_mask, bin_min_area = bin_min_area, dist_bin_size = 10, bins_length = 50)
#        continue
#        #%%
#        from scipy.spatial.distance import pdist, squareform
#        from scipy.spatial import Voronoi, voronoi_plot_2d
#        
#        
#        dist_from_wound = cv2.distanceTransform(~wound_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
#        coords_dist_wound = dist_from_wound[coords[:, 0], coords[:, 1]]
#        dists_digitized = (np.floor(coords_dist_wound/dist_bin_size)).astype(np.int)
#        
#        dists = pdist(coords)
#        dists = squareform(dists)
#        np.fill_diagonal(dists, 1e10)
#        closest_cell_dists = np.min(dists, axis=0)
#        
#        median_closests_dist = np.median(closest_cell_dists)
#        neighbor_th = median_closests_dist*2
#        n_neighbors = np.sum(dists<neighbor_th, axis=1)
#        
#        
#        minlength = 0
#        counts = np.bincount(dists_digitized, minlength = minlength)
#        weighted_counts = np.bincount(dists_digitized, weights = closest_cell_dists, minlength = minlength)
#        
#        avg_closest = (weighted_counts/counts)
#        dist_bins = np.arange(counts.size)*dist_bin_size
#        
#        plt.figure()
#        plt.plot(avg_closest)
#        
#        
#        #%%
#        vor = Voronoi(coords)
#        #%%
#        voronoi_plot_2d(vor,show_vertices=False, point_size=2)
        #%%
        
        
        