from get_cell_density import get_wound_mask, get_distance_from_wound

from pathlib import Path
import pandas as pd
from tqdm import tqdm
import numpy as np
import pickle



def _process_file(fname,
    density_kernel_size = 13,
    opening_kernel_size = 35,
    dist_bin_size = 10,
    bin_min_area = 1000,
    image_shape = (512, 1392),
    empty_th = 0,
    bins_length = 50
    ):
    
    
    df = pd.read_csv(fname)
    coords = df[['cy', 'cx']].values
    
    cell_density, wound_mask = get_wound_mask(coords, 
                                                   image_shape,
                                                   density_kernel_size,
                                                   opening_kernel_size,
                                                   empty_th
                                                   )
    
    wound_area = (wound_mask>0).sum()
    if wound_area > 0:
        avg_density_from_wound, area_counts = get_distance_from_wound(cell_density, 
                                                            wound_mask, 
                                                            dist_bin_size = dist_bin_size, 
                                                            bin_min_area = bin_min_area,
                                                            bins_length = bins_length
                                                           )
        
    else:
        avg_density_from_wound = np.full(bins_length, np.nan)
    
    assert avg_density_from_wound.size == bins_length
    
    n_cells = coords.shape[0]
    return wound_area, n_cells, avg_density_from_wound

if __name__ == '__main__':
    bn = 'woundhealing-v2-mix+Fwoundhealing+roi48_unet-simple_maxlikelihood_20190719_021908_adam_lr0.000256_wd0.0_batch256'
    
    results_dir = Path.home() / 'workspace/localization/predictions/woundhealing' / bn
    save_name = results_dir / 'cell_density_data.p'
    
    
    
    coordinates_files = [x for x in (results_dir).rglob('*.csv') if not x.name.startswith('.')]
    
    #%%
    exp_groups = dict(
        control_mock = ['A01', 'A12', 'F01', 'F12', 'H01', 'H12'], #Control (mock treatment so here our baseline)
        control_none = ['B01', 'B12', 'G01', 'G12'], #Control (no treatment)
        CDC42 = ['D01', 'D12'], #CDC42 gene KD (this one affect migration): D01 and D12
        CDH5 = ['E01', 'E12'] #CDH5 gene KD (this one affect cell-cell contact): E01 and E12
    )
    wells_groups = {}
    for k, well_ids in exp_groups.items():
        for well_id in well_ids:
            wells_groups[well_id] = k
    
    
    dist_bin_size = 10
    image_shape = (512, 1392)
    image_area = image_shape[0]*image_shape[1]
    data = []
    wound_densities = []
    for fname in tqdm(coordinates_files):
        bn = fname.name[:-14].replace('-', '_')
        label1, label2, plate_id, time_point, _, well_id = bn.split('_')
        exp_id = int(plate_id[:-1])
        replicate_id = plate_id[-1]
        time_point =  int(time_point[1:])
        
        condition = wells_groups[well_id] if well_id in wells_groups else 'unknown'
        
        wound_area, n_cells, avg_density_from_wound =  _process_file(fname, dist_bin_size = dist_bin_size, image_shape = image_shape)
        wound_densities.append(avg_density_from_wound)
        global_cell_density = n_cells/(image_area - wound_area)
        
        
        
        row = (str(bn), label1, label2, plate_id, exp_id, replicate_id, well_id, time_point, condition, wound_area, n_cells, global_cell_density)
        data.append(row)

    df = pd.DataFrame(data, columns = ['basename', 'label1', 'label2', 'plate_id', 'exp_id', 'replicate_id', 'well_id', 'time_point', 'condition', 'wound_area', 'n_cells', 'global_cell_density'])
    wound_densities = np.array(wound_densities)
    
    dist_from_wound = np.arange(wound_densities.shape[1])*dist_bin_size
    #%%
    data2save = dict(
            data_info = df,
            cell_densities_from_wound = wound_densities,
            dist_from_wound = dist_from_wound
            )
    
    with open(save_name, 'wb') as fid:
        pickle.dump(data2save, fid)
    
    