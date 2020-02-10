from get_cell_density import get_wound_mask, get_distance_from_wound

from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import tqdm


def _process_file(fname,
    density_kernel_size = 13,
    opening_kernel_size = 35,
    dist_bin_size = 20, #10, #25, 
    bins_length = 25, #50,#20,
    bin_min_area_frac = 0.0025, #0.005,#0.1,
    image_shape = (512, 1392),
    empty_th = 0,
    
    coord_x_str = 'x',
    coord_y_str = 'y'
    ):
    #%%
    
    df = pd.read_csv(fname)
    coords = df[[coord_x_str, coord_y_str]].values
    
    
    assert (coords[:, 0] < image_shape[0]).all() & (coords[:, 1] < image_shape[1]).all()
    
    
    cell_density, wound_mask = get_wound_mask(coords, 
                                                   image_shape,
                                                   density_kernel_size,
                                                   opening_kernel_size,
                                                   empty_th
                                                   )
    #%%
    wound_area = (wound_mask>0).sum()
    if wound_area > 0:
        avg_density_from_wound, area_counts = get_distance_from_wound(cell_density, 
                                                            wound_mask, 
                                                            dist_bin_size = dist_bin_size, 
                                                            bin_min_area_frac = bin_min_area_frac,
                                                            bins_length = bins_length
                                                           )
        
    else:
        avg_density_from_wound = np.full(bins_length, np.nan)
    
    assert avg_density_from_wound.size == bins_length
    
    n_cells = coords.shape[0]
    return wound_area, n_cells, avg_density_from_wound

def main_bkp():
    bn = 'woundhealing-v2-mix+Fwoundhealing+roi48_unet-simple_maxlikelihood_20190719_021908_adam_lr0.000256_wd0.0_batch256'
    #results_dir = Path.home() / 'workspace/localization/predictions/woundhealing' / bn
    results_dir =  Path.home() / 'OneDrive - Nexus365/heba/WoundHealing/predictions/' / bn
    
    save_name = results_dir / 'cell_density_data.p'
    
    coordinates_files = [x for x in (results_dir).rglob('*.csv') if not x.name.startswith('.')]
    
    
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
    
    
    dist_bin_size = 25
    image_shape = (512, 1392)
    image_area = image_shape[0]*image_shape[1]
    data = []
    wound_densities = []
    for fname in tqdm.tqdm(coordinates_files):
        
        bn = fname.name[:-14]
        
        bn_s = bn.replace('-', '_')
        label1, label2, plate_id, time_point, _, well_id = bn_s.split('_')
        exp_id = int(plate_id[:-1])
        replicate_id = plate_id[-1]
        time_point =  int(time_point[1:])
        
        if well_id in wells_groups:
            condition = wells_groups[well_id]
        else:
            condition = 'unknown'
        
        wound_area, n_cells, avg_density_from_wound =  _process_file(fname, 
                                                                     dist_bin_size = dist_bin_size, 
                                                                     image_shape = image_shape,
                                                                     coord_x_str = 'cy',
                                                                     coord_y_str = 'cx')
        wound_densities.append(avg_density_from_wound)
        global_cell_density = n_cells/(image_area - wound_area)
        
        
        
        row = (str(bn), label1, label2, plate_id, exp_id, replicate_id, well_id, time_point, condition, wound_area, n_cells, global_cell_density)
        data.append(row)

    df = pd.DataFrame(data, columns = ['basename', 'label1', 'label2', 'plate_id', 'exp_id', 'replicate_id', 'well_id', 'time_point', 'condition', 'wound_area', 'n_cells', 'global_cell_density'])
    wound_densities = np.array(wound_densities)
    
    dist_from_wound = np.arange(wound_densities.shape[1])*dist_bin_size
    
    data2save = dict(
            data_info = df,
            cell_densities_from_wound = wound_densities,
            dist_from_wound = dist_from_wound
            )
    
    with open(save_name, 'wb') as fid:
        pickle.dump(data2save, fid)
    
def file2row(fname, 
             dist_bin_size = 10,
             image_shape = (512, 1392),
             coord_postfix = '__coords.csv'
    
             ):
    bn = fname.name[:-len(coord_postfix)]
        
    bn_s = bn.replace('-', '_')
    
    
    label1, label2, plate_id, time_point, _, well_id, *condition = bn_s.split('_')
    exp_id = int(plate_id[:-1])
    replicate_id = plate_id[-1]
    time_point =  int(time_point[1:])
    condition = '-'.join(condition)
    
    wound_area, n_cells, avg_density_from_wound =  _process_file(fname, dist_bin_size = dist_bin_size, image_shape = image_shape)
    
    
    image_area = image_shape[0]*image_shape[1]
    global_cell_density = n_cells/(image_area - wound_area)
    
    row = (str(bn), label1, label2, plate_id, exp_id, replicate_id, well_id, time_point, condition, wound_area, n_cells, global_cell_density)
    return row, avg_density_from_wound

if __name__ == '__main__':
    
    results_dir = Path('/Users/avelinojaver/Nexus365/Heba Sailem - HitDataJan2020/Coord/')
    
    save_name = results_dir.parent / 'cell_density_data.p'
    
    dist_bin_size = 25
    image_shape = (512, 1392)
    coord_postfix = '__coords.csv'
    
    coord_postfix = '__coords.csv'
    
    
    #coordinates_files = [x for x in (results_dir).rglob('*.csv') if not x.name.startswith('.')]
    
    files_f = {}
    for fname in results_dir.rglob('*.csv'):
        bn = fname.name[:-len(coord_postfix)]
        bn_s = bn.replace('-', '_')
        label1, label2, plate_id, time_point, _, well_id, *condition = bn_s.split('_')
        key = (plate_id, well_id)
        if not key in files_f:
            files_f[key] = []
        files_f[key].append(fname)
    
    files_f = {k : x for k, x in files_f.items() if len(x) == 2}
    
    
    data = []
    for key, fnames in tqdm.tqdm(list(files_f.items())):
        for fname in fnames:
            row = file2row(fname,
                           dist_bin_size = dist_bin_size,
                           image_shape = image_shape,
                           coord_postfix = coord_postfix
                           )
            data.append(row)
    rows, wound_densities = zip(*data)
    
    
    df = pd.DataFrame(rows, columns = ['basename', 'label1', 'label2', 'plate_id', 'exp_id', 'replicate_id', 'well_id', 'time_point', 'condition', 'wound_area', 'n_cells', 'global_cell_density'])
    wound_densities = np.array(wound_densities)
    
    dist_from_wound = np.arange(wound_densities.shape[1])*dist_bin_size
    
    data2save = dict(
            data_info = df,
            cell_densities_from_wound = wound_densities,
            dist_from_wound = dist_from_wound
            )
    
    with open(save_name, 'wb') as fid:
        pickle.dump(data2save, fid)
        
        