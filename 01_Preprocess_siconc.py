""" 
Wednesday 13th December 2023

Alistair Duffey

This script calculates the total sea ice area in each hemisphere from CMIP, 
GeoMIP and ARISE siconc outputs (aice for CESM ARISE), and saves the monthly
timeseries to csv files. The code is sensitive to directory structure so will
only run on JASMIN sci servers, and requires access to the jasmin cpom gws. 

The whole script takes ~ 20 mins to run on a jasmin sci server. 

We do not do any regridding, instead, we simply multiply be each model's ocean
grid areas file, then sum over x and y (after selecting for hemisphere). NB the
hemisphere selection is a bit of a hack, I manually checked the latitudinal direction
of the y-axis for the ocean grid. MPI is upside down, so i flip the selection here. If
expanding to more CMIP models, one would need to check that the correct hemisphere is 
being selected. An errors on this would be pretty obvious in the outputted annual cycles,
since the arctic and antarctic would be labelled the wrong way round. 

"""


import os
import glob
import pandas as pd
import numpy as np
import esmvalcore.preprocessor
import xarray as xr
from tqdm import tqdm
from xmip.preprocessing import rename_cmip6
import dask
from tqdm import tqdm


def read_in(dir):
    files = []
    for x in os.listdir(dir):
        files.append(dir + x)
    with dask.config.set(**{'array.slicing.split_large_chunks': True}):
        ds = rename_cmip6(xr.open_mfdataset(files, use_cftime=True, engine='netcdf4'))
    return ds

models = ['IPSL-CM6A-LR', 'UKESM1-0-LL', 'MPI-ESM1-2-HR', 'MPI-ESM1-2-LR', 'CESM2-WACCM']

reverse_y_mods = ['MPI-ESM1-2-HR', 'MPI-ESM1-2-LR'] # ocean grid is upside-down, which matters for selecting hemispheres on y
cmip_scens = ['piControl', 'historical', 'ssp245', 'ssp585']


### set up dict of grid cell areas for ocean grid of each model:
ocean_cell_areas = []
for mod in models:
    path = glob.glob('/badc/cmip6/data/CMIP6/CMIP/*/{}/piControl/r1i1p1f*/Ofx/areacello/gn/latest/'.format(mod))[0]
    full_path = path + os.listdir(path)[0]
    ocean_cell_areas.append(rename_cmip6(xr.open_mfdataset(full_path, use_cftime=True, engine='netcdf4')).areacello)
ocean_cell_areas_dict = dict(zip(models, ocean_cell_areas))

### define main function to calculate total hemipsheric sea ice area
def get_SI_areas_ds(dir, model):
    ds = read_in(dir)
    ocean_cell_areas = ocean_cell_areas_dict[model]
    ds['si_area'] = ds['siconc']*ocean_cell_areas
    n_ys = len(ds.y)
    if not model in reverse_y_mods:        
        ds_nh = ds.sel(y=slice(int(n_ys/2), 10000))
        ds_sh = ds.sel(y=slice(0, int(n_ys/2)))
    elif model in reverse_y_mods:  
        ds_nh = ds.sel(y=slice(0, int(n_ys/2)))
        ds_sh = ds.sel(y=slice(int(n_ys/2), 10000))

    out_ds_nh, out_ds_sh = ds_nh.si_area.sum(dim=('x', 'y')).to_dataset().rename({'si_area':'si_area_NH'}), ds_sh.si_area.sum(dim=('x', 'y')).to_dataset().rename({'si_area':'si_area_SH'})
    out_ds = xr.merge([out_ds_nh, out_ds_sh])
    return out_ds

################################################################

print('run for GeoMIP G6, should take a couple of mins')
GeoMIP_dirs = glob.glob('/gws/nopw/j04/cpom/aduffey/.synda/data/CMIP6/GeoMIP/*/*/G6*/*/SImon/siconc/*/*/')
print(GeoMIP_dirs)
## runs for 7 secs per iteration, a couple of mins for GeoMIP runs
DF = pd.DataFrame()
for dir in tqdm(GeoMIP_dirs):
    model, experiment, ensemble_member = dir.split('/')[11], dir.split('/')[12], dir.split('/')[13]
    ds = get_SI_areas_ds(dir=dir, model=model)
    df = pd.DataFrame({
                       'Year':ds.time.dt.year.values,
                       'Month':ds.time.dt.month.values,
                       'si_area_NH':ds.si_area_NH.values,
                       'si_area_SH':ds.si_area_SH.values,
                       })
    df['Model'] = model
    df['Experiment'] = experiment
    df['Ensemble_member'] = ensemble_member
    DF = pd.concat([DF, df])
DF.to_csv('processed_sea_ice/GeoMIP_G6_si_area.csv')

################################################################

### now repeat for the ssp scenarios:
print('now repeat for the cmip historical and ssp scenarios:')
cmip_dirs = []
for model in models:
    for scenario in cmip_scens:
        for x in glob.glob('/badc/cmip6/data/CMIP6/*/*/{m}/{s}/*/SImon/siconc/gn/latest/'.format(m=model, s=scenario)):
            cmip_dirs.append(x)
print(cmip_dirs)
## runs a bit slower cause runs are longer, 15 secs per it, 20 mins
DF = pd.DataFrame()
for dir in tqdm(cmip_dirs):
    model, experiment, ensemble_member = dir.split('/')[7], dir.split('/')[8], dir.split('/')[9]
    ds = get_SI_areas_ds(dir=dir, model=model)
    df = pd.DataFrame({
                       'Year':ds.time.dt.year.values,
                       'Month':ds.time.dt.month.values,
                       'si_area_NH':ds.si_area_NH.values,
                       'si_area_SH':ds.si_area_SH.values,
                       })
    df['Model'] = model
    df['Experiment'] = experiment
    df['Ensemble_member'] = ensemble_member
    DF = pd.concat([DF, df])
DF.to_csv('processed_sea_ice/CMIP_scens_si_area.csv')

################################################################

## now repeat for UKESM ARISE runs: 
print('Now repeat for UKESM ARISE runs')
UKESM_arise_dirs = glob.glob('/badc/deposited2022/arise/data/ARISE/MOHC/UKESM1-0-LL/arise-sai-1p5/*/SImon/siconc/*/*/')
print(UKESM_arise_dirs)

DF = pd.DataFrame()
for dir in tqdm(UKESM_arise_dirs):
    model, experiment, ensemble_member = dir.split('/')[7], dir.split('/')[8], dir.split('/')[9]
    ds = get_SI_areas_ds(dir=dir, model=model)
    df = pd.DataFrame({
                       'Year':ds.time.dt.year.values,
                       'Month':ds.time.dt.month.values,
                       'si_area_NH':ds.si_area_NH.values,
                       'si_area_SH':ds.si_area_SH.values,
                       })
    df['Model'] = model
    df['Experiment'] = experiment
    df['Ensemble_member'] = ensemble_member
    DF = pd.concat([DF, df])
DF.to_csv('processed_sea_ice/UKESM_arise_si_area.csv')

################################################################


## finally, repeat for CESM ARISE runs:

print('Now repeat for CESM2-WACCM ARISE runs')
# downloaded the ARISE CESM archive from NCAR Climate Data Gateway on Monday 11 December. 
def get_SI_areas_ds_cesm_arise(file, model):
    ds = rename_cmip6(xr.open_mfdataset(file, use_cftime=True, engine='netcdf4'))
    ds = ds.rename({'aice':'siconc'})
    ocean_cell_areas = ocean_cell_areas_dict[model]
    ds['si_area'] = ds['siconc']*ocean_cell_areas
    n_ys = len(ds.y)
    if not model in reverse_y_mods:        
        ds_nh = ds.sel(y=slice(int(n_ys/2), 10000))
        ds_sh = ds.sel(y=slice(0, int(n_ys/2)))
    elif model in reverse_y_mods:  
        ds_nh = ds.sel(y=slice(0, int(n_ys/2)))
        ds_sh = ds.sel(y=slice(int(n_ys/2), 10000))

    out_ds_nh, out_ds_sh = ds_nh.si_area.sum(dim=('x', 'y')).to_dataset().rename({'si_area':'si_area_NH'}), ds_sh.si_area.sum(dim=('x', 'y')).to_dataset().rename({'si_area':'si_area_SH'})
    out_ds = xr.merge([out_ds_nh, out_ds_sh])
    return out_ds
    
ens_mems = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010']
cesm_arise_si_path = '/gws/nopw/j04/cpom/aduffey/ARISE/CESM/arise/aice/'
CESM_ARISE_files = os.listdir(cesm_arise_si_path)
CESM_ARISE_files_full = [cesm_arise_si_path+x for x in CESM_ARISE_files]

DF = pd.DataFrame()
for file in CESM_ARISE_files_full:
    model = 'CESM2-WACCM'
    experiment='arise-sai-1p5'
    ensemble_member = file.split('.')[5]
    print(ensemble_member)
    ds = get_SI_areas_ds_cesm_arise(file=file, model=model)
    df = pd.DataFrame({
                       'Year':ds.time.dt.year.values,
                       'Month':ds.time.dt.month.values,
                       'si_area_NH':ds.si_area_NH.values,
                       'si_area_SH':ds.si_area_SH.values,
                       })
    df['Model'] = model
    df['Experiment'] = experiment
    df['Ensemble_member'] = ensemble_member
    DF = pd.concat([DF, df])
DF.to_csv('processed_sea_ice/CESM2-WACCM_arise_si_area.csv')
