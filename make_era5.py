import os
import numpy as np
import pandas as pd
import xarray as xr


def make_era5(init_time, data_dir):

    init_time = pd.to_datetime(init_time)
    print(f"process {init_time} ...")

    pl_file = os.path.join(data_dir, init_time.strftime('P%Y%m%d%H.nc'))
    pl = xr.open_dataset(pl_file)

    sfc_file = os.path.join(data_dir, init_time.strftime('S%Y%m%d%H.nc'))
    sfc = xr.open_dataset(sfc_file)

    tp_file = os.path.join(data_dir, init_time.strftime('R%Y%m%d.nc'))
    tp = xr.open_dataarray(tp_file).fillna(0)
    tp = tp.rolling(time=6).sum() * 1000
    tp = tp.sel(time=tp.time[::6])
    tp = tp.clip(min=0, max=1000)
    sfc['tp'] = tp

    pl_names = ['z', 't', 'u', 'v', 'r']
    sfc_names = ['t2m', 'u10', 'v10', 'msl', 'tp']
    levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

    channel = [f'{n.upper()}{l}' for n in pl_names for l in levels]
    channel +=[n.upper() for n in sfc_names]

    ds = []
    for name in pl_names + sfc_names:
        if name in ['z', 't', 'u', 'v', 'r']:
            v = pl[name]

        if name in ['t2m', 'u10', 'v10', 'msl', 'tp']:
            v = sfc[name]
            level = xr.DataArray([1], coords={'level': [1]}, dims=['level'])
            v = v.expand_dims({'level': level}, axis=1)             

        if np.isnan(v).sum() > 0:
            print(f"{name} has nan value")
            raise ValueError

        v.name = "data"
        v.attrs = {}                
        print(f"{name}: {v.shape}, {v.min().values} ~ {v.max().values}")
        ds.append(v)
     
    ds = xr.concat(ds, 'level')
    ds = ds.assign_coords(level=channel)
    ds = ds.rename({'longitude': 'lon', 'latitude': 'lat'})
    ds = ds.astype(np.float32)
    return ds
    
ds12 = make_era5('20230725-12', 'ERA520230725')
ds18 = make_era5('20230725-18', 'ERA520230725')
ds = xr.concat([ds12, ds18], 'time')
ds.to_netcdf('input.nc')

