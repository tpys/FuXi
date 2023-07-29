import os

import numpy as np
import pandas as pd
import xarray as xr


def chunk_time(ds, shape=None):
    if shape is None:
        dims = {k:v for k, v in ds.dims.items()}
    else:
        dims = {k:v for k, v in zip(ds.dims, shape)}

    for k in ['time', 'lead_time']:
        if k in dims:
            dims[k] = 1

    ds = ds.chunk(dims)
    return ds


def make_input(init_time, data_dir, save_dir, deg=0.25):
    # These are fixed for FuXi
    pl_names = ['z', 't', 'u', 'v', 'r']
    sfc_names = ['t2m', 'u10', 'v10', 'msl', 'tp']
    levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
    lat = np.linspace(-90, 90, int(180/deg)+1, dtype=np.float32)
    lon = np.arange(0, 360, deg, dtype=np.float32)

    fcst_time = init_time + pd.Timedelta(hours=6) # utc time 
        
    input = []
    level = []
    for name in pl_names + sfc_names:
        src_name = '{}_{}'.format(name, init_time.strftime("%Y%m%d%H.nc"))
        src_file = os.path.join(data_dir, src_name)

        if not os.path.exists(src_file):
            return          

        try:
            v = xr.open_dataset(src_file).sel(time=init_time, drop=True).data
        except:
            print(f"open {src_file} failed")
            return 

        # is there nan in raw data ?
        if np.isnan(v).sum() > 0:
            print(f"{src_name} has nan value")
            return 

        # interpolate to 0.25 deg 
        v = v.interp(lat=lat, lon=lon, kwargs={"fill_value": "extrapolate"})

        # make sure on nan 
        if np.isnan(v).sum() > 0:
            print(f"{src_name} has nan value")
            return 


        # reverse pressure level 
        try:        
            if name in pl_names:
                v = xr.concat([v.sel(level=l) for l in levels], 'level')
                level.extend([f'{name}{l}' for l in levels])
        except:
            print("missing pressure level")
            return

        if name in sfc_names:
            level.append(name)

        # temperature in kelvin
        if name == "t":
            v = v + 273.15 
        
        # FuXi take two step as input
        if name == "tp":
            v = v.clip(min=0, max=1000)
            zero = v * 0
            zero = zero.assign_coords(dtime=[0])
            v = xr.concat([zero, v], "dtime")

        print(f'{src_name}: {v.min().values:.2f} ~ {v.max().values:.2f}')

        v.attrs = {}
        v = v.rename({'member': 'time', 'dtime': 'step'})
        v = v.assign_coords(time=[fcst_time]) #
        input.append(v)


    # concat and reshape 
    input = xr.concat(input, "level") 
    input = input.transpose("time", "step", "level", "lat", "lon")

    # reverse latitude 
    input = input.reindex(lat=input.lat[::-1])
    input = input.assign_coords(level=level)
    input.name = 'data'
    input = chunk_time(input, input.shape)

    # save to nc 
    save_name = os.path.join(save_dir, fcst_time.strftime("%Y%m%d-%H.nc"))
    input = input.astype(np.float32)
    input.to_netcdf(save_name)



if __name__ == "__main__":
    times = pd.date_range('20210101-00', '20211231-00', freq='12H')
    for init_time in times:
        make_input(init_time)





