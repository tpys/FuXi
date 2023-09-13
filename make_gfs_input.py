import os
import numpy as np
import pandas as pd
import pygrib as pg
import xarray as xr

def make_gfs(src_name):
    assert os.path.exists(src_name)

    levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
    pl_names = ['gh', 't', 'u', 'v', 'r']
    sf_names = ['2t', '10u', '10v', 'mslet']
        
    try:
        ds = pg.open(src_name)
    except:
        print(f"{src_name} not found")
        return

    input = []
    level = []

    for name in pl_names + sf_names + ["tp"]:
        
        if name in pl_names:
            try:
                data = ds.select(shortName=name, level=levels)
            except:
                print("pl wrong")
                return

            data = data[:len(levels)]

            if len(data) != len(levels):
                print("pl wrong")
                return 

            if name == "gh":
                name = "z"
            
            for v in data:
                init_time = f'{v.date}-{v.time//100:02d}'

                lat = v.distinctLatitudes
                lon = v.distinctLongitudes    

                img, _, _ = v.data()

                if name == "z":
                    img = img * 9.8

                input.append(img)
                level.append(f'{name}{v.level}')
                print(f"{v.name}: {v.level}, {img.shape}, {img.min()} ~ {img.max()}")

        if name in sf_names:
            try:
                data = ds.select(shortName=name)
            except:
                print('sfc wrong')
                return

            name_map = {'2t': 't2m', '10u': 'u10', '10v': 'v10', 'mslet': 'msl'}
            name = name_map[name]

            for v in data:
                img, _, _ = v.data()
                input.append(img)
                level.append(name)
                print(f"{v.name}: {img.shape}, {img.min()} ~ {img.max()}")

        if name == "tp":
            tp = img * 0
            input.append(tp)
            level.append("tp")

    input = np.stack(input)
    assert input.shape[-3:] == (70, 721, 1440)
    assert input.max() < 1e10

    times = [pd.to_datetime(init_time)]
    input = xr.DataArray(
        data=input[None],
        dims=['time', 'level', 'lat', 'lon'],
        coords={'time': times, 'level': level, 'lat': lat, 'lon': lon},
    )

    if np.isnan(input).sum() > 0:
        print("Field has nan value")
        return 
    
    return input



def test_make_gfs():
    d1 = make_gfs('30/gfs.t06z.pgrb2.0p25.f000')
    d2 = make_gfs('30/gfs.t12z.pgrb2.0p25.f000')

    if d1 and d2:
        ds = xr.concat([d1, d2], 'time')
        ds = ds.assign_coords(time=ds.time.astype(np.datetime64))
        ds.to_netcdf('input.nc')