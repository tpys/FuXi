import os

import numpy as np
import pandas as pd
import xarray as xr

__all__ = ['make_input', "save_like"]

pl_names = ['z', 't', 'u', 'v', 'r']
sfc_names = ['t2m', 'u10', 'v10', 'msl', 'tp']
levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]


def split_variable(ds, name):
    if name in sfc_names:
        v = ds.sel(level=[name])
        v = v.assign_coords(level=[0])
        v = v.rename({"level": "level0"})
        v = v.transpose('member', 'level0', 'time', 'dtime', 'lat', 'lon')        
    elif name in pl_names:
        level = [f'{name}{l}' for l in levels]
        v = ds.sel(level=level)
        v = v.assign_coords(level=levels)
        v = v.transpose('member', 'level', 'time', 'dtime', 'lat', 'lon')
    return v


def save_like(output, input, step, save_dir="", freq=6, split=False):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        dtime = (step+2) * freq # 
        init_time = pd.to_datetime(input.time.values[0])

        ds = xr.DataArray(
            output[None, None],
            dims=['member', 'time', 'dtime', 'level', 'lat', 'lon'],
            coords=dict(
                member=['FuXi'],
                time=[init_time],
                dtime=[dtime],
                level=input.level,
                lat=input.lat.values,
                lon=input.lon.values,
            )
        ).astype(np.float32)  
        
        if split:
            def rename(name):
                if name == "tp":
                    return "TP06"
                elif name == "r":
                    return "RH"
                return name.upper()
            
            new_ds = []
            for k in pl_names + sfc_names:
                v = split_variable(ds, k)
                v.name = rename(k)
                new_ds.append(v)
            ds = xr.merge(new_ds, compat="no_conflicts")

        print(f'Save to {save_name} ...')
        save_name = os.path.join(save_dir, f'{dtime:03d}.nc')
        ds.to_netcdf(save_name)        


def make_input(init_time, data_dir, save_dir, deg=0.25):
    lat = np.linspace(-90, 90, int(180/deg)+1, dtype=np.float32)
    lon = np.arange(0, 360, deg, dtype=np.float32)

    input = []
    level = []
    
    for name in pl_names + sfc_names:
        src_name = '{}_{}'.format(name, init_time.strftime("%Y%m%d%H.nc"))
        src_file = os.path.join(data_dir, src_name)

        if not os.path.exists(src_file):
            return

        try:
            v = xr.open_dataset(src_file)
            v = v.sel(time=init_time, drop=True).data
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
        v = v.rename({'dtime': 'time'})
        v = v.squeeze('member').drop('member')
        input.append(v)

    # concat and reshape
    input = xr.concat(input, "level")
    input = input.transpose("time", "level", "lat", "lon")
    valid_time = init_time + pd.Timedelta(hours=6)  # utc time
    v = v.assign_coords(time=[init_time, valid_time])

    # reverse latitude
    input = input.reindex(lat=input.lat[::-1])
    input = input.assign_coords(level=level)
    input.name = 'data'

    # save to nc
    print(input)
    save_name = os.path.join(save_dir, init_time.strftime("%Y%m%d-%H.nc"))
    input = input.astype(np.float32)
    input.to_netcdf(save_name)


def visualize(save_name, vars=[], titles=[], vmin=None, vmax=None):
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(len(vars), 1, figsize=(8, 6), subplot_kw={
                           "projection": ccrs.PlateCarree()})

    def plot(ax, v, title):
        v.plot(
            ax=ax,
            x='lon',
            y='lat',
            vmin=vmin,
            vmax=vmax,
            transform=ccrs.PlateCarree(),
            add_colorbar=False
        )
        # ax.coastlines()
        ax.set_title(title)
        gl = ax.gridlines(draw_labels=True, linewidth=0.5)
        gl.top_labels = False
        gl.right_labels = False

    for i, v in enumerate(vars):
        if len(vars) == 1:
            plot(ax, v, titles[i])
        else:
            plot(ax[i], v, titles[i])

    plt.savefig(save_name, bbox_inches='tight',
                pad_inches=0.1, transparent='true', dpi=200)
    plt.close()


def test_make_input():
    init_time = pd.to_datetime("20230731-12")  # must utc
    data_dir = "data/HRES"
    save_dir = "data/HRES/input"
    os.makedirs(save_dir, exist_ok=True)
    make_input(init_time, data_dir, save_dir)


def test_visualize():
    ds = xr.open_dataarray('data/HRES/output/072.nc')
    tp = ds.sel(level='tp')
    visualize('tp.jpg', [tp], ['tp'], vmin=0, vmax=20)


# test_make_input()