import os

import numpy as np
import pandas as pd
import xarray as xr

__all__ = ["save_like"]

pl_names = ['z', 't', 'u', 'v', 'r']
sfc_names = ['t2m', 'u10', 'v10', 'msl', 'tp']
levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
degree = 0.25


def weighted_rmse(out, tgt):
    wlat = np.cos(np.deg2rad(tgt.lat))
    wlat /= wlat.mean()
    error = ((out - tgt) ** 2 * wlat)
    return np.sqrt(error.mean(('lat', 'lon')))


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


def save_like(output, input, step, save_dir="", input_type="hres", freq=6, split=False):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        step = (step+1) * freq
        init_time = pd.to_datetime(input.time.values[-1])

        if input_type.upper() == "HRES":
            step = (step+2) * freq
            init_time = pd.to_datetime(input.time.values[0])

        ds = xr.DataArray(
            output[None],
            dims=['time', 'step', 'level', 'lat', 'lon'],
            coords=dict(
                time=[init_time],
                step=[step],
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

        save_name = os.path.join(save_dir, f'{step:03d}.nc')
        # print(f'Save to {save_name} ...')
        ds.to_netcdf(save_name)


def make_era5_input(init_time, data_dir, save_dir):   
    ds = []
    init_time =  pd.to_datetime(init_time)
    hist_time = init_time - pd.Timedelta(hours=6)
    print(f"init_time: {init_time}")
    level = []

    for name in pl_names + sfc_names:
        data_name = os.path.join(data_dir, name, f'{init_time.year}')
        v = xr.open_zarr(data_name)
        v = v.sel(time=[hist_time, init_time])
        v = v.rename({name: 'data'})
        v.attrs = {}                
        ds.append(v)

        if name in pl_names:
            level.extend([f'{name.lower()}{l}' for l in levels])

        if name in sfc_names:
            level.append(name.lower())
        
    ds = xr.concat(ds, 'level')
    ds = ds.assign_coords(level=level)  

    os.makedirs(save_dir, exist_ok=True)
    save_name = os.path.join(save_dir, init_time.strftime("input.%Y%m%d.t%H.nc"))
    print(f"save to {save_name} ...")
    ds = ds.astype(np.float32)
    ds.to_netcdf(save_name)    


def make_era5(init_time, data_dir):
    import os
    import numpy as np
    import pandas as pd
    import xarray as xr
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
    
# ds12 = make_era5('20230725-12', 'ERA520230725')
# ds18 = make_era5('20230725-18', 'ERA520230725')
# ds = xr.concat([ds12, ds18], 'time')
# ds.to_netcdf('new_input.nc')


def make_hres_input(init_time, data_dir, save_dir):
    lat = np.linspace(-90, 90, int(180/degree)+1, dtype=np.float32)
    lon = np.arange(0, 360, degree, dtype=np.float32)

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


def make_gfs_input(init_time, data_dir, save_dir):
    pl_names = ['Z', 'T', 'U', 'V', 'R']
    sfc_names = ['t2m', 'u10', 'v10', 'msl', 'tp']

    lon = np.arange(0, 360, degree, dtype=np.float32)
    lat = np.arange(90, -90, -degree, dtype=np.float32)

    input = []
    level = []
    for name in pl_names + sfc_names:
        src_name = '{}_{}'.format(name, init_time.strftime("%Y%m%d.nc"))
        src_file = os.path.join(data_dir, src_name)

        if not os.path.exists(src_file):
            print(src_file)
            return

        try:
            v = xr.open_dataset(src_file)[name]
        except:
            print(f"open {src_file} failed")
            return

        if np.isnan(v).sum() > 0:
            print(f"{src_name} has nan value")
            return

        if v.shape[-2:] != (721, 1440):
            v = v.interp(lat=lat, lon=lon, kwargs={
                         "fill_value": "extrapolate"})
            if np.isnan(v).sum() > 0:
                print(f"{src_name} has nan value")
                return

        if name in pl_names:
            level.extend([f'{name.lower()}{l}' for l in levels])

        if name in sfc_names:
            level.append(name.lower())

        if name == "Z":
            v = v * 9.8

        if name == "tp":
            v = v.clip(min=0, max=1000)

        v = v.squeeze('step').drop('step')
        v.attrs = {}
        v.name = 'data'

        vmin = v.min().values
        vmax = v.max().values

        if vmax > 1e10:
            v = v.where(v < 1e10, 0)
            vmax = v.max().values

        assert vmax < 1e10
        print(f'{src_name}: {v.shape}, {vmin:.2f} ~ {vmax:.2f}')
        input.append(v)

    input = xr.concat(input, "level")  # T
    input = input.rename({"latitude": "lat", "longitude": "lon"})
    times = [pd.to_datetime(str(t), format='%Y%m%d%H')
             for t in input.time.values]
    input = input.assign_coords(level=level)
    input = input.assign_coords(time=times)

    # TODO, we only need two time step input with dims: 2 x 70 x 721 x 1440
    save_name = os.path.join(save_dir, init_time.strftime("%Y%m%d.nc"))
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
    make_hres_input(init_time, data_dir, save_dir)


def test_visualize(step, data_dir):
    src_name = os.path.join(data_dir, f"{step:03d}.nc")
    ds = xr.open_dataarray(src_name).isel(time=0)
    ds = ds.sel(lon=slice(90, 150), lat=slice(50, 0)) 
    print(ds)
    u850 = ds.sel(level='U850', step=step)
    v850 = ds.sel(level='V850', step=step)
    ws850 = np.sqrt(u850 ** 2 + v850 ** 2)
    visualize(f'ws850/{step:03d}.jpg', [ws850], [f'20230725-18+{step:03d}h'], vmin=0, vmax=30)


def test_rmse(output_name, target_name):
    output = xr.open_dataarray(output_name)
    output = output.isel(time=0).sel(step=120)
    target = xr.open_dataarray(target_name)

    for level in ["z500", "t850", "t2m", "u10", "v10", "msl", "tp"]:
        out = output.sel(level=level)
        tgt = target.sel(level=level)
        rmse = weighted_rmse(out, tgt).load()
        print(f"{level.upper()} 120h rmse: {rmse:.3f}")

