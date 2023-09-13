import os
import numpy as np
import pandas as pd
import xarray as xr


"""
FuXi模型的输入变量:
    5个气压层变量: ['Z', 'T', 'U', 'V', 'R'], 
    每个变量包含13层: [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000], 
    5个地面变量: ['T2M', 'U10', 'V10', 'MSL', 'TP'];
    
注意事项:
    1. 输入是连续的两个历史时刻, 间隔6小时, 分辨率是0.25; eg: [00, 06]做为输入那么起报时刻是06点;
    2. Z不是Geopential Height, 是 Geopential;
    3. 降水是6小时累积单位mm, 第一个时刻的降水可以置0;
    4. 温度是开尔文单位;
    5. R表示相对湿度;
    6. 纬度方向是90 ~ -90;
    7. 气压层的顺序是从高空到地面50 ~ 1000; eg: Z50, Z100, ... , Z1000;
    8. 数据中不能有NAN;
"""

def make_hres_input(init_time, data_dir, save_dir, degree=0.25):
    lat = np.linspace(-90, 90, int(180 / degree) + 1, dtype=np.float32)
    lon = np.arange(0, 360, degree, dtype=np.float32)

    pl_names = ["z", "t", "u", "v", "r"]
    sfc_names = ["t2m", "u10", "v10", "msl", "tp"]
    levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

    input = []
    level = []

    for name in pl_names + sfc_names:
        src_name = "{}_{}".format(name, init_time.strftime("%Y%m%d%H.nc"))
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
                v = xr.concat([v.sel(level=l) for l in levels], "level")
                level.extend([f"{name}{l}" for l in levels])
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

        print(f"{src_name}: {v.min().values:.2f} ~ {v.max().values:.2f}")

        v.attrs = {}
        v = v.rename({"dtime": "time"})
        v = v.squeeze("member").drop("member")
        input.append(v)

    # concat and reshape
    input = xr.concat(input, "level")
    input = input.transpose("time", "level", "lat", "lon")
    valid_time = init_time + pd.Timedelta(hours=6)  # utc time
    v = v.assign_coords(time=[init_time, valid_time])

    # reverse latitude
    input = input.reindex(lat=input.lat[::-1])
    input = input.assign_coords(level=level)
    input.name = "data"

    # save to nc
    print(input)
    save_name = os.path.join(save_dir, init_time.strftime("%Y%m%d-%H.nc"))
    input = input.astype(np.float32)
    input.to_netcdf(save_name)


def test_make_input():
    init_time = pd.to_datetime("20230731-12")  # must utc
    data_dir = "data/HRES"
    save_dir = "data/HRES/input"
    os.makedirs(save_dir, exist_ok=True)
    make_hres_input(init_time, data_dir, save_dir)
