## FuXi


This is the official repository for the FuXi paper.

[FuXi: A cascade machine learning forecasting system for 15-day global weather forecast
](https://arxiv.org/abs/2306.12873)

Published on npj Climate and Atmospheric Science: [FuXi: a cascade machine learning forecasting system for 15-day global weather forecast
](https://www.nature.com/articles/s41612-023-00512-1)

by Lei Chen, Xiaohui Zhong, Feng Zhang, Yuan Cheng, Yinghui Xu, Yuan Qi, Hao Li



## Installation

Both Zenodo (https://doi.org/10.5281/zenodo.10401602) and Baidu disk (https://pan.baidu.com/s/1PDeb-nwUprYtu9AKGnWnNw?pwd=fuxi) contain the FuXi model, and sample input and output data, all of which are essential resources for this study. For inquiries regarding having any kind of collaboration, please contact Professor Li Hao at the email address: lihao_lh@fudan.edu.cn.

The downloaded files shall be organized as the following hierarchy:

```plain
├──FuXi_EC
│   ├── short
│   ├── short.onnx
│   ├── medium
│   ├── medium.onnx
│   ├── long
│   ├── long.onnx

Sample_Data
│   ├── output --> FuXi model generated output data, only T+6 forecasts, T is the forecast initialization time
│   ├── 20231012-06_input_grib.nc --> FuXi model input data generated using the grib files of ECMWF HRES data
│   ├── 20231012-06_input_netcdf.nc --> FuXi model input data generated using the netcf files of ECMWF HRES data
│   ├── hres_input_grib_raw.zip --> the grib files of ECMWF HRES data
│   ├── hres_input_netcdf_raw.zip --> the netcdf files of ECMWF HRES data
│   ├── make_hres_input_public_version.py --> the script used to generate input data from either the grib or netcdf files of ECMWF HRES data

```

1. Install xarray 

```bash
conda install -c conda-forge xarray dask netCDF4 bottleneck
```

2. Install onnxruntime

```bash
pip install -r requirement.txt
```

## Demo

```bash 
python fuxi.py --model model_dir --input input_file --num_steps 20 20 20
```


## Data preparation 

The `input.nc` file contains preprocessed data from the origin ERA5 files. The file has a shape of (2, 70, 721, 1440), where the first dimension represents two time steps. The second dimension represents all variable and level combinations, named in the following exact order:

```plain
'Z50', 'Z100', 'Z150', 'Z200', 'Z250', 'Z300', 'Z400', 'Z500', 'Z600', 'Z700', 'Z850', 'Z925', 'Z1000', 
'T50', 'T100', 'T150', 'T200', 'T250', 'T300', 'T400', 'T500', 'T600', 'T700', 'T850', 'T925', 'T1000', 
'U50', 'U100', 'U150', 'U200', 'U250', 'U300', 'U400', 'U500', 'U600', 'U700', 'U850', 'U925', 'U1000', 
'V50', 'V100', 'V150', 'V200', 'V250', 'V300', 'V400', 'V500', 'V600', 'V700', 'V850', 'V925', 'V1000', 
'R50', 'R100', 'R150', 'R200', 'R250', 'R300', 'R400', 'R500', 'R600', 'R700', 'R850', 'R925', 'R1000', 
'T2M', 'U10', 'V10', 'MSL', 'TP'
```

The last five variables ('T2M', 'U10', 'V10', 'MSL', 'TP') are surface variables, while the remaining variables represent atmosphere variables with numbers representing pressure levels.


**_NOTE:_**

- The variable 'Z' represents geopotential and not geopotential height.
- The variable 'TP' represents total precipitation accumulated over a period of 6 hours.


