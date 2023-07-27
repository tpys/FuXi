import argparse
import os
import time 
import numpy as np
import xarray as xr
import pandas as pd
import onnxruntime as ort

ort.set_default_logger_severity(3)


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, help="FuXi onnx model dir")
parser.add_argument('--input', type=str, required=True, help="The input data file, store in netcdf format")
parser.add_argument('--save_dir', type=str, default="")
parser.add_argument('--num_steps', type=int, nargs="+", default=[20, 20, 20])
args = parser.parse_args()


def time_encoding(init_time, total_step, freq=6):
    init_time = np.array([init_time])
    tembs = []
    for i in range(total_step):
        hours = np.array([pd.Timedelta(hours=t*freq) for t in [i-1, i, i+1]])
        times = init_time[:, None] + hours[None]
        times = [pd.Period(t, 'H') for t in times.reshape(-1)]
        times = [(p.day_of_year/366, p.hour/24) for p in times]
        temb = np.array(times, dtype=np.float32)
        temb = np.concatenate([np.sin(temb), np.cos(temb)], axis=-1)
        temb = temb.reshape(1, -1)
        tembs.append(temb)
    return np.stack(tembs)


def load_model(mo):
    sessions = []
    for name in ["short", "medium", "long"]:
        model_name = os.path.join(mo, f"{name}.onnx")
        if os.path.exists(model_name):
            start = time.perf_counter()
            print(f'Load model from {model_name} ...')
            session = ort.InferenceSession(model_name,  providers=['CUDAExecutionProvider'])
            load_time = time.perf_counter() - start
            print(f'Load model take {load_time:.2f} sec')
            sessions.append(session)
    return sessions


def load_data(data_file):
    input = xr.open_dataarray(data_file)
    return input


def save_like(output, data, step, save_dir="", freq=6, grid=0.25):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

        lead_time = (step+1) * freq
        init_time = pd.to_datetime(data.time.values[-1])

        lat = np.linspace(-90, 90, int(180/grid)+1, dtype=np.float32)
        lon = np.arange(0, 360, grid, dtype=np.float32)  
        fcst_time = init_time + pd.Timedelta(hours=lead_time)

        output = xr.DataArray(
            output, # 1 x 70 x 721 x 1440
            dims=['time', 'level', 'lat', 'lon'],
            coords=dict(
                time=[fcst_time],
                level=data.level,
                lat=lat,
                lon=lon,
            )
        )            
        save_name = os.path.join(save_dir, f'{lead_time:03d}.nc')
        output.to_netcdf(save_name)



def run_inference(sessions, data, num_steps, save_dir=""):
    total_step = sum(num_steps)
    init_time = pd.to_datetime(data.time.values[-1])

    tembs = time_encoding(init_time, total_step)
    input = data.values[None]

    print(f'input: {input.shape}, {input.min():.2f} ~ {input.max():.2f}')
    print(f'tembs: {tembs.shape}, {tembs.mean():.4f}')

    print('Inference ...')
    start = time.perf_counter()

    step = 0
    for i, session in enumerate(sessions):
        for _ in range(0, num_steps[i]):
            temb = tembs[step]
            new_input, = session.run(None, {'input': input, 'temb': temb})
            output = new_input[:, -1] 

            save_like(output, data, step, save_dir)

            print(f'stage: {i}, step: {step+1:02d}, output: {output.min():.2f} {output.max():.2f}')
            input = new_input
            step += 1

        if step > total_step:
            break

    run_time = time.perf_counter() - start
    print(f'Inference take {run_time:.2f} for {total_step} step')

    
if __name__ == "__main__":
    sessions = load_model(args.model)
    data = xr.open_dataarray(args.input)
    run_inference(sessions, data, args.num_steps, args.save_dir)

