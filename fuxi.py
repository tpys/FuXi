import argparse
import os
import time 
import numpy as np
import xarray as xr
import pandas as pd
import onnxruntime as ort

from util import save_like

ort.set_default_logger_severity(3)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, help="FuXi onnx model dir")
parser.add_argument('--input', type=str, required=True, help="The input data file, store in netcdf format")
parser.add_argument('--save_dir', type=str, default="")
parser.add_argument('--num_steps', type=int, nargs="+", default=[20])
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



def load_model(model_name):
    # Set the behavier of onnxruntime
    options = ort.SessionOptions()
    options.enable_cpu_mem_arena=False
    options.enable_mem_pattern = False
    options.enable_mem_reuse = False
    # Increase the number for faster inference and more memory consumption
    options.intra_op_num_threads = 1
    cuda_provider_options = {'arena_extend_strategy':'kSameAsRequested',}

    session = ort.InferenceSession(
        model_name,  
        sess_options=options, 
        providers=[('CUDAExecutionProvider', cuda_provider_options)]
    )
    return session


def run_inference(model_dir, data, num_steps, save_dir=""):

    total_step = sum(num_steps)
    init_time = pd.to_datetime(data.time.values[-1])
    tembs = time_encoding(init_time, total_step)

    print(f'init_time: {init_time.strftime(("%Y%m%d-%H"))}')
    print(f'latitude: {data.lat.values[0]} ~ {data.lat.values[-1]}')
    
    assert data.lat.values[0] == 90
    assert data.lat.values[-1] == -90

    input = data.values[None]
    print(f'input: {input.shape}, {input.min():.2f} ~ {input.max():.2f}')
    print(f'tembs: {tembs.shape}, {tembs.mean():.4f}')

    stages = ['short', 'medium', 'long']
    step = 0

    for i, num_step in enumerate(num_steps):
        stage = stages[i]
        start = time.perf_counter()
        model_name = os.path.join(model_dir, f"{stage}.onnx")
        print(f'Load model from {model_name} ...')        
        session = load_model(model_name)
        load_time = time.perf_counter() - start
        print(f'Load model take {load_time:.2f} sec')

        print(f'Inference {stage} ...')
        start = time.perf_counter()

        for _ in range(0, num_step):
            temb = tembs[step]
            new_input, = session.run(None, {'input': input, 'temb': temb})
            output = new_input[:, -1] 
            save_like(output, data, step, save_dir)
            print(f'stage: {i}, step: {step+1:02d}, output: {output.min():.2f} {output.max():.2f}')
            input = new_input
            step += 1

        run_time = time.perf_counter() - start
        print(f'Inference {stage} take {run_time:.2f}')

        if step > total_step:
            break


    
if __name__ == "__main__":
    data = xr.open_dataarray(args.input)
    run_inference(args.model, data, args.num_steps, args.save_dir)

