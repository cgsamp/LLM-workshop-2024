import torch
import torch.nn as nn
import py3nvml.py3nvml as nvml
import argparse

from importlib.metadata import version

class ExampleModel(nn.Module):
    def forward(self, x):
        # Heavy computation: Matrix multiplication and element-wise operations
        for _ in range(200):  # Increase the loop count to load the GPU more
            x = torch.matmul(x, x.T)  # Matrix multiplication
            x = x ** 2  # Element-wise square
        return x

def check_dependencies():
    pkgs = [
            "torch"
        ]
    for p in pkgs:
        print(f"{p} version: {version(p)}")

    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")

    if cuda_available:
        # Check the number of GPUs
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs: {num_gpus}")
        for i in range(num_gpus):
            print(f"GPU {i} Name: {torch.cuda.get_device_name(i)}")

    try:
        nvml.nvmlInit()
        print(f"Driver Version: {nvml.nvmlSystemGetDriverVersion()}")
        print(f"Number of GPUs: {nvml.nvmlDeviceGetCount()}")
        for i in range(nvml.nvmlDeviceGetCount()):
            handle = nvml.nvmlDeviceGetHandleByIndex(i)
            print(f"GPU {i}: {nvml.nvmlDeviceGetName(handle)}")
            print(f"Temp {i}: {nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)}")

        nvml.nvmlShutdown()
    except nvml.NVMLError as e:
        print(f"NVML Error: {str(e)}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process GPU settings.")

    # Positional argument: max_temp
    parser.add_argument(
        "max_temp", 
        type=int, 
        help="Maximum temperature for GPU throttling (integer)."
    )

    # Optional keyword argument: batch_size
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=1000,  # Set a default value
        help="Batch size for processing (integer, default: 1000)."
    )

    # Optional keyword argument: dims
    parser.add_argument(
        "--dims", 
        type=int, 
        default=1000,  # Set a default value
        help="Dims for testing."
    )

    # Optional keyword argument: gpus
    parser.add_argument(
        "--gpus", 
        type=lambda s: [int(item) for item in s.split(",")],  # Convert comma-separated string to a list of integers
        default=[],  # Default is an empty list
        help="Comma-separated list of GPU indices (e.g., '0' or '0,2,3')."
    )

    return parser.parse_args()


def check_current_temperature(nvml):
    max_gpu_temp = 0
    try:
        for i in range(nvml.nvmlDeviceGetCount()):
            handle = nvml.nvmlDeviceGetHandleByIndex(i)
            gpu_temp = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
            if max_gpu_temp < gpu_temp:
                max_gpu_temp = gpu_temp
    except nvml.NVMLError as e:
        print(f"NVML Error: {str(e)}")
    return max_gpu_temp

def load_gpu(max_allowed_gpu_temp, batch_size, dims, gpus):
    nvml.nvmlInit()
    # Create a model and move it to DataParallel for multi-GPU computation
    model = ExampleModel()
    model = nn.DataParallel(model)  # Replicates model on all GPUs
    model = model.cuda()  # Move the model to GPUs

    # Generate a large tensor to load the GPUs
    batch_size = batch_size
    input_dim = dims  # Adjust these values to control GPU load
    x = torch.randn(batch_size, input_dim, device="cuda")

    print("Starting GPU stress test... Press Ctrl+C to stop.")
    try:
        current_temp = check_current_temperature(nvml)
        while current_temp < max_allowed_gpu_temp:  # Loop indefinitely until interrupted_size
            output = model(x)  # Perform the computation
            torch.cuda.synchronize()  # Ensure all GPU computations are completed
            current_temp = check_current_temperature(nvml)
            print(f".{current_temp}",end="",flush=True)
    except KeyboardInterrupt:
        print("Interrupted by user. Exiting...")




if __name__ == "__main__":

    args = parse_arguments()
    print(f"Max Temperature: {args.max_temp}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Dims: {args.dims}")
    print(f"GPUs: {args.gpus}")
    check_dependencies()
    #load_gpu(args.max_temp,args.batch_size, args.dims, args.gpus)