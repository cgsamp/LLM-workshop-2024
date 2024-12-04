import py3nvml.py3nvml as nvml
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import time

def create_graph(filename):
    # Read the CSV file
    data = pd.read_csv(
        filename + ".log",
        header=None,
        names=["datetime", "series1", "series2", "series3", "series4"]
    )

    # Convert the 'datetime' column to a datetime object
    data["datetime"] = pd.to_datetime(data["datetime"], format="%Y%m%d-%H:%M:%S")

    # Plot the data
    plt.figure(figsize=(12, 6))

    # Plot each series
    plt.plot(data["datetime"], data["series1"], label="Series 1", marker="o")
    plt.plot(data["datetime"], data["series2"], label="Series 2", marker="s")
    plt.plot(data["datetime"], data["series3"], label="Series 3", marker="^")
    plt.plot(data["datetime"], data["series4"], label="Series 4", marker="x")

    # Add labels, title, and legend
    plt.xlabel("Datetime")
    plt.ylabel("Values")
    plt.title("Line Graph of Four Data Series")
    plt.legend()

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(filename+".png")


def monitor():
    log_file_name = f'nvidia-temperature-{datetime.now().strftime("%Y%m%d%H%M%S")}'
    file = open(log_file_name+".csv", "w")

    try:
        nvml.nvmlInit()
        print(f"Driver Version: {nvml.nvmlSystemGetDriverVersion()}")
        print(f"Number of GPUs: {nvml.nvmlDeviceGetCount()}")
        for i in range(nvml.nvmlDeviceGetCount()):
            handle = nvml.nvmlDeviceGetHandleByIndex(i)
            print(f"GPU {i}: {nvml.nvmlDeviceGetName(handle)} Bus ID {nvml.nvmlDeviceGetPciInfo(handle).busId.decode()} Temp: {nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)}")

        while True:
            current_time = datetime.now().strftime("%Y%m%d-%H:%M:%S")
            line = [current_time]
            for i in range(nvml.nvmlDeviceGetCount()):
                handle = nvml.nvmlDeviceGetHandleByIndex(i)
                line.append(str(nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)))
            print(",".join(line))
            file.write(",".join(line)+"\n")
            time.sleep(1)

        nvml.nvmlShutdown()
    except nvml.NVMLError as e:
        print(f"NVML Error: {str(e)}")
    except KeyboardInterrupt:
        print("Interrupted by user. Creating grapg and xiting...")
        create_graph(log_file_name)


if __name__ == "__main__":
    monitor()