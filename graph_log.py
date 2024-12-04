import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import argparse

parser = argparse.ArgumentParser(description="Process GPU settings.")
parser.add_argument("filename")
filename = parser.parse_args().filename

# Sample CSV file
csv_file = "nvidia-temperature-20241204190035.log"  # Replace with your CSV file path

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
