# generate_report.py
# Reads raw per-run benchmark timings from a CSV file and computes summary statistics
# writes a formatted markdown table comparing CPU, GPU, and GPU-optimized performance.

import sys
import csv
import statistics
from collections import defaultdict

def main():

    # Check for correct number of command-line arguments
    if len(sys.argv) != 3:
        print("Usage: python generate_report.py <input_csv> <output_md>")
        sys.exit(1)

    
    input_csv = sys.argv[1]
    output_md = sys.argv[2]

    # Group all timing values using (resolution, mode) as the key
    data = defaultdict(list)
    with open(input_csv, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            key = (row['resolution'], row['mode'])
            data[key].append(float(row['time_ms']))

    # Compute summary statistics for each (resolution, mode) key
    summary = {}
    for key, times in data.items():
        avg_time = statistics.mean(times)
        min_time = min(times)
        max_time = max(times)
        summary[key] = {
            "mean": avg_time,
            "stddev": statistics.stdev(times) if len(times) > 1 else 0.0,
            "min": min_time,
            "max": max_time
        }

    resolutions = list(dict.fromkeys(res for res, mode in summary.keys()))
    modes = list(dict.fromkeys(mode for res, mode in summary.keys()))

    # Generate the markdown report
    with open(output_md, 'w') as mdfile:
        mdfile.write("# Benchmark Results\n\n")
        mdfile.write("| Resolution | Mode | Mean (ms) | StdDev (ms) | Min (ms) | Max (ms) | Speedup vs CPU |\n")
        mdfile.write("|---|---|---|---|---|---|---|\n")

        for res in resolutions:
            # Get the mean CPU time for this resolution to calculate speedup
            cpu_mean = summary.get((res, 'cpu'), {}).get('mean')
            for mode in modes:
                if (res, mode) not in summary:
                    continue
                s = summary[(res, mode)]
                speedup = f"{cpu_mean / s['mean']:.2f}x" if cpu_mean else "N/A"
                mdfile.write(f"| {res} | {mode} | {s['mean']:.2f} | {s['stddev']:.2f} | {s['min']:.2f} | {s['max']:.2f} | {speedup} |\n")

    print(f"Benchmark report generated and saved to {output_md}")


if __name__ == "__main__":
    main()