# cuTrace (CUDA / C++)

A GPU-accelerated ray tracer built from scratch. Includes core rendering logic (reflections, shadows, anti-aliasing), optimizations using Nsight, CPU/GPU benchmarking and CMake for build automation.

## Project Features

* 💡 Ray tracer with reflections shadows, and anti-aliasing 
* 🕹️ 3 render paths (multithreaded CPU, GPU, Opt. GPU)
* 🚀 One command builds, renders, and displays the result
* ⚙️ CLI configurability for resolution and render path
* 📈 Automated benchmarks with warmup runs and repeated trials
* 🧩 Modular file structure separating rendering logic, kernels, and CPU path
* 📁 Clean folder structure and CMake build automation
* 📐 Documented math derivations

## Rendered Output (NVIDIA RTX 5080, 1920 x 1080)

![cuTrace render](docs/Tracing.png)

## Project Structure

```
cuTrace/
├── CMakeLists.txt
├── render.sh
├── .gitignore
├── .gitattributes
├── LICENSE
├── README.md
├── src/
│   ├── main.cu
│   ├── gpu_naive.cu
│   ├── gpu_optimized.cu
│   └── cpu_render.cu
├── include/
│   ├── vec3.cuh
│   ├── ray.cuh
│   ├── sphere.cuh
│   ├── intersect.cuh
│   └── render.cuh
├── benchmark/
│   ├── benchmark.sh
│   ├── generate_report.py
│   └── benchmark_results.md
└── docs/
    ├── derivations.md
    └── Tracing.png
```

## Prerequisites

- NVIDIA GPU
- CUDA Toolkit 12.x or newer
- C++17 compatible compiler (usually installed automatically with the CUDA Toolkit)
- Python 3
- CMake 3.18+
- Bash
- Git
- ImageMagick (converts ppm to png)

## Setup

**Option A: Manually Build + Execute**


```bash
git clone https://github.com/MohamedFouad132/cuTrace.git
cd cuTrace
mkdir build && cd build
cmake ..
make
./raytracer [Flags]
convert output.ppm output.png
```

**Option B: Use `render.sh` to Build + Execute**


```bash
git clone https://github.com/MohamedFouad132/cuTrace.git
cd cuTrace
chmod +x render.sh
./render.sh [Flags]
```

**Flags**
```
--width <n>     Image width in pixels (default: 1920)
--height <n>    Image height in pixels (default: 1080)
--mode <mode>   Render path: cpu, gpu, gpu-optimized (default: gpu)
--benchmark     Write precise render time to last_render_time.txt
--help, -h      Show usage information
```

Open `build/output.png` to view result.

## Benchmarks

To benchmark across all modes and resolutions

```bash
cd benchmark
chmod +x benchmark.sh
./benchmark.sh
```

## Results


**NVIDIA RTX 5080**

| Resolution | Mode | Mean (ms) | StdDev (ms) | Min (ms) | Max (ms) | Speedup vs CPU|
|---|---|---|---|---|---|---|
| 640x480 | cpu | 173.43 | 3.55 | 167.70 | 176.29 | 1.00x |
| 640x480 | gpu | 0.18 | 0.01 | 0.17 | 0.19 | 967.39x |
| 640x480 | gpu-optimized | 0.08 | 0.00 | 0.08 | 0.09 | 2122.80x |
| 1280x720 | cpu | 518.46 | 2.42 | 515.32 | 520.91 | 1.00x |
| 1280x720 | gpu | 0.21 | 0.00 | 0.21 | 0.22 | 2429.53x |
| 1280x720 | gpu-optimized | 0.12 | 0.01 | 0.12 | 0.13 | 4271.39x |
| 1920x1080 | cpu | 1171.76 | 10.67 | 1162.40 | 1190.17 | 1.00x |
| 1920x1080 | gpu | 0.30 | 0.00 | 0.29 | 0.31 | 3911.59x |
| 1920x1080 | gpu-optimized | 0.19 | 0.00 | 0.19 | 0.19 | 6141.28x |
| 3840x2160 | cpu | 4699.12 | 24.10 | 4676.46 | 4740.13 | 1.00x |
| 3840x2160 | gpu | 0.70 | 0.01 | 0.69 | 0.72 | 6693.52x |
| 3840x2160 | gpu-optimized | 0.57 | 0.00 | 0.56 | 0.58 | 8239.73x |

## Optimizations:

**Problem:** Theoretical occupancy  was capped at 75% due to register pressure (68 registers/thread, 3 resident blocks/SM vs a 4-block ceiling)

**Solution:** Replaced `curandState` with a lightweight LCG. Register pressure dropped enough to reach 100% theoretical occupancy (~70% → ~90% achieved).

**Problem:** Shadow rays used `hit_scene`, which finds the closest blocker even though shadows only need a yes/no answer

**Solution:** Replaced with `hit_anything`, an early-exit check that returns on the first blocker found

**Problem:** `spheres[hit_index]` was read from global memory up to 5 times per bounce, and accessed per-thread with a varying index (uncoalesced, 4 of 32 bytes/sector utilized)

**Solution:** Cached the hit sphere's data in a local variable, and moved the sphere array into shared memory to reduce redundant global reads

**Investigated and ruled out:** Warp divergence was hypothesized as a risk given variable bounce depth, but Nsight showed 99.47% branch efficiency. This is likely because the scene is not very complex (only 4 objects)


## Key Findings

- CPU (multithreaded) vs GPU speedup: 3,500.51x average
- CPU (multithreaded) vs optimized GPU speedup: 5,193.80x average
- Largest single speedup: 8,239.73x (3840×2160, optimized GPU vs CPU)
- Naive → optimized GPU speedup: 1.70x average



## Derivations

Full derivations for the intersection and reflection formulas: [docs/derivations.md](docs/derivations.md)

## License

MIT License: [LICENSE](LICENSE)


