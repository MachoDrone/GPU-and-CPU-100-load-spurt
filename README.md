# GPU/CPU 100% Load Test, Stress Test

Load your GPU(s) and CPU to 100% for a specified duration.

## Usage

```bash
curl -s https://raw.githubusercontent.com/MachoDrone/GPU-and-CPU-100-load-spurt/refs/heads/main/loadup.py | python3 - --gpu 0 --duration 60 --cleanup y --docker y
```

### Arguments
- `--gpu N`: GPU(s) to stress (e.g., `0`, `1`, `0,2`, or `all`). N is the GPU index from `nvidia-smi`.
- `--duration N`: Seconds to run the test (default: 60).
- `--cleanup y|n`: Remove virtual env and Docker artifacts after test (default: y).
- `--docker y|n`: Run inside a Docker container (requires Docker and NVIDIA Container Toolkit; default: y).

Use environment variables instead of args:
```bash
curl -s URL | GPU=0 DURATION=60 CLEANUP=y DOCKER=y python3 -
```
