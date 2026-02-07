# Load your GPU and CPU 100% with customized durations.
```
!/usr/bin/env python3
 ──────────────────────────────────────────────────────────
 Usage:
   curl -s https://raw.githubusercontent.com/MachoDrone/GPU-and-CPU-100-load-spurt/refs/heads/main/loadup.py | python3 - --gpu 0 --duration 60 --cleanup y --docker y

 Arguments:
   --gpu N        GPU number to stress test (0, 1, 2, etc.)
   --duration N   How many seconds to run the stress test
   --cleanup y|n  Remove venv and Docker artifacts after test
   --docker y|n   Run inside a Docker container (requires Docker + NVIDIA Container Toolkit)

 Environment variables can be used instead of args:
   curl -s URL | GPU=0 DURATION=60 CLEANUP=y DOCKER=y python3 -
 ──────────────────────────────────────────────────────────
