#!/usr/bin/env python3
# ──────────────────────────────────────────────────────────
# Usage:
#   curl -s https://raw.githubusercontent.com/MachoDrone/GPU-and-CPU-100-load-spurt/refs/heads/main/loadup.py | python3 - --gpu 0 --duration 60 --cleanup y --docker y
#
# Arguments:
#   --gpu N        GPU number to stress test (0, 1, 2, etc.)
#   --duration N   How many seconds to run the stress test
#   --cleanup y|n  Remove venv and Docker artifacts after test
#   --docker y|n   Run inside a Docker container (requires Docker + NVIDIA Container Toolkit)
#
# Environment variables can be used instead of args:
#   curl -s URL | GPU=0 DURATION=60 CLEANUP=y DOCKER=y python3 -
# ──────────────────────────────────────────────────────────
import subprocess
import time
import multiprocessing as mp
import sys
import os
import re
import platform
import argparse
import tempfile
VERSION = "0.5.1"
SCRIPT_URL = "https://raw.githubusercontent.com/MachoDrone/GPU-and-CPU-100-load-spurt/refs/heads/main/loadup.py"

# Parse command-line arguments FIRST (before any setup)
parser = argparse.ArgumentParser(
    description="CPU/GPU Stress Test Script",
    epilog="""Environment variables (override defaults when no TTY):
  GPU=N         GPU number to stress (same as --gpu)
  DURATION=N    Duration in seconds (same as --duration)
  CLEANUP=y|n   Auto-cleanup after test (same as --cleanup)
  DOCKER=y|n    Run inside Docker container (same as --docker)

Piped usage examples:
  curl -s URL | python3 - --gpu 0 --duration 60 --cleanup y             # on host
  curl -s URL | python3 - --gpu 0 --duration 60 --cleanup y --docker y  # in Docker
""",
    formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument("--gpu", type=int, default=None, help="GPU number to stress (default: prompt or 0)")
parser.add_argument("--duration", type=int, default=None, help="Duration in seconds (default: prompt or 30)")
parser.add_argument("--cleanup", type=str, default=None, choices=["y", "n"],
                    help="Cleanup venv/Dockerfile after test: y or n (default: prompt or skip)")
parser.add_argument("--docker", type=str, default=None, choices=["y", "n"],
                    help="Run inside Docker container: y or n (requires Docker + NVIDIA Container Toolkit)")
args = parser.parse_args()

# Apply environment variable fallbacks (CLI args take priority)
if args.gpu is None and os.environ.get("GPU", "").strip().lstrip("-").isdigit():
    args.gpu = int(os.environ["GPU"])
if args.duration is None and os.environ.get("DURATION", "").strip().isdigit():
    args.duration = int(os.environ["DURATION"])
if args.cleanup is None and os.environ.get("CLEANUP", "").strip().lower() in ("y", "n"):
    args.cleanup = os.environ["CLEANUP"].strip().lower()
if args.docker is None and os.environ.get("DOCKER", "").strip().lower() in ("y", "n"):
    args.docker = os.environ["DOCKER"].strip().lower()

print(f"\033[94mVersion: {VERSION}\033[0m")
# Show Docker warning only when Docker mode is requested and not already in a container
if args.docker == 'y' and not (os.path.exists('/.dockerenv') or 'docker' in platform.uname().release.lower()):
    print("\033[91mDOCKER CONTAINER CREATION MAY HAVE LONG PAUSES -- THIS IS NORMAL\033[0m")
    time.sleep(3)

# Detect if script is being piped (e.g., curl | python3 -)
is_piped = not os.isatty(sys.stdin.fileno())

if is_piped:
    print("Piped execution detected. To customize, use CLI args or env vars:")
    print("  curl -s URL | python3 - --gpu 0 --duration 60 --cleanup y")
    print("  curl -s URL | python3 - --gpu 0 --duration 60 --cleanup y --docker y")
    print("For interactive prompts, download and run as file instead:")
    print(f"  curl -s -O {SCRIPT_URL} && python3 loadup.py")
    # Drain any remaining stdin so it doesn't interfere with subprocesses
    try:
        sys.stdin.read()
    except Exception:
        pass

# ──────────────────────────────────────────────────────────
# AUTO-INSTALL SYSTEM DEPENDENCIES (hands-free operation)
# ──────────────────────────────────────────────────────────

def install_system_deps():
    """Detect and auto-install missing system packages."""
    packages_needed = []

    # python3.X-venv: needed for virtual environment creation
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    try:
        subprocess.check_call([sys.executable, '-m', 'ensurepip', '--version'],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError):
        packages_needed.append(f"python{py_ver}-venv")

    # lm-sensors: needed for CPU temperature readings
    try:
        subprocess.check_call(["sensors", "--version"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError):
        packages_needed.append("lm-sensors")

    if not packages_needed:
        return  # All deps present

    # Determine sudo mode:
    # -n (passwordless) if available, otherwise regular sudo (prompts via /dev/tty)
    has_passwordless_sudo = subprocess.run(
        ["sudo", "-n", "true"], capture_output=True).returncode == 0
    if has_passwordless_sudo:
        sudo = ["sudo"]
    elif os.path.exists("/dev/tty"):
        # sudo will prompt for password on the terminal even in piped mode
        print(f"Missing packages: {' '.join(packages_needed)}")
        print("sudo password may be required...")
        sudo = ["sudo"]
    else:
        print(f"Missing packages: {' '.join(packages_needed)}")
        print(f"No sudo access and no terminal for password prompt.")
        print(f"Please run manually: sudo apt install {' '.join(packages_needed)}")
        sys.exit(1)

    print(f"Installing system packages: {' '.join(packages_needed)}")
    installed = False

    # Attempt 1: install directly (works if package cache exists)
    r = subprocess.run(sudo + ["apt-get", "install", "-y"] + packages_needed)
    if r.returncode == 0:
        installed = True

    # Attempt 2: update package lists, then install
    if not installed:
        print("Package cache outdated. Running apt-get update...")
        subprocess.run(sudo + ["apt-get", "update"])
        r = subprocess.run(sudo + ["apt-get", "install", "-y"] + packages_needed)
        if r.returncode == 0:
            installed = True

    # Attempt 3: fix broken packages, then install
    if not installed:
        print("Attempting to fix broken packages...")
        subprocess.run(sudo + ["apt-get", "install", "--fix-broken", "-y"])
        r = subprocess.run(sudo + ["apt-get", "install", "-y"] + packages_needed)
        if r.returncode == 0:
            installed = True

    if installed:
        print("System packages installed.")
    else:
        print(f"ERROR: Could not install packages automatically.")
        print(f"Please run manually: sudo apt install {' '.join(packages_needed)}")
        sys.exit(1)

install_system_deps()

# ──────────────────────────────────────────────────────────
# DETECT CUDA VERSION FROM NVIDIA DRIVER
# ──────────────────────────────────────────────────────────

def detect_cuda_version():
    """Parse nvidia-smi to get the max CUDA version the driver supports.
    Returns the best matching PyTorch CUDA wheel tag (e.g., 'cu126')."""
    try:
        output = subprocess.check_output(["nvidia-smi"]).decode("utf-8")
        match = re.search(r"CUDA Version:\s+([\d.]+)", output)
        if not match:
            print("WARNING: Could not parse CUDA version from nvidia-smi.")
            return None
        cuda_ver = match.group(1)  # e.g., "12.0", "12.6", "13.0"
        major, minor = [int(x) for x in cuda_ver.split(".")[:2]]
        print(f"NVIDIA driver supports CUDA {cuda_ver}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("WARNING: nvidia-smi not found. Cannot detect CUDA version.")
        return None

    # Map driver CUDA version to best matching PyTorch wheel.
    # Rule: driver CUDA version must be >= PyTorch CUDA build version.
    # Available PyTorch wheels: cu118, cu121, cu124, cu126, cu130
    if major >= 13:
        tag = "cu130"
    elif major == 12 and minor >= 6:
        tag = "cu126"
    elif major == 12 and minor >= 4:
        tag = "cu124"
    elif major == 12 and minor >= 1:
        tag = "cu121"
    elif major >= 11 and (major > 11 or minor >= 8):
        tag = "cu118"
    else:
        print(f"WARNING: CUDA {cuda_ver} is too old for PyTorch GPU support (need >= 11.8).")
        print("Please update your NVIDIA driver: https://www.nvidia.com/Download/index.aspx")
        return None

    print(f"Using PyTorch CUDA build: {tag}")
    return tag

cuda_tag = detect_cuda_version()
torch_index_url = f"https://download.pytorch.org/whl/{cuda_tag}" if cuda_tag else None

# ──────────────────────────────────────────────────────────
# EARLY PROMPTS: GPU selection and duration BEFORE any setup
# ──────────────────────────────────────────────────────────

def get_gpu_list():
    """Get list of available NVIDIA GPUs via nvidia-smi"""
    try:
        output = subprocess.check_output(["nvidia-smi", "-L"]).decode("utf-8").strip()
        lines = output.split("\n")
        gpus = []
        for line in lines:
            if line.startswith("GPU "):
                parts = line.split(": ", 1)
                idx = parts[0].split(" ")[1]
                name = parts[1].split(" (")[0].strip()
                gpus.append((int(idx), name))
        return gpus
    except Exception as e:
        print(f"Error listing GPUs: {e}")
        return []

# Resolve GPU selection: CLI/env > interactive prompt > default
gpus = get_gpu_list()
if not gpus:
    print("No NVIDIA GPUs found. GPU stress will be disabled.")
    args.gpu = -1  # Sentinel: no GPU available
else:
    print("Available GPUs:")
    for idx, name in gpus:
        print(f"  {idx}: {name}")

    if args.gpu is None:
        if os.isatty(sys.stdin.fileno()):
            gpu_input = input("Enter GPU number to stress (default 0): ").strip()
            args.gpu = int(gpu_input) if gpu_input else 0
        else:
            args.gpu = 0
            print(f"Using default GPU {args.gpu}.")
    else:
        print(f"Using GPU {args.gpu} (from {'--gpu' if '--gpu' in sys.argv else 'GPU env var'}).")

    # Always validate GPU ID
    if args.gpu not in [idx for idx, _ in gpus]:
        print(f"Invalid GPU number {args.gpu}. Available: {[idx for idx, _ in gpus]}. Exiting.")
        sys.exit(1)

# Resolve duration: CLI/env > interactive prompt > default
if args.duration is None:
    if os.isatty(sys.stdin.fileno()):
        duration_input = input("Enter number of seconds to run (default 30): ").strip()
        args.duration = 30 if not duration_input else int(duration_input)
    else:
        args.duration = 30
        print(f"Using default duration {args.duration} seconds.")
else:
    print(f"Using duration {args.duration}s (from {'--duration' if '--duration' in sys.argv else 'DURATION env var'}).")

# Inject resolved args into sys.argv so they carry through os.execv
def ensure_args_in_argv():
    """Make sure --gpu, --duration, --cleanup, --docker are in sys.argv for re-execution"""
    new_argv = [sys.argv[0]]
    skip_args = ('--gpu', '--duration', '--cleanup', '--docker')
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] in skip_args and i + 1 < len(sys.argv):
            i += 2  # Skip existing arg and its value
        else:
            new_argv.append(sys.argv[i])
            i += 1
    if args.gpu is not None:
        new_argv.extend(["--gpu", str(args.gpu)])
    if args.duration is not None:
        new_argv.extend(["--duration", str(args.duration)])
    if args.cleanup is not None:
        new_argv.extend(["--cleanup", str(args.cleanup)])
    # NOTE: --docker is NOT passed through (container runs on bare metal inside)
    sys.argv = new_argv

ensure_args_in_argv()

# ──────────────────────────────────────────────────────────
# Docker setup (--docker y)
# ──────────────────────────────────────────────────────────

def is_in_docker():
    """Check if running inside Docker container"""
    return os.path.exists('/.dockerenv') or 'docker' in platform.uname().release.lower()

def setup_docker():
    """Build and run stress test in a Docker container."""
    if args.docker != 'y':
        return  # Docker not requested

    if is_in_docker():
        print("Already running in Docker. Proceeding...")
        return

    print("Docker mode requested. Setting up container...")

    # Check Docker is installed
    try:
        subprocess.check_output(["docker", "--version"], stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ERROR: Docker not found. Install Docker + NVIDIA Container Toolkit first.")
        print("  https://docs.docker.com/engine/install/")
        print("  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html")
        sys.exit(1)

    # Check NVIDIA Container Toolkit (docker --gpus)
    r = subprocess.run(["docker", "run", "--rm", "--gpus", "all", "ubuntu:22.04", "true"],
                       capture_output=True, timeout=30)
    if r.returncode != 0:
        print("ERROR: NVIDIA Container Toolkit not working (docker --gpus all failed).")
        print("  Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html")
        sys.exit(1)

    # Download the script to a real file (needed for Docker COPY)
    script_path = os.path.join(os.getcwd(), "loadup.py")
    if not os.path.exists(script_path):
        print(f"Downloading script to {script_path}...")
        try:
            import urllib.request
            urllib.request.urlretrieve(SCRIPT_URL, script_path)
        except Exception as e:
            print(f"ERROR: Failed to download script: {e}")
            sys.exit(1)

    # Map CUDA tag to Docker base image
    cuda_docker_images = {
        "cu118": "nvidia/cuda:11.8.0-base-ubuntu22.04",
        "cu121": "nvidia/cuda:12.1.0-base-ubuntu22.04",
        "cu124": "nvidia/cuda:12.4.0-base-ubuntu22.04",
        "cu126": "nvidia/cuda:12.6.0-base-ubuntu22.04",
        "cu130": "nvidia/cuda:13.0.0-base-ubuntu22.04",
    }
    base_image = cuda_docker_images.get(cuda_tag, "nvidia/cuda:12.6.0-base-ubuntu22.04")
    pip_index = torch_index_url or "https://download.pytorch.org/whl/cu126"

    dockerfile_content = f"""FROM {base_image}
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
RUN apt-get update && apt-get install -y python3 python3-pip python3-venv lm-sensors && \\
    rm -rf /var/lib/apt/lists/*
COPY loadup.py /app/loadup.py
WORKDIR /app
RUN python3 -m venv /app/venv && \\
    . /app/venv/bin/activate && \\
    pip install numpy psutil && \\
    pip install torch --index-url {pip_index}
"""
    dockerfile_path = os.path.join(os.getcwd(), "Dockerfile")
    with open(dockerfile_path, "w") as f:
        f.write(dockerfile_content)

    # Build image
    print("Building Docker image (this may take a few minutes on first run)...")
    try:
        subprocess.check_call(["docker", "build", "-t", "loadup-gpu", "."])
    except subprocess.CalledProcessError:
        print("ERROR: Docker build failed.")
        sys.exit(1)

    # Build docker run command -- pass all args EXCEPT --docker (bare metal inside container)
    print("Running stress test in Docker container...")
    docker_cmd = ["docker", "run", "--gpus", "all", "--rm"]
    if os.isatty(sys.stdin.fileno()):
        docker_cmd.extend(["-it"])
    docker_cmd.extend([
        "loadup-gpu",
        "/app/venv/bin/python", "/app/loadup.py",
        "--gpu", str(args.gpu), "--duration", str(args.duration),
    ])
    if args.cleanup:
        docker_cmd.extend(["--cleanup", args.cleanup])

    ret = subprocess.call(docker_cmd)

    # Cleanup Docker artifacts on host
    if args.cleanup == 'y':
        if os.path.exists(dockerfile_path):
            os.remove(dockerfile_path)
        if is_piped and os.path.exists(script_path):
            os.remove(script_path)
        subprocess.call(["docker", "rmi", "loadup-gpu"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("\033[94mDocker artifacts cleaned up.\033[0m")

    sys.exit(ret)

setup_docker()

# ──────────────────────────────────────────────────────────
# CUDA check
# ──────────────────────────────────────────────────────────

def check_cuda_installed():
    """Check for CUDA toolkit. Informational only -- PyTorch ships its own CUDA runtime."""
    try:
        output = subprocess.check_output(["nvcc", "--version"]).decode("utf-8")
        print(f"CUDA toolkit found: {output.strip().splitlines()[-1]}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("CUDA toolkit (nvcc) not installed. Not required -- PyTorch ships its own CUDA runtime.")

# ──────────────────────────────────────────────────────────
# Virtual environment setup
# ──────────────────────────────────────────────────────────

# Track venv python binary for subprocess use (e.g., GPU stress in piped mode)
_venv_python = sys.executable  # Default: current interpreter (correct if already in venv)

# Check if running in a virtual environment
if sys.prefix == sys.base_prefix:
    venv_dir = 'venv'
    venv_pip = os.path.join(venv_dir, 'bin', 'pip')
    venv_python = os.path.join(venv_dir, 'bin', 'python')

    # Create venv if missing or broken (pip missing after partial cleanup)
    if not os.path.exists(venv_pip):
        print("Setting up virtual environment...")
        subprocess.check_call([sys.executable, '-m', 'venv', '--clear', venv_dir])
        # If pip still missing, try ensurepip
        if not os.path.exists(venv_pip):
            print("pip not found in venv, bootstrapping with ensurepip...")
            subprocess.check_call([os.path.join(venv_dir, 'bin', 'python'),
                                   '-m', 'ensurepip', '--upgrade'])
        if not os.path.exists(venv_pip):
            print("ERROR: Could not create venv with pip. Try: sudo apt install python3-venv")
            sys.exit(1)
    else:
        print("Virtual environment found.")

    # Install dependencies in the venv
    subprocess.check_call([venv_pip, 'install', 'numpy', 'psutil'])
    subprocess.check_call([venv_pip, 'uninstall', '-y', 'torch', 'torchaudio', 'torchvision'])
    if torch_index_url:
        subprocess.check_call([venv_pip, 'install', 'torch', '--index-url', torch_index_url])
    else:
        # CPU-only fallback if no CUDA detected
        subprocess.check_call([venv_pip, 'install', 'torch'])

    if is_piped:
        # Piped mode: os.execv won't work because Python already consumed stdin
        # (the temp file would be empty). Activate venv inline instead.
        print("Activating virtual environment inline (piped mode)...")
        _venv_python = os.path.abspath(venv_python)
        import site
        venv_lib = os.path.join(os.path.abspath(venv_dir), 'lib')
        # Find the pythonX.Y directory inside venv/lib/
        python_dirs = [d for d in os.listdir(venv_lib) if d.startswith('python')]
        if python_dirs:
            venv_site = os.path.join(venv_lib, python_dirs[0], 'site-packages')
            sys.path.insert(0, venv_site)
            site.addsitedir(venv_site)
        else:
            print("ERROR: Could not find venv site-packages. Exiting.")
            sys.exit(1)
    else:
        print("Starting script in virtual environment...")
        # sys.argv already contains --gpu and --duration from ensure_args_in_argv()
        os.execv(venv_python, [venv_python] + sys.argv)

# ──────────────────────────────────────────────────────────
# Now import the dependencies (we are in venv or activated inline)
# ──────────────────────────────────────────────────────────
import numpy as np
import psutil

def check_torch_cuda():
    """Verify PyTorch can see CUDA. Exit if not."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"PyTorch CUDA available: {torch.version.cuda}")
        else:
            print("PyTorch CUDA not available. Ensure CUDA toolkit is installed and paths are set.")
            sys.exit(1)
    except ImportError:
        print("Torch not installed correctly.")
        sys.exit(1)

# Run checks
check_cuda_installed()
check_torch_cuda()

# ──────────────────────────────────────────────────────────
# Stress test functions
# ──────────────────────────────────────────────────────────

# Function to stress CPU: Heavy matrix multiplication on all cores
def cpu_stress_worker():
    while True:
        a = np.random.rand(5000, 5000)
        b = np.random.rand(5000, 5000)
        np.dot(a, b)

# Function to stress GPU: Tensor operations on CUDA
def gpu_stress(gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    try:
        import torch
        print(f"Torch version: {torch.__version__}")
        print(f"CUDA version in Torch: {torch.version.cuda}")
        available = torch.cuda.is_available()
        print(f"CUDA available on GPU {gpu_id}: {available}")
        if not available:
            print(f"CUDA not available on GPU {gpu_id}. GPU stress disabled.")
            return
        device = torch.device("cuda")
        print(f"Using device: {torch.cuda.get_device_name(device)}")
        size = 20000  # Increased size for more stress
        _start = time.time()
        _dur = args.duration if args.duration else 30
        while True:
            try:
                a = torch.rand(size, size, device=device)
                b = torch.rand(size, size, device=device)
                for _ in range(10):
                    c = torch.mm(a, b)
                torch.cuda.synchronize()
                remaining = max(0, int(_dur - (time.time() - _start)))
                print(f"Running system load  {remaining}s remaining")
            except Exception as e:
                print(f"GPU stress error: {e}")
                break
    except ImportError as e:
        print(f"Import error in GPU stress: {e}")

# Function to get GPU metrics via nvidia-smi for a specific GPU
def get_gpu_metrics(gpu_id):
    try:
        command = [
            "nvidia-smi", "-i", str(gpu_id),
            "--query-gpu=fan.speed,pstate,power.draw,power.limit,memory.used,memory.total,utilization.gpu,temperature.gpu,clocks_throttle_reasons.hw_thermal_slowdown,clocks_throttle_reasons.sw_thermal_slowdown",
            "--format=csv,noheader"
        ]
        output = subprocess.check_output(command).decode("utf-8").strip()
        metrics = output.split(", ")
        return {
            "Fan Speed": metrics[0],
            "Perf (P-State)": metrics[1],
            "Power Usage/Cap": f"{metrics[2]}/{metrics[3]}",
            "VRAM Usage/Cap": f"{metrics[4]}/{metrics[5]}",
            "GPU Utilization": metrics[6],
            "GPU Temperature": metrics[7],
            "HW Throttle": metrics[8],
            "SW Throttle": metrics[9]
        }
    except Exception as e:
        print(f"Error querying nvidia-smi for GPU {gpu_id}: {e}")
    return None

# Function to get CPU metrics
def get_cpu_metrics():
    cpu_percent = psutil.cpu_percent(interval=0.1)
    cpu_freq = psutil.cpu_freq()
    freq_current = cpu_freq.current if cpu_freq else "N/A"

    try:
        temp_output = subprocess.check_output(["sensors"]).decode("utf-8")
        match = re.search(r"Package id 0:\s+\+([\d.]+)°C", temp_output)
        temp = match.group(1) + "°C" if match else "N/A"
    except Exception:
        temp = "N/A (Install lm-sensors or adapt for your OS)"

    return {
        "CPU Utilization": f"{cpu_percent}%",
        "CPU Frequency": f"{freq_current} MHz" if freq_current != "N/A" else "N/A",
        "CPU Temperature": temp
    }

# Function to get RAM metrics
def get_ram_metrics():
    mem = psutil.virtual_memory()
    return {
        "RAM Usage/Cap": f"{mem.used / (1024**2):.1f} MiB / {mem.total / (1024**2):.1f} MiB",
        "RAM Utilization": f"{mem.percent}%"
    }

# Function to get Storage metrics (I/O activity)
def get_storage_metrics():
    io1 = psutil.disk_io_counters()
    time.sleep(1)
    io2 = psutil.disk_io_counters()

    read_rate = (io2.read_bytes - io1.read_bytes) / (1024 ** 2)
    write_rate = (io2.write_bytes - io1.write_bytes) / (1024 ** 2)

    return {
        "Disk Read Rate": f"{read_rate:.2f} MB/s",
        "Disk Write Rate": f"{write_rate:.2f} MB/s"
    }

# Function to print metrics in blue
def print_blue(text):
    print(f"\033[94m{text}\033[0m")

# ──────────────────────────────────────────────────────────
# Main execution
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    # GPU and duration were already collected at the top of the script
    gpu_id = args.gpu if args.gpu is not None and args.gpu >= 0 else None
    duration = args.duration if args.duration is not None else 30

    # Multiprocessing strategy depends on execution mode:
    # - Piped mode (curl | python3 -): 'spawn' fails because __file__ is <stdin>,
    #   and child processes can't re-import from '/home/user/<stdin>'.
    #   Use 'fork' for CPU workers (numpy only, no CUDA), and subprocess for GPU.
    # - File mode (python3 loadup.py): 'spawn' works normally for CUDA safety.
    if is_piped:
        cpu_ctx = mp.get_context('fork')
    else:
        mp.set_start_method('spawn')
        cpu_ctx = mp.get_context('spawn')

    print(f"\nStarting CPU and GPU stress test for {duration} seconds on GPU {gpu_id if gpu_id is not None else 'N/A'}. Press Ctrl+C to stop early.")
    print("Metrics will update every 5 seconds.\n")

    # ── GPU stress FIRST (needs CPU to import torch before CPU saturation) ──
    gpu_process = None
    _gpu_temp_file = None  # Track temp file for cleanup
    if gpu_id is not None:
        if is_piped:
            # Piped mode: use subprocess with venv python to avoid spawn __file__ issue.
            # Write GPU code to a temp file (more reliable than -c for complex scripts).
            # Split imports so we can see torch loading progress.
            gpu_stress_code = f"""#!/usr/bin/env python3
import os, sys, time as _time
print("GPU stress subprocess started, importing torch...", flush=True)
import torch
print(f"Torch loaded: {{torch.__version__}}, CUDA: {{torch.version.cuda}}", flush=True)
if not torch.cuda.is_available():
    print("CUDA not available on GPU {gpu_id}. GPU stress disabled.", flush=True)
    sys.exit(1)
device = torch.device("cuda")
print(f"CUDA available on GPU {gpu_id}: True", flush=True)
print(f"Using device: {{torch.cuda.get_device_name(device)}}", flush=True)
size = 20000
_start = _time.time()
_duration = {duration}
while True:
    remaining = max(0, int(_duration - (_time.time() - _start)))
    try:
        a = torch.rand(size, size, device=device)
        b = torch.rand(size, size, device=device)
        for _ in range(10):
            c = torch.mm(a, b)
        torch.cuda.synchronize()
        remaining = max(0, int(_duration - (_time.time() - _start)))
        print(f"Running system load  {{remaining}}s remaining", flush=True)
    except Exception as e:
        print(f"GPU stress error: {{e}}", flush=True)
        break
"""
            _gpu_temp = tempfile.NamedTemporaryFile(
                mode='w', suffix='_gpu_stress.py', delete=False, dir=os.getcwd()
            )
            _gpu_temp.write(gpu_stress_code)
            _gpu_temp.close()
            _gpu_temp_file = _gpu_temp.name
            print(f"GPU stress: launching subprocess with {_venv_python}")
            gpu_process = subprocess.Popen(
                [_venv_python, "-u", _gpu_temp_file],
                env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id), "PYTHONUNBUFFERED": "1"}
            )
            # Brief check: did the subprocess crash immediately?
            time.sleep(1)
            rc = gpu_process.poll()
            if rc is not None:
                print(f"WARNING: GPU stress subprocess exited immediately with code {rc}")
                gpu_process = None
            else:
                print(f"GPU stress subprocess running (PID {gpu_process.pid})")
        else:
            gpu_process = mp.Process(target=gpu_stress, args=(gpu_id,))
            gpu_process.daemon = True
            gpu_process.start()

        # Wait for GPU stress to initialize before saturating CPU.
        # Torch import + CUDA init is CPU-intensive; CPU workers would starve it.
        print("Waiting for GPU stress to become active before starting CPU stress...")
        gpu_ready = False
        for _wait in range(20):  # Up to 20 seconds
            time.sleep(1)
            metrics = get_gpu_metrics(gpu_id)
            if metrics:
                util_str = metrics.get("GPU Utilization", "0 %").strip()
                # Check if GPU utilization is above 0%
                if util_str and not util_str.startswith("0"):
                    print(f"GPU stress active (utilization: {util_str}). Starting CPU stress...")
                    gpu_ready = True
                    break
            # Also check if subprocess crashed
            if is_piped and gpu_process and gpu_process.poll() is not None:
                print(f"WARNING: GPU subprocess exited with code {gpu_process.returncode}")
                break
        if not gpu_ready:
            print("GPU stress may still be initializing. Starting CPU stress anyway...")

    # ── CPU stress AFTER GPU is active ──
    num_cores = mp.cpu_count()
    cpu_processes = [cpu_ctx.Process(target=cpu_stress_worker) for _ in range(num_cores)]
    for p in cpu_processes:
        p.daemon = True
        p.start()

    start_time = time.time()
    try:
        while time.time() - start_time < duration:
            if gpu_id is not None:
                gpu_data = get_gpu_metrics(gpu_id)
                if gpu_data:
                    print("GPU Metrics:")
                    for key, value in gpu_data.items():
                        print(f"  {key}: {value}")

            cpu_data = get_cpu_metrics()
            print("\nCPU Metrics:")
            for key, value in cpu_data.items():
                print(f"  {key}: {value}")

            print("\n" + "-" * 40 + "\n")
            time.sleep(5)
    except KeyboardInterrupt:
        print("Stopping stress test early.")
    finally:
        print("Stress test completed.")
        for p in cpu_processes:
            p.terminate()
        if gpu_process:
            gpu_process.terminate()

        # Wait a moment for systems to settle
        time.sleep(2)

        # Print final metrics in blue
        print_blue("-" * 40)
        print_blue("Completed Full Load -- MANUALLY CHECK FOR IDLE/NORMALCY")
        if gpu_id is not None:
            gpu_data = get_gpu_metrics(gpu_id)
            if gpu_data:
                print_blue("GPU Metrics:")
                for key, value in gpu_data.items():
                    print_blue(f"  {key}: {value}")

        cpu_data = get_cpu_metrics()
        print_blue("\nCPU Metrics:")
        for key, value in cpu_data.items():
            print_blue(f"  {key}: {value}")

        ram_data = get_ram_metrics()
        print_blue("\nRAM Metrics:")
        for key, value in ram_data.items():
            print_blue(f"  {key}: {value}")

        storage_data = get_storage_metrics()
        print_blue("\nStorage Metrics:")
        for key, value in storage_data.items():
            print_blue(f"  {key}: {value}")

        # Cleanup prompt
        if args.cleanup is not None:
            # Cleanup was specified via --cleanup or CLEANUP env var
            do_cleanup = args.cleanup == 'y'
            print(f"Cleanup {'enabled' if do_cleanup else 'skipped'} (from {'--cleanup' if '--cleanup' in sys.argv else 'CLEANUP env var'}).")
        elif os.isatty(sys.stdin.fileno()):
            cleanup_input = input("\033[91mCleanup now? (Y/n): \033[0m").strip().lower()
            do_cleanup = cleanup_input == '' or cleanup_input == 'y'
        else:
            print("No TTY detected. Skipping cleanup (use --cleanup y or CLEANUP=y to auto-cleanup).")
            do_cleanup = False

        if do_cleanup:
            if os.path.exists('venv'):
                subprocess.call(["rm", "-rf", "venv"])
            if os.path.exists('Dockerfile'):
                os.remove('Dockerfile')
            try:
                subprocess.call(["docker", "rmi", "loadup-gpu"],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass
            print_blue("Cleanup completed.")
        else:
            print("Cleanup skipped.")

        # Clean up GPU temp script if it was created
        if _gpu_temp_file and os.path.exists(_gpu_temp_file):
            os.remove(_gpu_temp_file)

        sys.exit(0)
