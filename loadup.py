#!/usr/bin/env python3
import subprocess
import time
import multiprocessing as mp
import sys
import os
import re
import platform
import argparse
import tempfile

VERSION = "0.0.6"
print(f"Version: {VERSION}")
time.sleep(3)

# Detect if script is being piped (e.g., curl | python3 -)
is_piped = not os.isatty(sys.stdin.fileno())

if is_piped:
    # Save piped input to a temporary file to allow execv
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
        temp_file.write(sys.stdin.read())
        temp_script = temp_file.name
    os.chmod(temp_script, 0o755)
    sys.argv[0] = temp_script  # Update argv for execv

def is_in_docker():
    """Check if running inside Docker container"""
    return os.path.exists('/.dockerenv') or 'docker' in platform.uname().release.lower()

def setup_docker():
    """Build and run in Docker if not already in container"""
    if is_in_docker():
        print("Already running in Docker. Proceeding...")
        return
    
    print("Not in Docker. Setting up container for reliability...")
    
    # Check if Docker is installed
    try:
        subprocess.check_output(["docker", "--version"])
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Docker not found. Please install Docker to enable containerized mode for better reliability.")
        print("Continuing without Docker...")
        return
    
    # Generate Dockerfile in current dir
    dockerfile_content = """
FROM nvidia/cuda:13.0.0-base-ubuntu22.04

RUN apt-get update && apt-get install -y software-properties-common && \\
    add-apt-repository ppa:deadsnakes/ppa -y && apt-get update && \\
    apt-get install -y python3.12 python3.12-venv python3.12-dev build-essential lm-sensors && \\
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \\
    rm -rf /var/lib/apt/lists/*

COPY loadup.py /app/loadup.py

WORKDIR /app

RUN python3 -m venv /app/venv && \\
    . /app/venv/bin/activate && \\
    pip install numpy psutil && \\
    pip install torch --index-url https://download.pytorch.org/whl/cu130

CMD ["/app/venv/bin/python", "/app/loadup.py"]
"""
    with open("Dockerfile", "w") as f:
        f.write(dockerfile_content)
    
    # Build image
    try:
        subprocess.check_call(["docker", "build", "-t", "loadup-gpu", "."])
    except subprocess.CalledProcessError:
        print("Docker build failed. Continuing without Docker...")
        return
    
    # Run container with GPU access, interactive if TTY, remove on exit
    print("Running in Docker container...")
    docker_cmd = [
        "docker", "run", "--gpus", "all", "--rm",
        "-v", f"{os.getcwd()}:/app",  # Mount current dir if needed
        "loadup-gpu"
    ]
    if os.isatty(sys.stdin.fileno()):
        docker_cmd.insert(3, "-it")  # Add -it only if TTY available
    os.execvp("docker", docker_cmd)  # Replace current process with Docker run

# Run Docker setup first
setup_docker()

def check_cuda_installed():
    try:
        output = subprocess.check_output(["nvcc", "--version"]).decode("utf-8")
        if "release 13.0" in output:  # Check for your CUDA version
            print("CUDA 13.0 detected.")
            return True
        else:
            print("CUDA version mismatch. Expected 13.0.")
            return False
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("CUDA toolkit (nvcc) not found. Please install CUDA 13.0 following the instructions.")
        print("Run these commands:")
        print("wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb")
        print("sudo dpkg -i cuda-keyring_1.1-1_all.deb")
        print("sudo apt update")
        print("sudo apt install cuda-toolkit-13-0")
        print("Then add to ~/.bashrc:")
        print("export PATH=/usr/local/cuda-13.0/bin${PATH:+:${PATH}}")
        print("export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}")
        print("source ~/.bashrc")
        sys.exit(1)

# Check if running in a virtual environment
if sys.prefix == sys.base_prefix:
    venv_dir = 'venv'
    print("Setting up virtual environment...")
    if not os.path.exists(venv_dir):
        subprocess.check_call([sys.executable, '-m', 'venv', venv_dir])
    
    venv_pip = os.path.join(venv_dir, 'bin', 'pip')
    venv_python = os.path.join(venv_dir, 'bin', 'python')
    
    # Install dependencies in the venv
    subprocess.check_call([venv_pip, 'install', 'numpy', 'psutil'])
    subprocess.check_call([venv_pip, 'uninstall', '-y', 'torch', 'torchaudio', 'torchvision'])
    subprocess.check_call([venv_pip, 'install', 'torch', '--index-url', 'https://download.pytorch.org/whl/cu130'])
    
    print("Starting script in virtual environment...")
    os.execv(venv_python, [venv_python] + sys.argv)

# Now import the dependencies (we are in venv)
import numpy as np
import psutil

def check_torch_cuda():
    try:
        import torch
        if torch.cuda.is_available():
            print(f"PyTorch CUDA available: {torch.version.cuda}")
            return True
        else:
            print("PyTorch CUDA not available. Ensure CUDA toolkit is installed and paths are set.")
            sys.exit(1)
    except ImportError:
        print("Torch not installed correctly.")
        sys.exit(1)

# Run checks
check_cuda_installed()
check_torch_cuda()

# Function to stress CPU: Heavy matrix multiplication on all cores
def cpu_stress_worker():
    while True:
        a = np.random.rand(5000, 5000)
        b = np.random.rand(5000, 5000)
        np.dot(a, b)  # Compute-intensive operation

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
        while True:
            try:
                a = torch.rand(size, size, device=device)
                b = torch.rand(size, size, device=device)
                for _ in range(10):  # Multiple operations per iteration
                    c = torch.mm(a, b)
                torch.cuda.synchronize()  # Ensure completion
                print("Completed GPU stress iteration")  # Diagnostic print
            except Exception as e:
                print(f"GPU stress error: {e}")
                break
    except ImportError as e:
        print(f"Import error in GPU stress: {e}")

# Function to get list of available GPUs
def get_gpu_list():
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
            "HW Throttle": metrics[8],  # Active or Not Active
            "SW Throttle": metrics[9]   # Active or Not Active
        }
    except Exception as e:
        print(f"Error querying nvidia-smi for GPU {gpu_id}: {e}")
    return None

# Function to get CPU metrics
def get_cpu_metrics():
    cpu_percent = psutil.cpu_percent(interval=0.1)
    cpu_freq = psutil.cpu_freq()
    freq_current = cpu_freq.current if cpu_freq else "N/A"
    
    # Get temperature (Linux with lm-sensors; install if needed)
    try:
        temp_output = subprocess.check_output(["sensors"]).decode("utf-8")
        # Parse for CPU package temp (adapt regex/pattern as needed for your sensors output)
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
    # Take two samples 1 second apart to calculate rates
    io1 = psutil.disk_io_counters()
    time.sleep(1)
    io2 = psutil.disk_io_counters()
    
    read_rate = (io2.read_bytes - io1.read_bytes) / (1024 ** 2)  # MB/s
    write_rate = (io2.write_bytes - io1.write_bytes) / (1024 ** 2)  # MB/s
    
    return {
        "Disk Read Rate": f"{read_rate:.2f} MB/s",
        "Disk Write Rate": f"{write_rate:.2f} MB/s"
    }

# Function to print metrics in blue
def print_blue(text):
    print(f"\033[94m{text}\033[0m")

# Function to print in red
def print_red(text):
    print(f"\033[91m{text}\033[0m")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="CPU/GPU Stress Test Script")
    parser.add_argument("--gpu", type=int, default=None, help="GPU number to stress (default: prompt or 0)")
    parser.add_argument("--duration", type=int, default=None, help="Duration in seconds (default: prompt or 30)")
    args = parser.parse_args()
    
    # Set multiprocessing start method to 'spawn' for CUDA compatibility
    mp.set_start_method('spawn')
    
    # List available GPUs
    gpus = get_gpu_list()
    if not gpus:
        print("No NVIDIA GPUs found. GPU stress disabled.")
        gpu_id = None
    else:
        print("Available GPUs:")
        for idx, name in gpus:
            print(f"{idx}: {name}")
        if args.gpu is not None:
            gpu_id = args.gpu
        elif os.isatty(sys.stdin.fileno()):
            gpu_input = input("Enter GPU number to stress (default 0): ").strip()
            gpu_id = int(gpu_input) if gpu_input else 0
        else:
            print("No TTY detected. Using default GPU 0.")
            gpu_id = 0
        # Validate GPU ID
        if gpu_id not in [idx for idx, _ in gpus]:
            print(f"Invalid GPU number {gpu_id}. Exiting.")
            sys.exit(1)
    
    # Prompt for duration
    if args.duration is not None:
        duration = args.duration
    elif os.isatty(sys.stdin.fileno()):
        duration_input = input("Enter number of seconds to run (default 30): ").strip()
        duration = 30 if not duration_input else int(duration_input)
    else:
        print("No TTY detected. Using default duration 30 seconds.")
        duration = 30
    
    print(f"Starting CPU and GPU stress test for {duration} seconds on GPU {gpu_id if gpu_id is not None else 'N/A'}. Press Ctrl+C to stop early.")
    print("Metrics will update every 5 seconds.\n")
    
    # Start CPU stress processes (one per logical core)
    num_cores = mp.cpu_count()
    cpu_processes = [mp.Process(target=cpu_stress_worker) for _ in range(num_cores)]
    for p in cpu_processes:
        p.daemon = True
        p.start()
    
    # Start GPU stress in a separate process if GPU available
    gpu_process = None
    if gpu_id is not None:
        gpu_process = mp.Process(target=gpu_stress, args=(gpu_id,))
        gpu_process.daemon = True
        gpu_process.start()
    
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
            time.sleep(5)  # Update interval
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
        print_blue("Completed Full Load -- CHECK FOR IDLE/NORMALCY")
        if gpu_id is not None:
            gpu_data = get_gpu_metrics(gpu_id)
            if gpu_data:
                print_blue("GPU Metrics:")
                for key, value in gpu_data.items():
                    print_blue(f"{key}: {value}")
        
        cpu_data = get_cpu_metrics()
        print_blue("\nCPU Metrics:")
        for key, value in cpu_data.items():
            print_blue(f"{key}: {value}")
        
        ram_data = get_ram_metrics()
        print_blue("\nRAM Metrics:")
        for key, value in ram_data.items():
            print_blue(f"{key}: {value}")
        
        storage_data = get_storage_metrics()
        print_blue("\nStorage Metrics:")
        for key, value in storage_data.items():
            print_blue(f"{key}: {value}")
        
        # Cleanup prompt
        print("\nyou can cleanup now or cleanup later by running this script again")
        if os.isatty(sys.stdin.fileno()):
            cleanup_input = input("\033[91mCleanup now? (Y/n): \033[0m").strip().lower()
            do_cleanup = cleanup_input == '' or cleanup_input == 'y'
        else:
            print("No TTY detected. Skipping cleanup prompt (use Y/n interactively next time).")
            do_cleanup = False
        
        if do_cleanup:
            # Cleanup venv and Dockerfile
            if os.path.exists('venv'):
                subprocess.call(["rm", "-rf", "venv"])
            if os.path.exists('Dockerfile'):
                os.remove('Dockerfile')
            # Attempt to remove Docker image if exists
            try:
                subprocess.call(["docker", "rmi", "loadup-gpu"])
            except:
                pass
            print("Cleanup completed.")
        else:
            print("Cleanup skipped.")
        
        # Clean up temp script if piped
        if is_piped and 'temp_script' in globals():
            os.remove(temp_script)
        
        sys.exit(0)
