#!/usr/bin/env python3
"""
Containerized CPU/GPU stress test.
Version v0.00.2
"""
from __future__ import annotations
 
import argparse
import importlib.util
import logging
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
 
__version__ = "0.00.3"
 
DEFAULT_BASE_IMAGE = "nvidia/cuda:13.0.0-devel-ubuntu22.04"
DEFAULT_IMAGE_TAG = "gpu-stress:local"
CUDA_VERSION_PREFIX = "13.0"
 
DEFAULT_DURATION_SECONDS = 30
DEFAULT_GPU_ID = 0
METRICS_INTERVAL_SECONDS = 5
CPU_MATRIX_SIZE = 5000
GPU_MATRIX_SIZE = 20000
 
VENV_DIR_DEFAULT = "/opt/venv"
 
PROMPT_DONE_ENV = "STRESS_PROMPTS_DONE"
GPU_RAW_ENV = "STRESS_GPU_RAW"
DURATION_RAW_ENV = "STRESS_DURATION_RAW"
CONTAINER_MODE_ENV = "STRESS_CONTAINER_MODE"
 
CUDA_KEYRING_URL = (
    "https://developer.download.nvidia.com/compute/cuda/repos/"
    "ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb"
)
 
REQUIREMENTS_TEXT = (
    "--index-url https://download.pytorch.org/whl/cu130\n"
    "--extra-index-url https://pypi.org/simple\n"
    "numpy\n"
    "psutil\n"
    "torch\n"
)
 
logger = logging.getLogger("gpu_stress")
 
 
@dataclass(frozen=True)
class PromptData:
    """Raw prompt values captured at script start."""
    gpu_raw: str
    duration_raw: str
 
 
@dataclass(frozen=True)
class ContainerEngine:
    """Container engine command wrapper."""
    name: str
    cmd_prefix: List[str]
 
 
def configure_logging() -> None:
    """Configure logging output."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
 
 
def log_version() -> None:
    """Log the script version."""
    logger.info("gpu_stress version v%s", __version__)
 
 
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Containerized CPU/GPU stress test.")
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=None,
        help="GPU index to stress (overrides prompt).",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU stress.",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=None,
        help="Duration in seconds (overrides prompt).",
    )
    parser.add_argument(
        "--engine",
        choices=["auto", "docker", "podman"],
        default="auto",
        help="Container engine to use.",
    )
    parser.add_argument(
        "--base-image",
        default=DEFAULT_BASE_IMAGE,
        help="Base CUDA image for the container.",
    )
    parser.add_argument(
        "--tag",
        default=DEFAULT_IMAGE_TAG,
        help="Image tag to build/run.",
    )
    parser.add_argument(
        "--force-build",
        action="store_true",
        help="Rebuild the container image.",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip building and use an existing image.",
    )
    parser.add_argument("--container", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()
 
 
def collect_prompts(args: argparse.Namespace, container_mode: bool) -> PromptData:
    """Collect user prompts at the start of the script."""
    if container_mode:
        if os.environ.get(PROMPT_DONE_ENV) == "1":
            return PromptData(
                gpu_raw=os.environ.get(GPU_RAW_ENV, ""),
                duration_raw=os.environ.get(DURATION_RAW_ENV, ""),
            )
        raise RuntimeError(
            "Container mode requires prompt values via environment variables."
        )
 
    if args.no_gpu:
        gpu_raw = "none"
    elif args.gpu_id is not None:
        gpu_raw = str(args.gpu_id)
    else:
        gpu_raw = input(
            "Enter GPU number to stress (default 0, or 'none' to disable): "
        ).strip()
 
    if args.duration is not None:
        duration_raw = str(args.duration)
    else:
        duration_raw = input(
            "Enter number of seconds to run (default 30): "
        ).strip()
 
    return PromptData(gpu_raw=gpu_raw, duration_raw=duration_raw)
 
 
def parse_duration(raw: str) -> int:
    """Parse and validate duration input."""
    if not raw:
        return DEFAULT_DURATION_SECONDS
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("Duration must be an integer.") from exc
    if value <= 0:
        raise ValueError("Duration must be a positive integer.")
    return value
 
 
def parse_gpu_request(raw: str) -> Optional[int]:
    """Parse and validate GPU selection input."""
    if not raw:
        return DEFAULT_GPU_ID
    lowered = raw.strip().lower()
    if lowered in {"none", "no", "off", "disable", "cpu"}:
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("GPU id must be an integer or 'none'.") from exc
    if value < 0:
        raise ValueError("GPU id must be >= 0.")
    return value
 
 
def run_command(
    command: Sequence[str],
    capture: bool = False,
    env: Optional[Dict[str, str]] = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run a system command."""
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    stdout = subprocess.PIPE if capture else None
    stderr = subprocess.STDOUT if capture else None
    return subprocess.run(
        list(command),
        check=check,
        text=True,
        stdout=stdout,
        stderr=stderr,
        env=merged_env,
    )
 
 
def is_root() -> bool:
    """Return True if running as root."""
    return hasattr(os, "geteuid") and os.geteuid() == 0
 
 
def can_sudo_no_prompt() -> bool:
    """Return True if sudo works without prompting."""
    if shutil.which("sudo") is None:
        return False
    result = subprocess.run(
        ["sudo", "-n", "true"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0
 
 
def get_privilege_prefix(allow_sudo: bool) -> List[str]:
    """Return command prefix for privileged operations."""
    if is_root():
        return []
    if allow_sudo and can_sudo_no_prompt():
        return ["sudo", "-n"]
    raise RuntimeError("Root or passwordless sudo is required for installation.")
 
 
def apt_install(packages: Sequence[str], allow_sudo: bool) -> None:
    """Install apt packages quietly."""
    if not packages:
        return
    if shutil.which("apt-get") is None:
        raise RuntimeError("apt-get not available for installation.")
    prefix = get_privilege_prefix(allow_sudo)
    env = {"DEBIAN_FRONTEND": "noninteractive"}
    run_command(prefix + ["apt-get", "update", "-qq"], env=env)
    run_command(
        prefix
        + [
            "apt-get",
            "install",
            "-y",
            "-qq",
            "--no-install-recommends",
        ]
        + list(packages),
        env=env,
    )
 
 
def maybe_install_package(package: str, friendly_name: str) -> None:
    """Attempt to install a host package with apt-get."""
    try:
        apt_install([package], allow_sudo=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"{friendly_name} is required. Install with: sudo apt-get install -y {package}"
        ) from exc
 
 
def is_wsl() -> bool:
    """Return True if running under WSL."""
    return os.path.exists("/proc/sys/fs/binfmt_misc/WSLInterop")
 
 
def nested_podman_available() -> bool:
    """Return True if nested Podman is reachable via docker exec."""
    try:
        run_command(["docker", "exec", "podman", "podman", "-v"], capture=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
 
 
def ensure_docker_engine() -> ContainerEngine:
    """Ensure Docker is available and return engine wrapper."""
    if shutil.which("docker") is None:
        logger.info("Docker not found; attempting installation...")
        try:
            maybe_install_package("docker.io", "Docker")
        except RuntimeError as exc:
            logger.error("%s", exc)
    if shutil.which("docker") is None:
        raise RuntimeError(
            "Docker is required. Install with: sudo apt-get install -y docker.io"
        )
    return ContainerEngine(name="docker", cmd_prefix=["docker"])
 
 
def ensure_podman_engine() -> ContainerEngine:
    """Ensure Podman is available and return engine wrapper."""
    if is_wsl():
        if shutil.which("podman") is None:
            logger.info("Podman not found; attempting installation...")
            try:
                maybe_install_package("podman", "Podman")
            except RuntimeError as exc:
                logger.error("%s", exc)
        if shutil.which("podman") is None:
            raise RuntimeError(
                "Podman is required on WSL. Install with: sudo apt-get install -y podman"
            )
        return ContainerEngine(name="podman", cmd_prefix=["podman"])
 
    # Native Linux Podman uses a nested container, per requirements.
    if shutil.which("docker") is None:
        raise RuntimeError(
            "Podman on native Linux requires Docker and a running 'podman' container."
        )
    if not nested_podman_available():
        raise RuntimeError(
            "Nested Podman not available. Start a container named 'podman' "
            "or install Docker and use --engine docker."
        )
    return ContainerEngine(
        name="podman",
        cmd_prefix=["docker", "exec", "podman", "podman"],
    )
 
 
def detect_engine(preferred: str) -> ContainerEngine:
    """Select container engine based on preference."""
    if preferred == "docker":
        return ensure_docker_engine()
    if preferred == "podman":
        return ensure_podman_engine()
    try:
        return ensure_docker_engine()
    except RuntimeError as exc:
        logger.warning("Docker unavailable: %s", exc)
        return ensure_podman_engine()
 
 
def engine_run(
    engine: ContainerEngine,
    args: Sequence[str],
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run a container engine command."""
    return run_command(engine.cmd_prefix + list(args), capture=capture)
 
 
def image_exists(engine: ContainerEngine, tag: str) -> bool:
    """Return True if the image tag exists."""
    try:
        engine_run(engine, ["image", "inspect", tag], capture=True)
        return True
    except subprocess.CalledProcessError:
        return False
 
 
def render_dockerfile(base_image: str) -> str:
    """Return Dockerfile contents."""
    return (
        f"FROM {base_image}\n"
        "ENV DEBIAN_FRONTEND=noninteractive\n"
        "ENV PYTHONUNBUFFERED=1\n"
        "RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends "
        "python3 python3-venv python3-pip ca-certificates lm-sensors && "
        "rm -rf /var/lib/apt/lists/*\n"
        "WORKDIR /app\n"
        "COPY requirements.txt /app/requirements.txt\n"
        "RUN pip3 install --no-cache-dir --quiet --upgrade pip setuptools wheel && "
        "pip3 install --no-cache-dir --quiet -r /app/requirements.txt\n"
        "COPY gpu_stress.py /app/gpu_stress.py\n"
        "RUN chmod +x /app/gpu_stress.py\n"
        "ENTRYPOINT [\"python3\", \"/app/gpu_stress.py\", \"--container\"]\n"
    )
 
 
def build_image(
    engine: ContainerEngine,
    base_image: str,
    tag: str,
    script_path: Path,
) -> None:
    """Build the container image from a temporary context."""
    if not script_path.exists():
        raise RuntimeError(f"Script path not found: {script_path}")
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        dockerfile_path = temp_path / "Dockerfile"
        requirements_path = temp_path / "requirements.txt"
        script_dest = temp_path / "gpu_stress.py"
 
        dockerfile_path.write_text(render_dockerfile(base_image), encoding="utf-8")
        requirements_path.write_text(REQUIREMENTS_TEXT, encoding="utf-8")
        shutil.copy2(script_path, script_dest)
        script_dest.chmod(0o755)
 
        # For nested Podman, ensure the context is accessible to that container.
        engine_run(engine, ["build", "-t", tag, str(temp_path)])
 
 
def run_container(
    engine: ContainerEngine,
    tag: str,
    env_vars: Dict[str, str],
    gpu_enabled: bool,
) -> None:
    """Run the container image with the provided environment."""
    cmd: List[str] = ["run", "--rm"]
    if gpu_enabled:
        if engine.name == "docker":
            cmd += ["--gpus", "all"]
        else:
            # Podman GPU support relies on NVIDIA OCI hooks.
            cmd += ["--hooks-dir=/usr/share/containers/oci/hooks.d"]
    for key, value in env_vars.items():
        cmd += ["-e", f"{key}={value}"]
    cmd.append(tag)
    engine_run(engine, cmd)
 
 
def install_cuda_toolkit() -> None:
    """Install CUDA toolkit 13.0 via NVIDIA keyring."""
    apt_install(["wget", "gnupg", "ca-certificates"], allow_sudo=True)
    prefix = get_privilege_prefix(allow_sudo=True)
    env = {"DEBIAN_FRONTEND": "noninteractive"}
    run_command(["wget", "-q", "-O", "/tmp/cuda-keyring.deb", CUDA_KEYRING_URL])
    run_command(prefix + ["dpkg", "-i", "/tmp/cuda-keyring.deb"], env=env)
    run_command(prefix + ["apt-get", "update", "-qq"], env=env)
    run_command(
        prefix
        + [
            "apt-get",
            "install",
            "-y",
            "-qq",
            "--no-install-recommends",
            "cuda-toolkit-13-0",
        ],
        env=env,
    )
 
 
def ensure_nvcc() -> None:
    """Ensure CUDA toolkit nvcc is available and correct version."""
    if shutil.which("nvcc"):
        output = run_command(["nvcc", "--version"], capture=True).stdout or ""
        if f"release {CUDA_VERSION_PREFIX}" in output:
            return
        raise RuntimeError(
            f"CUDA version mismatch. Expected {CUDA_VERSION_PREFIX}."
        )
 
    logger.info("CUDA toolkit not found; installing CUDA %s...", CUDA_VERSION_PREFIX)
    install_cuda_toolkit()
 
    if shutil.which("nvcc") is None:
        raise RuntimeError("nvcc still not found after installation.")
 
 
def ensure_nvidia_smi() -> None:
    """Ensure nvidia-smi is available when GPU is requested."""
    if shutil.which("nvidia-smi") is None:
        raise RuntimeError(
            "nvidia-smi not found. Ensure the NVIDIA driver is installed on the host "
            "and run the container with GPU access."
        )
 
 
def ensure_lm_sensors() -> bool:
    """Ensure lm-sensors is installed for CPU temperature metrics."""
    if shutil.which("sensors") is not None:
        return True
    try:
        apt_install(["lm-sensors"], allow_sudo=True)
    except RuntimeError as exc:
        logger.warning("lm-sensors not installed: %s", exc)
        return False
    return shutil.which("sensors") is not None
 
 
def ensure_container_system_deps(require_gpu: bool) -> bool:
    """Ensure system dependencies inside the container."""
    sensors_available = ensure_lm_sensors()
    if require_gpu:
        ensure_nvcc()
        ensure_nvidia_smi()
    return sensors_available
 
 
def get_venv_dir() -> Path:
    """Return a writable virtual environment directory."""
    base = Path(VENV_DIR_DEFAULT)
    if base.parent.exists() and os.access(base.parent, os.W_OK):
        return base
    return Path.home() / ".gpu-stress-venv"
 
 
def install_requirements(pip_exe: str, requirements_text: str) -> None:
    """Install Python requirements using pip."""
    with tempfile.NamedTemporaryFile("w", delete=False) as handle:
        handle.write(requirements_text)
        req_path = handle.name
    try:
        run_command(
            [
                pip_exe,
                "install",
                "--quiet",
                "--no-cache-dir",
                "--upgrade",
                "pip",
                "setuptools",
                "wheel",
            ]
        )
        run_command(
            [pip_exe, "install", "--quiet", "--no-cache-dir", "-r", req_path]
        )
    finally:
        try:
            os.unlink(req_path)
        except OSError:
            pass
 
 
def ensure_virtualenv() -> None:
    """Ensure a virtual environment is active."""
    if sys.prefix != sys.base_prefix:
        return
 
    venv_dir = get_venv_dir()
    venv_python = venv_dir / "bin" / "python"
    venv_pip = venv_dir / "bin" / "pip"
 
    if not venv_dir.exists():
        try:
            run_command([sys.executable, "-m", "venv", str(venv_dir)])
        except subprocess.CalledProcessError:
            logger.info("python3-venv missing; installing...")
            apt_install(["python3-venv"], allow_sudo=True)
            run_command([sys.executable, "-m", "venv", str(venv_dir)])
 
    if not venv_pip.exists():
        raise RuntimeError("pip not found in the virtual environment.")
 
    install_requirements(str(venv_pip), REQUIREMENTS_TEXT)
    os.execv(str(venv_python), [str(venv_python)] + sys.argv)
 
 
def ensure_python_deps() -> None:
    """Ensure numpy, psutil, and torch are installed."""
    missing = [
        pkg
        for pkg in ("numpy", "psutil", "torch")
        if importlib.util.find_spec(pkg) is None
    ]
    if not missing:
        return
    pip_path = Path(sys.executable).with_name("pip")
    pip_cmd = str(pip_path) if pip_path.exists() else shutil.which("pip")
    if not pip_cmd:
        raise RuntimeError("pip not available. Install python3-pip.")
    logger.info("Installing missing Python packages: %s", ", ".join(missing))
    install_requirements(pip_cmd, REQUIREMENTS_TEXT)
 
 
def check_torch_cuda(require_gpu: bool) -> None:
    """Validate torch and CUDA availability."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Torch not installed correctly.") from exc
 
    logger.info("Torch version: %s", torch.__version__)
    cuda_version = torch.version.cuda or "unknown"
    logger.info("CUDA version in Torch: %s", cuda_version)
 
    if require_gpu:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "PyTorch CUDA not available. Ensure CUDA toolkit and drivers are installed."
            )
        if not cuda_version.startswith(CUDA_VERSION_PREFIX):
            raise RuntimeError(
                f"CUDA version mismatch. Expected {CUDA_VERSION_PREFIX}."
            )
 
 
def get_gpu_list() -> List[Tuple[int, str]]:
    """Return list of available GPUs."""
    try:
        output = run_command(["nvidia-smi", "-L"], capture=True).stdout or ""
    except Exception as exc:
        logger.error("Error listing GPUs: %s", exc)
        return []
 
    gpus: List[Tuple[int, str]] = []
    for line in output.strip().splitlines():
        if line.startswith("GPU "):
            parts = line.split(": ", 1)
            idx = int(parts[0].split(" ")[1])
            name = parts[1].split(" (")[0].strip()
            gpus.append((idx, name))
    return gpus
 
 
def get_gpu_metrics(gpu_id: int) -> Optional[Dict[str, str]]:
    """Collect GPU metrics using nvidia-smi."""
    command = [
        "nvidia-smi",
        "-i",
        str(gpu_id),
        "--query-gpu=fan.speed,pstate,power.draw,power.limit,memory.used,"
        "memory.total,utilization.gpu,temperature.gpu,"
        "clocks_throttle_reasons.hw_thermal_slowdown,"
        "clocks_throttle_reasons.sw_thermal_slowdown",
        "--format=csv,noheader",
    ]
    try:
        output = run_command(command, capture=True).stdout or ""
        metrics = [item.strip() for item in output.split(",")]
        if len(metrics) < 10:
            raise RuntimeError("Unexpected nvidia-smi output.")
        return {
            "Fan Speed": metrics[0],
            "Perf (P-State)": metrics[1],
            "Power Usage/Cap": f"{metrics[2]}/{metrics[3]}",
            "VRAM Usage/Cap": f"{metrics[4]}/{metrics[5]}",
            "GPU Utilization": metrics[6],
            "GPU Temperature": metrics[7],
            "HW Throttle": metrics[8],
            "SW Throttle": metrics[9],
        }
    except Exception as exc:
        logger.error("Error querying nvidia-smi for GPU %s: %s", gpu_id, exc)
    return None
 
 
def get_cpu_metrics(sensors_available: bool) -> Dict[str, str]:
    """Return CPU utilization, frequency, and temperature metrics."""
    import psutil
 
    cpu_percent = psutil.cpu_percent(interval=0.1)
    cpu_freq = psutil.cpu_freq()
    freq_current = f"{cpu_freq.current:.0f} MHz" if cpu_freq else "N/A"
 
    temp = "N/A"
    if sensors_available:
        try:
            temp_output = run_command(["sensors"], capture=True).stdout or ""
            clean_output = temp_output.encode("ascii", "ignore").decode()
            match = re.search(r"Package id 0:\s+\+([\d.]+)", clean_output)
            if match:
                temp = f"{match.group(1)}C"
        except Exception:
            temp = "N/A"
    else:
        temp = "N/A (lm-sensors not installed)"
 
    return {
        "CPU Utilization": f"{cpu_percent}%",
        "CPU Frequency": freq_current,
        "CPU Temperature": temp,
    }
 
 
def get_ram_metrics() -> Dict[str, str]:
    """Return RAM usage metrics."""
    import psutil
 
    mem = psutil.virtual_memory()
    return {
        "RAM Usage/Cap": (
            f"{mem.used / (1024**2):.1f} MiB / {mem.total / (1024**2):.1f} MiB"
        ),
        "RAM Utilization": f"{mem.percent}%",
    }
 
 
def get_storage_metrics() -> Dict[str, str]:
    """Return storage I/O rates."""
    import psutil
 
    io1 = psutil.disk_io_counters()
    if io1 is None:
        return {"Disk Read Rate": "N/A", "Disk Write Rate": "N/A"}
 
    time.sleep(1)
    io2 = psutil.disk_io_counters()
    if io2 is None:
        return {"Disk Read Rate": "N/A", "Disk Write Rate": "N/A"}
 
    read_rate = (io2.read_bytes - io1.read_bytes) / (1024 ** 2)
    write_rate = (io2.write_bytes - io1.write_bytes) / (1024 ** 2)
 
    return {
        "Disk Read Rate": f"{read_rate:.2f} MB/s",
        "Disk Write Rate": f"{write_rate:.2f} MB/s",
    }
 
 
def cpu_stress_worker(matrix_size: int) -> None:
    """Continuously stress the CPU with matrix multiplication."""
    import numpy as np
 
    while True:
        a = np.random.rand(matrix_size, matrix_size)
        b = np.random.rand(matrix_size, matrix_size)
        np.dot(a, b)
 
 
def gpu_stress(gpu_id: int, matrix_size: int) -> None:
    """Continuously stress the GPU with tensor operations."""
    configure_logging()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
 
    try:
        import torch
    except ImportError as exc:
        logger.error("Torch import error in GPU stress: %s", exc)
        return
 
    if not torch.cuda.is_available():
        logger.error("CUDA not available on GPU %s. GPU stress disabled.", gpu_id)
        return
 
    torch.cuda.set_device(0)
    device = torch.device("cuda")
    logger.info("Using device: %s", torch.cuda.get_device_name(device))
 
    while True:
        try:
            a = torch.rand(matrix_size, matrix_size, device=device)
            b = torch.rand(matrix_size, matrix_size, device=device)
            for _ in range(10):
                torch.mm(a, b)
            torch.cuda.synchronize()
            logger.info("Completed GPU stress iteration")
        except Exception as exc:
            logger.error("GPU stress error: %s", exc)
            break
 
 
def log_blue(text: str) -> None:
    """Log a line in blue."""
    logger.info("\033[94m%s\033[0m", text)
 
 
def run_stress_test(
    duration: int,
    gpu_id: Optional[int],
    sensors_available: bool,
) -> int:
    """Run the stress test and report metrics."""
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
 
    logger.info(
        "Starting CPU and GPU stress test for %s seconds on GPU %s.",
        duration,
        gpu_id if gpu_id is not None else "N/A",
    )
    logger.info("Metrics will update every %s seconds.", METRICS_INTERVAL_SECONDS)
 
    num_cores = mp.cpu_count()
    cpu_processes: List[mp.Process] = []
    for _ in range(num_cores):
        proc = mp.Process(target=cpu_stress_worker, args=(CPU_MATRIX_SIZE,))
        proc.daemon = True
        proc.start()
        cpu_processes.append(proc)
 
    gpu_process = None
    if gpu_id is not None:
        gpu_process = mp.Process(target=gpu_stress, args=(gpu_id, GPU_MATRIX_SIZE))
        gpu_process.daemon = True
        gpu_process.start()
 
    start_time = time.time()
    try:
        while time.time() - start_time < duration:
            if gpu_id is not None:
                gpu_data = get_gpu_metrics(gpu_id)
                if gpu_data:
                    logger.info("GPU Metrics:")
                    for key, value in gpu_data.items():
                        logger.info("  %s: %s", key, value)
 
            cpu_data = get_cpu_metrics(sensors_available)
            logger.info("CPU Metrics:")
            for key, value in cpu_data.items():
                logger.info("  %s: %s", key, value)
 
            logger.info("-" * 40)
            time.sleep(METRICS_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        logger.info("Stopping stress test early.")
    finally:
        logger.info("Stress test completed.")
        for proc in cpu_processes:
            proc.terminate()
        if gpu_process is not None:
            gpu_process.terminate()
 
        for proc in cpu_processes:
            proc.join(timeout=2)
        if gpu_process is not None:
            gpu_process.join(timeout=2)
 
        time.sleep(2)
 
        log_blue("-" * 40)
        log_blue("Completed Full Load")
        if gpu_id is not None:
            gpu_data = get_gpu_metrics(gpu_id)
            if gpu_data:
                log_blue("GPU Metrics:")
                for key, value in gpu_data.items():
                    log_blue(f"{key}: {value}")
 
        cpu_data = get_cpu_metrics(sensors_available)
        log_blue("CPU Metrics:")
        for key, value in cpu_data.items():
            log_blue(f"{key}: {value}")
 
        ram_data = get_ram_metrics()
        log_blue("RAM Metrics:")
        for key, value in ram_data.items():
            log_blue(f"{key}: {value}")
 
        storage_data = get_storage_metrics()
        log_blue("Storage Metrics:")
        for key, value in storage_data.items():
            log_blue(f"{key}: {value}")
 
    return 0
 
 
def host_main(prompts: PromptData, args: argparse.Namespace) -> int:
    """Build and run the container from the host."""
    try:
        duration = parse_duration(prompts.duration_raw)
        gpu_request = None if args.no_gpu else parse_gpu_request(prompts.gpu_raw)
    except ValueError as exc:
        logger.error("%s", exc)
        return 1
 
    try:
        engine = detect_engine(args.engine)
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 1
 
    if args.skip_build:
        if not image_exists(engine, args.tag):
            logger.error("Image '%s' not found. Run without --skip-build.", args.tag)
            return 1
    else:
        if args.force_build or not image_exists(engine, args.tag):
            logger.info("Building container image '%s'...", args.tag)
            try:
                build_image(engine, args.base_image, args.tag, Path(__file__).resolve())
            except (RuntimeError, subprocess.CalledProcessError) as exc:
                logger.error("Image build failed: %s", exc)
                return 1
 
    env_vars = {
        PROMPT_DONE_ENV: "1",
        GPU_RAW_ENV: "none" if gpu_request is None else str(gpu_request),
        DURATION_RAW_ENV: str(duration),
        CONTAINER_MODE_ENV: "1",
    }
 
    try:
        run_container(engine, args.tag, env_vars, gpu_request is not None)
    except subprocess.CalledProcessError as exc:
        logger.error("Container run failed: %s", exc)
        return 1
    return 0
 
 
def container_main(prompts: PromptData, args: argparse.Namespace) -> int:
    """Run the stress workload inside the container."""
    try:
        duration = parse_duration(prompts.duration_raw)
        gpu_request = None if args.no_gpu else parse_gpu_request(prompts.gpu_raw)
    except ValueError as exc:
        logger.error("%s", exc)
        return 1
 
    require_gpu = gpu_request is not None
    try:
        sensors_available = ensure_container_system_deps(require_gpu)
        ensure_virtualenv()
        ensure_python_deps()
        check_torch_cuda(require_gpu)
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        logger.error("%s", exc)
        return 1
 
    gpu_id: Optional[int] = None
    if require_gpu:
        gpus = get_gpu_list()
        if not gpus:
            logger.warning("No NVIDIA GPUs found. GPU stress disabled.")
            gpu_id = None
        else:
            logger.info("Available GPUs:")
            for idx, name in gpus:
                logger.info("  %s: %s", idx, name)
            if gpu_request not in [idx for idx, _ in gpus]:
                logger.error("Invalid GPU number %s. Exiting.", gpu_request)
                return 1
            gpu_id = gpu_request
 
    return run_stress_test(duration, gpu_id, sensors_available)
 
 
def main() -> int:
    """Script entry point."""
    args = parse_args()
    configure_logging()
    container_mode = args.container or os.environ.get(CONTAINER_MODE_ENV) == "1"
 
    try:
        prompts = collect_prompts(args, container_mode)
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 1
 
    log_version()
 
    if container_mode:
        return container_main(prompts, args)
 
    return host_main(prompts, args)
 
 
if __name__ == "__main__":
    sys.exit(main())
