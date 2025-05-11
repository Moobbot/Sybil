#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Setup script for Sybil.
This script installs required packages and downloads model checkpoints.
"""

import argparse
import logging
import os
import platform
import re
import subprocess
import sys
import zipfile
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Minimum Python version required
MIN_PYTHON_VERSION = (3, 7)

# PyTorch versions
PYTORCH_VERSIONS = {
    "cpu": ["torch<2.0.0", "torchvision<2.0.0", "torchaudio<2.0.0"],
    "cuda": ["torch", "torchvision", "torchaudio"],
}

# CUDA versions
CUDA_VERSIONS = {
    "11.8": "cu118",
    "12.1": "cu121",
}

# Default CUDA version
DEFAULT_CUDA_VERSION = "12.1"


def check_python_version() -> bool:
    """
    Check if the current Python version meets the minimum requirements.

    Returns:
        bool: True if the Python version is sufficient, False otherwise.
    """
    current_version = sys.version_info[:2]
    if current_version < MIN_PYTHON_VERSION:
        logger.error(
            f"Python {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]} or higher is required. "
            f"You are using Python {current_version[0]}.{current_version[1]}."
        )
        return False
    return True


def download_file(url: str, destination: str, show_progress: bool = True) -> bool:
    """
    Download a file from a URL with progress reporting.

    Args:
        url: URL of the file to download
        destination: Path where the file will be saved
        show_progress: Whether to show download progress

    Returns:
        bool: True if download was successful, False otherwise
    """
    try:
        # Install requests if not already installed
        try:
            import requests
        except ImportError:
            logger.info("Installing requests package...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "requests"], check=True
            )
            import requests

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(destination)), exist_ok=True)

        # Check if file already exists
        if os.path.exists(destination):
            logger.info(f"File already exists: {destination}")
            return True

        # Download the file with a session to handle redirects
        logger.info(f"Downloading file from: {url}")
        session = requests.Session()
        response = session.get(url, stream=True)
        response.raise_for_status()

        # Get total file size if available
        total_size = int(response.headers.get("content-length", 0))

        # Save the file with progress reporting
        with open(destination, "wb") as f:
            if total_size > 0 and show_progress:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        # Print progress every 5%
                        if downloaded % (total_size // 20) < 8192:
                            percent = (downloaded / total_size) * 100
                            logger.info(f"Download progress: {percent:.1f}%")
            else:
                # If content-length is not available or progress not needed
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

        logger.info(f"Successfully downloaded file: {destination}")
        return True

    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        return False


def download_and_extract_zip(url: str, extract_path: str = ".") -> bool:
    """
    Download a ZIP file from the given URL and extract its contents.

    Args:
        url: The URL of the ZIP file to download
        extract_path: The path where the contents of the ZIP file will be extracted

    Returns:
        bool: True if the download and extraction were successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(extract_path, exist_ok=True)

        # Get the filename from the URL
        filename = url.split("/")[-1]
        temp_file = os.path.join(extract_path, f"temp_{filename}")

        # Download the ZIP file
        if not download_file(url, temp_file):
            return False

        # Extract the ZIP file
        logger.info(f"Extracting {filename}...")
        with zipfile.ZipFile(temp_file, "r") as zip_ref:
            zip_ref.extractall(extract_path)

        # Remove the temporary file
        os.remove(temp_file)

        logger.info(f"{filename} extracted successfully to {extract_path}")
        return True

    except Exception as e:
        logger.error(f"Error downloading or extracting ZIP file: {e}")
        return False


def check_gpu():
    """
    Check for GPU hardware without requiring PyTorch.
    Supports Windows, Linux, macOS and Docker environments.

    Returns:
        list: List of detected GPUs or None if none found.
    """
    try:
        # Check if running in Docker
        in_docker = (
            os.path.exists("/.dockerenv")
            or os.environ.get("DOCKER_CONTAINER") == "true"
        )
        if in_docker:
            logger.info("Running in Docker container")
            gpus = ["GPU in Docker"]
            return None

        logger.info(
            f"Checking GPU in environment: {platform.system()}"
            + (" (Docker)" if in_docker else "")
        )

        # === LINUX and DOCKER ===
        if platform.system() == "Linux" or in_docker:
            gpus = []
            logger.info("Checking GPU using Linux methods...")

            # Method 1: Check via nvidia-smi (best for Docker with NVIDIA GPU)
            try:
                logger.debug("Trying nvidia-smi method...")
                output = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                    universal_newlines=True,
                    stderr=subprocess.DEVNULL,
                )
                nvidia_gpus = [
                    line.strip() for line in output.split("\n") if line.strip()
                ]
                if nvidia_gpus:
                    logger.info(f"Found {len(nvidia_gpus)} GPUs via nvidia-smi")
                    return nvidia_gpus
            except (subprocess.SubprocessError, FileNotFoundError):
                logger.debug("nvidia-smi method failed")
                pass

            # Method 2: Check via lspci
            try:
                logger.debug("Trying lspci method...")
                output = subprocess.check_output(
                    ["lspci"], universal_newlines=True, stderr=subprocess.DEVNULL
                )
                gpu_lines = [
                    line
                    for line in output.split("\n")
                    if "VGA" in line or "3D controller" in line
                ]
                if gpu_lines:
                    logger.info(f"Found {len(gpu_lines)} GPUs via lspci")
                    return gpu_lines
            except (subprocess.SubprocessError, FileNotFoundError):
                logger.debug("lspci method failed")
                pass

            # Method 3: Check /proc/driver/nvidia directory
            if os.path.exists("/proc/driver/nvidia/gpus"):
                try:
                    logger.debug("Checking /proc/driver/nvidia/gpus...")
                    gpu_dirs = os.listdir("/proc/driver/nvidia/gpus")
                    if gpu_dirs:
                        logger.info(
                            f"Found {len(gpu_dirs)} GPUs via /proc/driver/nvidia/gpus"
                        )
                        return [f"NVIDIA GPU #{i}" for i in range(len(gpu_dirs))]
                except Exception:
                    logger.debug("/proc/driver/nvidia/gpus method failed")
                    pass

            # Method 4: Check via /dev/nvidia*
            try:
                logger.debug("Checking /dev/nvidia* devices...")
                nvidia_devices = [
                    dev
                    for dev in os.listdir("/dev")
                    if dev.startswith("nvidia")
                    and dev != "nvidiactl"
                    and dev != "nvidia-modeset"
                ]
                if nvidia_devices:
                    logger.info(
                        f"Found {len(nvidia_devices)} GPUs via /dev/nvidia* devices"
                    )
                    return [f"NVIDIA GPU device: {dev}" for dev in nvidia_devices]
            except (FileNotFoundError, PermissionError):
                logger.debug("/dev/nvidia* method failed")
                pass

        # === WINDOWS ===
        elif platform.system() == "Windows":
            logger.info("Checking GPU using Windows methods...")
            # Method 1: Using WMIC
            try:
                logger.debug("Trying WMIC method...")
                output = subprocess.check_output(
                    ["wmic", "path", "win32_VideoController", "get", "name"],
                    universal_newlines=True,
                    stderr=subprocess.STDOUT,
                )
                gpus = [
                    line.strip()
                    for line in output.split("\n")
                    if line.strip() and "Name" not in line
                ]
                if gpus:
                    logger.info(f"Found {len(gpus)} GPUs via WMIC")
                    return gpus
            except (subprocess.SubprocessError, FileNotFoundError):
                logger.debug("WMIC method failed")
                pass

            # Method 2: Using PowerShell if WMIC fails
            try:
                logger.debug("Trying PowerShell method...")
                output = subprocess.check_output(
                    [
                        "powershell",
                        "Get-WmiObject Win32_VideoController | Select-Object Name",
                    ],
                    universal_newlines=True,
                    stderr=subprocess.STDOUT,
                )
                gpus = [
                    line.strip()
                    for line in output.split("\n")
                    if line.strip() and "Name" not in line and "----" not in line
                ]
                if gpus:
                    logger.info(f"Found {len(gpus)} GPUs via PowerShell")
                    return gpus
            except (subprocess.SubprocessError, FileNotFoundError):
                logger.debug("PowerShell method failed")
                pass

        # === macOS ===
        elif platform.system() == "Darwin":
            logger.info("Checking GPU using macOS methods...")
            try:
                logger.debug("Trying system_profiler method...")
                output = subprocess.check_output(
                    ["system_profiler", "SPDisplaysDataType"],
                    universal_newlines=True,
                    stderr=subprocess.STDOUT,
                )
                # Find lines with "Chipset Model" and get GPU name
                gpu_pattern = re.compile(r"Chipset Model: (.+)")
                matches = gpu_pattern.findall(output)
                if matches:
                    logger.info(
                        f"Found {len(matches)} GPUs via system_profiler (regex)"
                    )
                    return [f"Chipset Model: {match}" for match in matches]

                # Alternative method if regex doesn't work
                gpus = [
                    line.strip()
                    for line in output.split("\n")
                    if "Chipset Model" in line
                ]
                if gpus:
                    logger.info(
                        f"Found {len(gpus)} GPUs via system_profiler (line search)"
                    )
                    return gpus
            except (subprocess.SubprocessError, FileNotFoundError):
                logger.debug("system_profiler method failed")
                pass

        # === Final method: Check environment variables ===
        logger.info("Checking GPU using environment variables...")
        # Check CUDA_VISIBLE_DEVICES environment variable
        cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_devices and cuda_devices != "-1":
            logger.info(f"Found GPUs via CUDA_VISIBLE_DEVICES: {cuda_devices}")
            return [f"CUDA Device #{dev}" for dev in cuda_devices.split(",")]

        # Check GPU_DEVICE_ORDINAL environment variable (for ROCm/AMD)
        rocm_devices = os.environ.get("GPU_DEVICE_ORDINAL")
        if rocm_devices:
            logger.info(f"Found GPUs via GPU_DEVICE_ORDINAL: {rocm_devices}")
            return [f"ROCm Device #{dev}" for dev in rocm_devices.split(",")]

        logger.warning("No GPU detected or unsupported operating system.")
        return None

    except Exception as e:
        logger.error(f"Error checking GPU: {e}")
        return None


def install_packages(
    cuda_version: str = DEFAULT_CUDA_VERSION, force_cpu: bool = False
) -> bool:
    """
    Install required packages including PyTorch.

    Args:
        cuda_version: CUDA version to use (e.g., '11.8', '12.1')
        force_cpu: Force CPU installation even if GPU is detected

    Returns:
        bool: True if installation was successful, False otherwise
    """
    try:
        # Install requests if not already installed
        try:
            import requests
        except ImportError:
            logger.info("Installing requests package...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "requests"], check=True
            )

        # Check for GPU
        gpus = None if force_cpu else check_gpu()

        # Install PyTorch
        logger.info("Installing PyTorch and related packages...")
        if gpus:
            logger.info("GPU detected:")
            for gpu in gpus:
                logger.info(f" - {gpu}")

            # Validate CUDA version
            if cuda_version not in CUDA_VERSIONS:
                logger.warning(
                    f"Unsupported CUDA version: {cuda_version}. Using default: {DEFAULT_CUDA_VERSION}"
                )
                cuda_version = DEFAULT_CUDA_VERSION

            cuda_suffix = CUDA_VERSIONS[cuda_version]
            logger.info(f"Installing PyTorch with CUDA {cuda_version} support...")

            # Install PyTorch with CUDA support
            cmd = [
                sys.executable,
                "-m",
                "pip",
                "install",
                *PYTORCH_VERSIONS["cuda"],
                "--index-url",
                f"https://download.pytorch.org/whl/{cuda_suffix}",
            ]
            result = subprocess.run(cmd, check=True)

            if result.returncode != 0:
                logger.error(
                    "Failed to install PyTorch with CUDA support. Falling back to CPU version."
                )
                cmd = [sys.executable, "-m", "pip", "install", *PYTORCH_VERSIONS["cpu"]]
                subprocess.run(cmd, check=True)
        else:
            logger.info(
                "No GPU detected or CPU version forced. Installing CPU version of PyTorch..."
            )
            cmd = [sys.executable, "-m", "pip", "install", *PYTORCH_VERSIONS["cpu"]]
            subprocess.run(cmd, check=True)

        # Install other requirements
        logger.info("Installing other requirements...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            check=True,
        )

        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"Error installing packages: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during package installation: {e}")
        return False


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Setup script for Sybil")

    parser.add_argument(
        "--skip-packages", action="store_true", help="Skip package installation"
    )

    parser.add_argument(
        "--cuda-version",
        type=str,
        choices=list(CUDA_VERSIONS.keys()),
        default=DEFAULT_CUDA_VERSION,
        help=f"CUDA version to use (default: {DEFAULT_CUDA_VERSION})",
    )

    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Force CPU installation even if GPU is detected",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    return parser.parse_args()


def main() -> int:
    """
    Main function to run the setup process.

    Returns:
        int: Exit code (0 for success, non-zero for failure)
    """
    # Parse command line arguments
    args = parse_arguments()

    # Set logging level based on verbosity
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Print welcome message
    logger.info("Starting Sybil setup...")

    # Check Python version
    if not check_python_version():
        return 1

    # Install packages
    if not args.skip_packages:
        logger.info("Installing packages...")
        if not install_packages(args.cuda_version, args.force_cpu):
            logger.error("Package installation failed.")
            return 1
    else:
        logger.info("Skipping package installation as requested.")

    logger.info("Setup completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
