import io
import logging
import platform
import zipfile
import os
import subprocess

try:
    import requests
except ModuleNotFoundError as e:
    # print("Requests module not found:", e)
    subprocess.run(["pip", "install", "requests"])
    import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_file(url, folder, filename):
    """
    Download a file from a URL and save it to a specified folder.

    Args:
    - url (str): URL of the file to download.
    - folder (str): Path to the folder where the file will be saved.
    - filename (str): Name of the file to be saved.

    Returns:
    - str: Full path of the downloaded file, or None if the file already exists.
    """

    # Tạo đường dẫn đầy đủ của tệp
    filepath = os.path.join(folder, filename)

    # Kiểm tra xem tệp đã tồn tại không
    if os.path.exists(filepath):
        print("Tệp đã tồn tại:", filepath)
        return filepath

    # Tạo thư mục nếu nó chưa tồn tại
    if not os.path.exists(folder):
        os.makedirs(folder)

    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(filepath, "wb") as f:
            f.write(response.content)
        logger.info(f"File downloaded and saved to: {filepath}")
        return filepath
    except requests.RequestException as e:
        logger.error(f"Error downloading file: {e}")
        return None


def download_and_extract_zip(url, extract_path="."):
    """
    Download a ZIP file from the given URL and extract its contents.

    Args:
    - url (str): The URL of the ZIP file to download.
    - extract_path (str): The path where the contents of the ZIP file will be extracted. Default is the current directory.

    Returns:
    - bool: True if the download and extraction were successful, False otherwise.
    """
    try:
        # Check if the destination folder exists, if not, create it
        if not os.path.exists(extract_path):
            os.makedirs(extract_path)

        # Get the filename from the URL
        filename = url.split("/")[-1]

        # Check if the file already exists in the destination folder
        if os.path.exists(os.path.join(extract_path, filename)):
            print(f"{filename} already exists. Skipping download.")
            return True

        # Download the ZIP file
        logger.info(f"Downloading {filename}...")
        response = requests.get(url)
        response.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(response.content), "r") as zip_ref:
            zip_ref.extractall(extract_path)
        logger.info(f"{filename} downloaded and extracted successfully.")
        return True
    except (requests.RequestException, zipfile.BadZipFile, Exception) as e:
        logger.error(f"An error occurred: {e}")
        return False


def check_gpu():
    """
    Kiểm tra xem máy có GPU hay không mà không cần PyTorch.
    Hỗ trợ Windows, Linux và macOS.
    """
    try:
        if platform.system() == "Windows":
            output = subprocess.check_output(
                ["wmic", "path", "win32_VideoController", "get", "name"],
                universal_newlines=True,
            )
            gpus = [
                line.strip()
                for line in output.split("\n")
                if line.strip() and "Name" not in line
            ]
        elif platform.system() == "Linux":
            output = subprocess.check_output(["lspci"], universal_newlines=True)
            gpus = [
                line
                for line in output.split("\n")
                if "VGA" in line or "3D controller" in line
            ]
        elif platform.system() == "Darwin":  # macOS
            output = subprocess.check_output(
                ["system_profiler", "SPDisplaysDataType"], universal_newlines=True
            )
            gpus = [
                line.strip() for line in output.split("\n") if "Chipset Model" in line
            ]
        else:
            print("Không xác định được hệ điều hành.")
            return None
        return gpus if gpus else None
    except Exception as e:
        print(f"Lỗi khi kiểm tra GPU: {e}")
        return None


# torch==1.13.1
# torchio==0.18.74
# torchvision==0.14.1
def install_packages():
    print("Install Torch")
    gpus = check_gpu()
    packages = ["torch", "torchvision", "torchaudio"]
    # Only use with Windows
    # if gpus:
    #     print("GPU detected:")
    #     for gpu in gpus:
    #         print(f" - {gpu}")
    #     print("Installing PyTorch with CUDA support...")
    #     subprocess.run(
    #         [
    #             "pip",
    #             "install",
    #             *packages,
    #             "--index-url",
    #             "https://download.pytorch.org/whl/cu121",
    #         ]
    #     )
    # else:
    #     print("No GPU detected, installing CPU version of PyTorch...")
    #     subprocess.run(["pip", "install", "torch==1.13.1", "torchvision==0.18.74", "torchaudio==0.14.1"])
    subprocess.run(["pip", "install", *packages])

    print("Install requirements")
    subprocess.run(["pip", "install", "-r", "requirements.txt"])


def main():
    print("install_packages")
    install_packages()


if __name__ == "__main__":
    main()
