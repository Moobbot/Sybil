import io
import logging
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


# torch==1.13.1
# torchio==0.18.74
# torchvision==0.14.1
def install_packages():
    print("Install Torch GPU")
    packages = ["torch", "torchvision", "torchaudio"]
    subprocess.run(
        [
            "pip",
            "install",
            *packages,
            "--index-url",
            "https://download.pytorch.org/whl/cu121",
        ]
    )
    print("Install requirements")
    subprocess.run(["pip", "install", "-r", "requirements.txt"])


def main():
    print("install_packages")
    install_packages()

if __name__ == "__main__":
    main()