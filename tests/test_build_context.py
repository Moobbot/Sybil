"""What `COPY . .` puts into the image.

docker-compose mounts ./sybil_checkpoints from the host. Once the service has run, that folder holds
the checkpoints (~700 MB); without a rule in .dockerignore the next build copies them into the
image (measured with a 5 MB file), where the mount hides them anyway.

Run: cd Sybil && python -m pytest tests/test_build_context.py -q
"""
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_checkpoints_in_the_host_folder_are_left_out_of_the_image():
    path = os.path.join(ROOT, ".dockerignore")
    if not os.path.exists(path):
        pytest.skip(".dockerignore is not in the image; run this test in the repository")
    with open(path, encoding="utf-8") as f:
        lines = [line.strip() for line in f]
    assert "sybil_checkpoints/*" in lines
    # the folder itself stays in the image: the service expects it to exist
    assert "!sybil_checkpoints/.gitignore" in lines
