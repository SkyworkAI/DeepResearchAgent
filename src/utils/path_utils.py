import os
from pathlib import Path


def get_project_root():
    root = str(Path(__file__).resolve().parents[2])
    return root


def assemble_project_path(path):
    """Assemble a path relative to the project root directory"""
    if not os.path.isabs(path):
        path = os.path.join(get_project_root(), path)
    return path
