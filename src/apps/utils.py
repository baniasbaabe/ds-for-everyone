import os
from typing import Any, List

def get_file_path(directory: List) -> str:
    return os.path.abspath(os.path.join(os.path.dirname( __file__ ), *directory))