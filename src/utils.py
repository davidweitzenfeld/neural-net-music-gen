import json
import os
import random
import string
from os import path
from typing import Dict
from pathlib import Path


def rand_str(str_len: int, alphabet: [str] = string.ascii_lowercase) -> str:
    return ''.join(random.choice(alphabet) for _ in range(str_len))


def write_dict(filename: str, content: Dict):
    Path(path.dirname(filename)).mkdir(parents=True, exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(content, f)


def load_dict(filename: str):
    with open(filename, 'r') as f:
        return json.load(f)


def assert_all_exist(filenames: [str]):
    existence = {filename: path.exists(filename) for filename in filenames}
    if not all(existence.values()):
        do_not_exist = [filename for filename, exists in existence.items() if not exists]
        raise RuntimeError(f'The following required files do not exist: {", ".join(do_not_exist)}.')


def root_dir(levels_to_root: int, current_file) -> str:
    current_working_dir = os.getcwd()
    current_file_dir = path.dirname(path.realpath(current_file))
    if current_working_dir == current_file_dir:
        return path.join(*(['..'] * levels_to_root))
    else:
        return '.'
