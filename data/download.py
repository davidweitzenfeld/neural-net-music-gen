import shutil
import wget
import humanize
import re
import os
import argparse
from os import path
from urllib.parse import urlparse
from pathlib import Path

DATA_DIR = path.join('data')


# noinspection HttpUrlsUsage
def main(datasets: str):
    if 'lmd' in datasets or len(datasets) == 0:
        print('Downloading data from The Lakh MIDI Dataset '
              '(https://colinraffel.com/projects/lmd/)')
        lakh_url = 'http://hog.ee.columbia.edu/craffel/lmd'
        lakh_dir = path.join(DATA_DIR, 'lmd')
        download_and_extract('LMD-matched', f'{lakh_url}/lmd_matched.tar.gz', lakh_dir)
        download_and_extract('LMD-aligned', f'{lakh_url}/lmd_aligned.tar.gz', lakh_dir)
        download_and_extract('LMD-matched metadata', f'{lakh_url}/lmd_matched_h5.tar.gz', lakh_dir)
        print()

    if 'maestro' in datasets or len(datasets) == 0:
        print('Downloading data from MAESTRO Dataset '
              '(https://magenta.tensorflow.org/datasets/maestro)')
        maestro_url = 'https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0'
        maestro_dir = path.join(DATA_DIR, 'maestro')
        download_and_extract('maestro-v3-midi', f'{maestro_url}/maestro-v3.0.0-midi.zip',
                             maestro_dir)
        print()


def download_and_extract(name: str, url: str, target_dir: str = '.', overwrite: bool = False):
    extracted_dirname, filename = get_extracted_dirname(url, target_dir)
    if not path.exists(extracted_dirname):
        filename = download(name, url, target_dir, overwrite=overwrite)
        extract(filename, extract_dir=extracted_dirname)
    else:
        print(f'{extracted_dirname} already exists')
    if path.exists(filename):
        os.remove(filename)


def download(name: str, url: str, target_dir: str = '.', overwrite: bool = False) -> str:
    filename = path.basename(urlparse(url).path)
    filename = path.join(target_dir, filename)

    if not path.exists(filename) or overwrite:
        Path(target_dir).mkdir(parents=True, exist_ok=True)
        filename = wget.download(url, out=target_dir, bar=download_bar(name))
        print()
    else:
        print(f'{filename} already exists')

    return filename


def get_extracted_dirname(url: str, target_dir: str = '.') -> (str, str):
    filename = path.basename(urlparse(url).path)
    dirname = re.sub(r'(\.tar\.gz|\.zip)$', '', filename)
    return path.join(target_dir, dirname), filename


def download_bar(name: str):
    # noinspection PyUnusedLocal
    def bar(current_bytes: int, total_bytes: int, width: int) -> str:
        percent = (current_bytes / total_bytes) * 100
        current_human, total_human = [humanize.naturalsize(b) for b in [current_bytes, total_bytes]]
        return f'Downloading {name}: {percent:.2f}% [{current_human} / {total_human}]'

    return bar


def extract(filename: str, extract_dir: str):
    print(f'Extracting {filename}...', end=' ')
    shutil.unpack_archive(filename, extract_dir=extract_dir)
    print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data downloading.')
    parser.add_argument('--datasets', type=str, nargs='*', choices=['lmd', 'maestro'])
    args = parser.parse_args()
    main(args.datasets or [])
