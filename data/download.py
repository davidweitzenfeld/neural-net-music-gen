import shutil
import wget
import humanize
import os
from os import path
from urllib.parse import urlparse


# noinspection HttpUrlsUsage
def main():
    print('Downloading data from The Lakh MIDI Dataset (https://colinraffel.com/projects/lmd/)')
    lakh_base_url = 'http://hog.ee.columbia.edu/craffel/lmd'
    download_and_extract('LMD-aligned', f'{lakh_base_url}/lmd_aligned.tar.gz')
    download_and_extract('LMD-matched metadata', f'{lakh_base_url}/lmd_matched_h5.tar.gz')


def download_and_extract(name: str, url: str, overwrite: bool = False):
    filename = download(name, url, overwrite=overwrite)
    extract(filename)
    os.remove(filename)


def download(name: str, url: str, overwrite: bool = False) -> str:
    filename = path.basename(urlparse(url).path)
    if not path.exists(filename) or overwrite:
        wget.download(url, bar=download_bar(name))
        print()
    else:
        print(f'{filename} already exists')

    return filename


def download_bar(name: str):
    # noinspection PyUnusedLocal
    def bar(current_bytes: int, total_bytes: int, width: int) -> str:
        percent = (current_bytes / total_bytes) * 100
        current_human, total_human = [humanize.naturalsize(b) for b in [current_bytes, total_bytes]]
        return f'Downloading {name}: {percent:.2f}% [{current_human} / {total_human}]'

    return bar


def extract(filename: str):
    print(f'Extracting {filename}...', end=' ')
    shutil.unpack_archive(filename)
    print('Done!')


if __name__ == '__main__':
    main()
