import argparse
import re
import subprocess

import numpy as np

from utils import *
from matplotlib import pyplot as plt

ROOT_DIR = root_dir(1, __file__)

name_by_training_id = {
    'wbcyq': {'name': 'Simple; $s=50$; Mozart', 'sort_key': 0},
    'recnk': {'name': 'Simple; $s=50$; Rachmaninoff', 'sort_key': 1},
    'ypwyy': {'name': 'Simple; $s=50$; Scriabin', 'sort_key': 2},
    'qzkor': {'name': 'Complex; $s=50$; Mozart', 'sort_key': 3},
    'lkahp': {'name': 'Complex; $s=150$; Mozart', 'sort_key': 4},
    'bpucu': {'name': 'Complex; $s=200$; Mozart', 'sort_key': 5},
}


def main(loss_filename_or_dirname: str, out_dirname: str):
    filenames = [loss_filename_or_dirname] if path.isfile(loss_filename_or_dirname) \
        else [path.join(root, file)
              for root, _, files in os.walk(loss_filename_or_dirname)
              for file in files if re.search(r'\.loss\.csv', file)]

    def get_training_id(filename: str) -> str:
        return re.search(r'^(\w+)\.', path.basename(filename)).group(1)

    sorted_filenames = sorted([(filename, name_by_training_id[get_training_id(filename)])
                               for filename in filenames], key=lambda f: f[1]['sort_key'])
    for f in sorted_filenames:
        data = np.loadtxt(f[0], delimiter=',', skiprows=1)
        plt.plot(data[:, 0], data[:, 1], label=name_by_training_id[get_training_id(f[0])]['name'])

    plt.xlabel('Epoch')
    plt.ylabel(r'Categorical cross-entropy loss $\mathcal{L}$')
    plt.legend(loc='upper right', prop={'size': 8})

    out_filename = path.join(out_dirname, 'loss.pdf')
    plt.savefig(out_filename, format='pdf', bbox_inches='tight')
    plt.show()

    try:
        os.system(f'pdfcrop {out_filename} {out_filename}')
    except subprocess.CalledProcessError:
        print('Failed to run pdfcrop.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Loss plot generation.')
    parser.add_argument('--loss', type=str, nargs='?',
                        default=path.join(ROOT_DIR, 'weights', 'rnn'))
    parser.add_argument('--output_dirname', type=str, nargs='?',
                        default=path.join(ROOT_DIR, 'weights', 'rnn'))
    args = parser.parse_args()
    main(args.loss, args.output_dirname)
