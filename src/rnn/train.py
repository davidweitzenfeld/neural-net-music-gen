import argparse

import tensorflow.keras as tfk

from midifile.processing import *
from model import create_lstm_rnn_model
from utils import *

ROOT_DIR = root_dir(2, __file__)
DATA_DIR = path.join(ROOT_DIR, 'data')
LMD_DIR = path.join(DATA_DIR, 'lmd')
MAESTRO_DIR = path.join(DATA_DIR, 'maestro')
DATASET_DIR = MAESTRO_DIR

WEIGHTS_DIR = path.join(ROOT_DIR, 'weights', 'rnn')


def main(seq_len: int, batch_size: int, epoch_count: int, artist: str):
    print(DATASET_DIR)
    song_filenames = get_maestro_midi_list_by_composer(DATASET_DIR)[artist]
    print(f'Found {len(song_filenames)} song files.')
    notes = load_shubham_training_data(song_filenames, seq_len)
    print('Loaded data.')

    train(notes, seq_len, batch_size, epoch_count)


def load_dshieble_training_data(song_filenames: [str], timestep_count: int):
    songs = [np.load(re.sub(r'\.midi?$', '-dshieble-matrix.npy', f)) for f in song_filenames]
    songs = [song[:int(np.floor(song.shape[0] / timestep_count) * timestep_count)] for song in
             songs]
    return songs


def load_shubham_training_data(song_filenames: [str], timestep_count: int):
    def load_pickle(filename: str):
        with open(filename, 'rb') as f:
            return pickle.load(f)

    data_filenames = [re.sub(r'\.midi?$', '-notes.pl', name) for name in song_filenames]
    notes = [note for f in data_filenames if path.exists(f) for note in load_pickle(f)]
    print(f'Got {len(notes)} types.')
    return notes


def train(notes: [int], seq_len: int, batch_size: int, epoch_count: int):
    print('Processing training data.')
    inputs, targets = notes_to_matrix(notes, seq_len)

    print('Creating model.')
    model = create_lstm_rnn_model(inputs.shape, vocab_size=targets.shape[1])
    print(model.summary())

    training_id = rand_str(5)
    config_filename = path.join(WEIGHTS_DIR, f'{training_id}.config.json')
    write_dict(config_filename,
               {'seq_len': seq_len, 'batch_size': batch_size, 'epoch_count': epoch_count})
    checkpoint_filename = path.join(WEIGHTS_DIR, f'{training_id}.checkpoint.hdf5')
    checkpoint = tfk.callbacks.ModelCheckpoint(checkpoint_filename,
                                               monitor='loss', save_best_only=True)

    print('Starting training.')
    model.fit(inputs, targets, batch_size, epoch_count, callbacks=[checkpoint])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model training.')
    parser.add_argument('--seq_len', type=int, nargs=1, default=50)
    parser.add_argument('--batch_size', type=int, nargs=1, default=32)
    parser.add_argument('--epoch_count', type=int, nargs=1, default=25)
    parser.add_argument('--artist', type=str, nargs=1, default='Alexander Scriabin')
    args = parser.parse_args()
    main(args.seq_len, args.batch_size, args.epoch_count, args.artist)
