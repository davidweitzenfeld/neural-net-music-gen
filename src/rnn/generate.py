import argparse
import numpy as np
import tensorflow.keras as tfk

from music21 import *
from midifile.processing import *
from train import load_shubham_training_data, notes_to_matrix
from model import create_lstm_rnn_model
from utils import *
from tqdm import trange

ROOT_DIR = root_dir(2, __file__)
DATA_DIR = path.join(ROOT_DIR, 'data')
LMD_DIR = path.join(DATA_DIR, 'lmd')
MAESTRO_DIR = path.join(DATA_DIR, 'maestro')
DATASET_DIR = MAESTRO_DIR

WEIGHTS_DIR = path.join(ROOT_DIR, 'weights', 'rnn')
OUTPUTS_DIR = path.join(ROOT_DIR, 'outputs', 'rnn')


def main(training_id: str, gen_seq_len: int, name: str):
    config_filename = path.join(WEIGHTS_DIR, f'{training_id}.config.json')
    weights_filename = path.join(WEIGHTS_DIR, f'{training_id}.weights.hdf5')
    output_filename = path.join(OUTPUTS_DIR, f'{training_id}.{name}.mid')
    assert_all_exist([config_filename, weights_filename])

    config = load_dict(config_filename)

    print('Loading data.')
    song_filenames = get_maestro_midi_list_by_composer(DATASET_DIR)[config['artist']]
    notes = load_shubham_training_data(song_filenames, config['seq_len'])

    song = generate_song(notes, weights_filename, output_filename, config['seq_len'], gen_seq_len)
    song.plot()


def generate_song(data: [str], weights_filename: str, output_filename: str, seq_len: int,
                  gen_seq_len: int) -> stream.Stream:
    note_names = sorted(set(data))
    vocab_size = len(note_names)

    inputs, _ = notes_to_matrix(data, seq_len)

    print('Loading model.')
    model = create_lstm_rnn_model(inputs.shape, vocab_size)
    print(model.summary())
    model.load_weights(weights_filename)

    generated = generate_notes(model, inputs, note_names, gen_seq_len)
    song = notes_to_midi(generated)
    song.write('midi', fp=output_filename)
    return song


def generate_notes(model: tfk.Model, inputs: np.ndarray, note_names: [str], gen_seq_len: int):
    vocab_size = len(note_names)

    start_note_idx = np.random.randint(0, len(inputs) - 1)
    int_to_note = {i: n for i, n in enumerate(note_names)}

    note_enc = list(inputs[start_note_idx])
    generated = []

    print('Generating notes.')
    for _ in trange(gen_seq_len):
        model_input = np.reshape(note_enc, (1, -1, 1)).astype('float32')

        gen_enc = model.predict(model_input)
        gen_note_idx = np.argmax(gen_enc)
        gen_note = int_to_note[gen_note_idx]
        generated.append(gen_note)

        note_enc.append(gen_note_idx / float(vocab_size))
        note_enc = note_enc[1:len(note_enc)]

    print('Done')
    return generated


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Song generation.')
    parser.add_argument('--training_id', type=str, nargs='?')
    parser.add_argument('--seq_len', type=int, nargs='?', default=150)
    args = parser.parse_args()
    main(args.training_id or input('Training ID? '), args.seq_len)
