import fractions
import re
import os
from os import path
from multiprocessing import Pool
from tqdm import tqdm
import midifile.notematrix as notematrix
import music21.midi.realtime
import numpy as np
from music21 import *
from sklearn.preprocessing import MultiLabelBinarizer

DATA_DIR = path.join('../..', 'data')

ENCODER_CLASSES = {
    'note': range(128),
    'time_signature_numerator': [float(i) for i in [2, 3, 4, 5, 6, 8, 9, 12]],
    'time_signature_denominator': [float(i) for i in [2, 4, 8, 16]],
    'note_duration': [0.0833, 0.1667, 0.25, 0.3333, 0.4167, 0.5, 0.5833, 0.6667, 0.75, 0.8333,
                      0.9167, 1.0, 1.0833, 1.1667, 1.25, 1.3333, 1.4167, 1.5, 1.5833, 1.6667, 1.75,
                      1.8333, 1.9167, 2.0, 2.0833, 2.1667, 2.25, 2.3333, 2.4167, 2.5, 2.5833,
                      2.6667, 2.75, 2.8333, 2.9167, 3.0, 3.1667, 3.25, 3.3333, 3.4167, 3.5, 3.6667,
                      3.75, 4.0, 4.25, 4.3333, 4.5, 4.6667, 4.75, 5.0, 5.25, 5.4167, 5.5, 5.75, 6.0,
                      6.5, 6.75, 6.8333, 7.0, 7.75, 8.0, 9.5, 10.0],
    'key_signature': [float(i) for i in range(-7, 7 + 1)]
}


def main():
    encode_dataset_to_matrix(path.join(DATA_DIR, 'maestro'), dshieble_parse_and_save)


def encode_dataset_to_matrix(dirname: str, parse_and_save_fn) -> int:
    filenames = get_midi_list(dirname)

    pool = Pool(processes=16)
    for _ in tqdm(pool.imap_unordered(parse_and_save_fn, filenames),
                  total=len(filenames), desc='Encoding', unit='song'):
        pass

    return len(filenames)


def dshieble_parse_and_save(filename: str):
    matrix = notematrix.midi_to_note_matrix(filename)
    np.save(re.sub(r'\.\w+$', '-dshieble-matrix.npy', filename), matrix)


def stanford_parse_and_save(filename: str):
    song = read_midi_music21(filename)
    matrix = midi_to_matrix(song)
    np.save(re.sub(r'\.\w+$', '-matrix.npy', filename), matrix)


# noinspection HttpUrlsUsage
def midi_to_matrix(song: stream.Stream) -> np.ndarray:
    """
    Creates a matrix encoding for the given song of size (n, 218),
    where n is the number of chords.

    Based on the implementation from the Stanford paper
    http://cs229.stanford.edu/proj2018/report/18.pdf.
    """
    chords = song.chordify()

    notes = []
    note_durations = []
    time_signature_numerators = []
    time_signature_denominators = []
    key_signatures = []

    for i, n in enumerate(chords.notesAndRests):
        notes.append([p.pitch.midi for p in ([] if type(n) is note.Rest else n)])

        ql = n.quarterLength.numerator / n.quarterLength.denominator \
            if type(n.quarterLength) is fractions.Fraction else n.quarterLength
        note_durations.append([np.min([10, round(ql, 4)])])
        time_signature_numerators.append([chords.timeSignature.numerator])
        time_signature_denominators.append([chords.timeSignature.denominator])
        key_signatures.append([chords.keySignature.sharps] if chords.keySignature is not None
                              else [])

    return np.hstack(
        [MultiLabelBinarizer(classes=classes).fit_transform(values) for classes, values in [
            (ENCODER_CLASSES['note'], notes),
            (ENCODER_CLASSES['note_duration'], note_durations),
            (ENCODER_CLASSES['time_signature_numerator'], time_signature_numerators),
            (ENCODER_CLASSES['time_signature_denominator'], time_signature_denominators),
            (ENCODER_CLASSES['key_signature'], key_signatures),
        ]])


def get_midi_list(dirname: str) -> [str]:
    files = [path.join(root, file)
             for root, _, files in os.walk(dirname)
             for file in files
             if re.search(r'\.midi?$', file)]
    return files


def read_midi_music21(filename: str) -> music21.stream.Stream:
    return converter.parseFile(filename, format='midi', quantizePost=True)


if __name__ == '__main__':
    main()
