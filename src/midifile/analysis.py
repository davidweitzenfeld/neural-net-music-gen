import argparse

from midifile.processing import *


def main(filename: str):
    song = read_midi_music21(filename)
    song.plot('pianoRoll', 'pitchClass', title='')
    song.plot('histogram', 'pitchClass', title='')
    print(song)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MIDI file analysis.')
    parser.add_argument('--filename', type=str, nargs='?',
                        default='../../outputs/rnn/lkahp.sample.mid')
    args = parser.parse_args()
    main(args.filename or input('Filename? '))
