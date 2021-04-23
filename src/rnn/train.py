import re

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from os import path
from tqdm import trange
from midifile.processing import get_midi_list
from model import RnnLstmModel

DATA_DIR = path.join('..', '..', 'data')
LMD_DIR = path.join(DATA_DIR, 'lmd')
MAESTRO_DIR = path.join(DATA_DIR, 'maestro')
DATASET_DIR = MAESTRO_DIR


def main():
    timestep_count = 15

    song_filenames = get_midi_list(DATASET_DIR)
    print(f'Found {len(song_filenames)} song files.')

    songs = [np.load(re.sub(r'\.midi?$', '-dshieble-matrix.npy', f)) for f in song_filenames]
    songs = [song[:int(np.floor(song.shape[0] / timestep_count) * timestep_count)] for song in
             songs]
    print('Loaded data.')

    print('Starting training.')
    train(songs, timestep_count)
    print('Training complete.')


def train(training_data: [np.ndarray], timestep_count: int):
    epoch_count = 5

    device = torch.device('cuda')
    input_size, output_size = training_data[0].shape[1] * timestep_count, training_data[0].shape[1]
    model = RnnLstmModel(input_size, lstm_hidden_size=100, output_size=output_size,
                         num_lstm_layers=1, device=device).to(device)
    loss_function = nn.L1Loss()
    optimizer = optim.RMSprop(model.parameters())

    # Convert songs
    # from shape (timesteps_in_song, 2 * note_range)
    # to   shape (timesteps_in_song / timestep_count, 2 * note_range * timestep_count).
    print('Processing training data.')
    inputs = [np.reshape(song, (song.shape[0] // timestep_count,
                                song.shape[1] * timestep_count))[:-1]
              for song in training_data]
    targets = [song[np.arange(timestep_count, song.shape[0], timestep_count)]
               for song in training_data]
    assert all(x.shape[0] == y.shape[0] for x, y in zip(inputs, targets))
    inputs = [torch.unsqueeze(torch.from_numpy(song).float(), dim=1) for song in inputs]
    targets = [torch.unsqueeze(torch.from_numpy(target), dim=1) for target in targets]

    with torch.no_grad():
        outputs = model(inputs[0].to(device))
        print(outputs.shape)

    print('Starting training epochs.')
    for epoch in trange(epoch_count):
        losses = []
        for song, target in zip(inputs, targets):
            # PyTorch accumulates gradients, so we need to clear them first.
            model.zero_grad()

            # Run forward pass.
            outputs = model(song.to(device))

            # Compute loss, gradients and update parameters.
            loss = loss_function(outputs, target.to(device))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f'Average loss of epoch {epoch} is {np.mean(losses)}.')

    torch.save(model, '../../weights/rnn.pt')





if __name__ == '__main__':
    main()
