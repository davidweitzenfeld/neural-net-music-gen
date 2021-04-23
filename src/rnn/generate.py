import numpy as np
import torch

from model import RnnLstmModel
import midifile.notematrix as notematrix


def main():
    model: RnnLstmModel = torch.load('../../weights/rnn.pt')
    model.eval()

    generate_song(model, timestep_count=15)


def generate_song(model: RnnLstmModel, timestep_count: int):
    device = torch.device('cuda')
    input_vec = np.zeros((1, 1, 2340), dtype=float)
    outputs = []

    for timestep in range(timestep_count):
        x = model(torch.from_numpy(input_vec).float().to(device)).detach().cpu()
        max_ind = np.argmax(x)
        output = np.zeros(x.shape)
        output[max_ind] = 1
        outputs.append(output)

    notematrix.note_matrix_to_midi(np.array(outputs), name='generated')


if __name__ == '__main__':
    main()
