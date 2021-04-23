import torch
import torch.nn as nn


class RnnLstmModel(nn.Module):

    def __init__(self, input_size: int, output_size: int,
                 lstm_hidden_size: int, num_lstm_layers: int, device):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.lstm_hidden_size = lstm_hidden_size
        self.num_lstm_layers = num_lstm_layers
        self.device = device

        self.model = nn.ModuleDict({
            'lstm_1': nn.LSTM(input_size, lstm_hidden_size, num_lstm_layers),
            'lstm_2': nn.LSTM(input_size + lstm_hidden_size, lstm_hidden_size, num_lstm_layers),
            'lstm_3': nn.LSTM(input_size + lstm_hidden_size, output_size, num_lstm_layers),
            'softmax': nn.Softmax()
        })

    def forward(self, notes: torch.Tensor):
        batch_size = notes.size()[1]
        lstm_1_hidden = self.init_hidden(batch_size, self.lstm_hidden_size)
        lstm_2_hidden = self.init_hidden(batch_size, self.lstm_hidden_size)
        lstm_3_hidden = self.init_hidden(batch_size, self.output_size)

        # LSTM layer 1.
        lstm_1_output, lstm_1_hidden = self.model['lstm_1'](notes, lstm_1_hidden)
        # LSTM layer 2.
        lstm_2_input = torch.cat([notes, lstm_1_output], dim=-1)
        lstm_2_output, lstm_2_hidden = self.model['lstm_2'](lstm_2_input, lstm_2_hidden)
        # LSTM layer 3.
        lstm_3_input = torch.cat([notes, lstm_2_output], dim=-1)
        lstm_3_output, lstm_3_hidden = self.model['lstm_3'](lstm_3_input, lstm_3_hidden)
        # Output softmax layer.
        final_output = self.model['softmax'](lstm_3_output)

        return final_output

    def init_hidden(self, batch_size: int, hidden_size: int):
        hidden_state = torch.zeros(self.num_lstm_layers, batch_size, hidden_size,
                                   device=self.device)
        cell_state = torch.zeros(self.num_lstm_layers, batch_size, hidden_size,
                                 device=self.device)
        return hidden_state, cell_state
