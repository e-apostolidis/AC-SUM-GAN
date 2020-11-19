# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class StackedLSTMCell(nn.Module):

    def __init__(self, num_layers, input_size, rnn_size, dropout=0.0):
        super(StackedLSTMCell, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, x, h_c):
        """
        Args:
            x: [batch_size, input_size]
            h_c: [2, num_layers, batch_size, hidden_size]
        Return:
            last_h_c: [2, batch_size, hidden_size] (h from last layer)
            h_c_list: [2, num_layers, batch_size, hidden_size] (h and c from all layers)
        """
        h_0, c_0 = h_c
        h_list, c_list = [], []
        for i, layer in enumerate(self.layers):
            # h of i-th layer
            h_i, c_i = layer(x, (h_0[i], c_0[i]))

            # x for next layer
            x = h_i
            if i + 1 != self.num_layers:
                x = self.dropout(x)
            h_list += [h_i]
            c_list += [c_i]

        last_h_c = (h_list[-1], c_list[-1])
        h_list = torch.stack(h_list)
        c_list = torch.stack(c_list)
        h_c_list = (h_list, c_list)

        return last_h_c, h_c_list
