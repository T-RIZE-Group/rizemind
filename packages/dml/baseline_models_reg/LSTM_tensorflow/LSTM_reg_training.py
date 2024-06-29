# import pandas as pd
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split

# # Define LSTM network
# class LSTMNet(nn.Module):
#     def __init__(self, input_size: int, hidden_size: int, num_layers: int) -> None:
#         super(LSTMNet, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, 1)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
#         c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
#         out, _ = self.lstm(x, (h0, c0))
#         out = self.fc(out[:, -1, :])
#         return out