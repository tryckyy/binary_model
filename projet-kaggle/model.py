import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.4)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.batch_norm3 = nn.BatchNorm1d(64)
        self.batch_norm4 = nn.BatchNorm1d(32)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.batch_norm1(x)
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.batch_norm2(x)
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.batch_norm3(x)
        x = torch.relu(self.fc4(x))
        x = self.batch_norm4(x)
        x = self.fc5(x)
        return x