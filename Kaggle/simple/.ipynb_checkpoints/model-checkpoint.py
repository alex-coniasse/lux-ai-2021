import torch
import copy
class MapEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(10, 64, 8, 2)
        self.conv2 = torch.nn.Conv2d(64, 128, 4, 2)
        self.relu = torch.nn.ReLU()
        self.ffn_map = torch.nn.Linear(3200,512)
    def forward(self, x):
        y = self.conv1(x)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.relu(y)
        flat = y.view(-1, 3200)
        out = self.ffn_map(flat)
        return out

class UnitRNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn_unit = torch.nn.GRU(8, 128, 2, dropout=0)
    def forward(self, x):
        hn = torch.zeros(2, 1, 128)
        for k in range (x.shape[1]):
            pred, hn = self.rnn_unit(x[:,k,:].unsqueeze(1), hn)
        return pred

class LuxrNet(torch.nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.unitRnn = UnitRNN()
        self.mapEncoder = MapEncoder()
        self.relu = torch.nn.ReLU()
        self.joiner = torch.nn.Linear(640,512)
        self.output = torch.nn.Linear(512,output_dim)

    def forward(self, global_features, units_features):
        glob = self.mapEncoder(global_features)
        units = self.unitRnn(units_features)
        hidden_out = self.joiner(torch.cat((glob, units.squeeze(1)), 1))
        out = self.output(self.relu(hidden_out))
        return out

class DDQN(torch.nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.online = LuxrNet(output_dim)
        self.target = copy.deepcopy(self.online)
        
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, global_features, units_features, mode):
        if mode == "online":
            return self.online(global_features, units_features)
        if mode == "target":
            return self.target(global_features, units_features)