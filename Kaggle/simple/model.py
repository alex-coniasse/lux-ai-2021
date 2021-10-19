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

class LuxrNet(torch.nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.rnn = torch.nn.GRU(8, 128, 2, dropout=0)
        self.mapEncoder = MapEncoder()
        self.relu = torch.nn.ReLU()
        self.joiner = torch.nn.Linear(640,512)
        self.output = torch.nn.Linear(512,output_dim)

    def forward(self, global_features, units_features):
        values =[]
        hk = torch.zeros(2, 1, 128)
        glob = self.mapEncoder(global_features)
        for k in range (units_features.shape[1]):
            pred, hk = self.rnn(units_features[:,k,:].unsqueeze(1), hk)
            hidden_out = self.joiner(torch.cat((glob, pred.squeeze(1)), 1))
            out_k = self.output(self.relu(hidden_out))
            values.append(out_k)
        return values

class DDQN(torch.nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.online = LuxrNet(output_dim)
        self.target = copy.deepcopy(self.online)
        
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, state, mode):
        global_features = state[0]
        units_features = state[1]
        if mode == "online":
            return self.online(global_features, units_features)
        if mode == "target":
            return self.target(global_features, units_features)
   

