import torch
import copy
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchsummary import summary

class MapEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(8, 64, 8, 2)
        self.conv2 = torch.nn.Conv2d(64, 64, 4, 2)
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
        self.rnn = torch.nn.GRU(9, 128, 1, dropout=0, batch_first=True)
        self.mapEncoder = MapEncoder()
        self.relu = torch.nn.ReLU()
        self.joiner = torch.nn.Linear(640,512)
        self.output = torch.nn.Linear(512,output_dim)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, global_features, units_features, packed=False):
        bs = global_features.shape[0]
        h0 = torch.zeros(1, bs, 128).to(self.device)
        glob = self.mapEncoder(global_features)
        preds, hn = self.rnn(units_features, h0)
        if packed:
            preds, lens_unpacked = torch.nn.utils.rnn.pad_packed_sequence(preds, batch_first=True)
        seq_len = preds.shape[1]
        hidden_out = self.joiner(torch.cat((glob.unsqueeze(1).repeat(1,seq_len,1), preds), 2))
        out = self.output(self.relu(hidden_out))
        return [out[:,k,:] for k in range(seq_len)]
        # return out


class SimpleNet(torch.nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 32, kernel_size=(16, 16), padding='same')
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=(8, 8), padding='same')
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=(8, 8), padding='same')
        self.out_unit = torch.nn.Conv2d(64, output_dim, kernel_size=(1,1), padding='same')
        self.out_city = torch.nn.Conv2d(64, 2, kernel_size=(1,1), padding='same')
        self.relu = torch.nn.ReLU()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, global_features):
        y = self.conv1(global_features)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.relu(y)
        y = self.conv3(y)
        y = self.relu(y)

        act_city = self.relu(self.out_city(y))
        act_units = self.relu(self.out_unit(y))

        return act_units, act_city

class DDQN(torch.nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.online = SimpleNet(output_dim)
        self.target = copy.deepcopy(self.online)

        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, state, mode):
 
        if mode == "online":
            return self.online(state)
        if mode == "target":
            return self.target(state)
   
