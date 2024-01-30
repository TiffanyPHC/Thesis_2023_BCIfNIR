import torch
import math
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class NIRS_ANN(nn.Module):
    def __init__(self, Hb_num, ch, time_point, class_num):
        super(NIRS_ANN, self).__init__()
        input_dim = Hb_num * ch * time_point

        self.fulcon1 = nn.Linear(input_dim, 20, bias=True)
        self.act3 = nn.ReLU()
        self.fulcon2 = nn.Linear(20, 10, bias=True)
        self.act4 = nn.ReLU()
        self.classifier = nn.Linear(10, class_num, bias=True)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input: (batch_size, Hb_num * ch * time_point)
        x = self.fulcon1(x)
        x = self.act3(x)
        x = self.fulcon2(x)
        x = self.act4(x)
        x = self.classifier(x)

        return x


class NIRS_CNN(nn.Module):
    # https://www.frontiersin.org/articles/10.3389/fnrgo.2023.994969/full
    # Benchmarking framework for machine learning classification from fNIRS data
    def __init__(self, Hb_num, ch, time_point=334, class_num=3):
        super(NIRS_CNN, self).__init__()
        # bs, 1, channel, sample
        self.conv1 = nn.Conv2d(Hb_num, 4, (1, 10), padding=(0, 5),stride=(1, 2)) #更改
        self.MaxPool1 = nn.MaxPool2d((1, 2))
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(4, 4, (1, 5), padding=(0, 2),stride=(1, 2)) #更改
        self.MaxPool2 = nn.MaxPool2d((1, 2))
        self.act2 = nn.ReLU()
        self.fulcon1 = nn.Linear(4*ch*math.ceil(time_point/16), 20, bias=True) # 更改
        self.act3 = nn.ReLU()
        self.fulcon2 = nn.Linear(20, 10, bias=True) # 更改
        self.act4 = nn.ReLU()
        self.classifier = nn.Linear(10, class_num, bias=True) # 更改

        self.linear = 4*ch*math.ceil(time_point/16)

    def forward(self, x):
        x = self.conv1(x)
        x = self.MaxPool1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.MaxPool2(x)
        x = x.view(-1, self.linear) # 更改
        x = self.fulcon1(x)
        x = self.act3(x)
        x = self.fulcon2(x)
        x = self.act4(x)
        x = self.classifier(x)

        #x = self.softmax(x)
        return x

class NIRS_LSTM(nn.Module):
    # https://www.frontiersin.org/articles/10.3389/fnrgo.2023.994969/full
    # Benchmarking framework for machine learning classification from fNIRS data
    def __init__(self, num_electrodes=20, hidden_size=36, num_classes=3):
        super(NIRS_LSTM, self).__init__()
        self.lstm = nn.LSTM(num_electrodes, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 16)
        self.fc2 = nn.Linear(16, num_classes)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        # x: (batch_size, sequence_length, input_size)

        # LSTM layer
        output, _ = self.lstm(x)

        # Take the last output of the sequence
        output = output[:, -1, :]

        # Fully connected layers with ReLU activation
        output = self.relu(self.fc1(output))
        output = self.relu(self.fc2(output))

        return output
        

def LSTM_getDataLoader(x, y, batch_size, onehot_encoding):
    data = torch.from_numpy(x).float()

    # # # label + onehot
    testlabel = torch.from_numpy(y).int() # change type to your use case
    testlabel = testlabel.type(torch.int64)

    if onehot_encoding:
        labels = torch.zeros([len(testlabel), len(torch.unique(testlabel)) ])
        labels.scatter_(1, testlabel, 1)
    else:
        labels = testlabel.float()

    dataset = torch.utils.data.TensorDataset(data, labels)
    data_loader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle = True, num_workers = 2, pin_memory = True)
    return data_loader