import torch
import math
import torch.nn as nn
import torch
import math
import torch.nn as nn

class cus_EEGNet(nn.Module):
    def __init__(self,conv1_size=64, sep_conv_size=16 ,n_Hb=1, time_point=134, channel_num =20,  acti_fun='elu', pooling_type = 'avgPool'):
        super(cus_EEGNet, self).__init__()
        self.conv1_size = conv1_size
        self.n_Hb = n_Hb
        self.time_point = time_point
        self.channel_num = channel_num
        self.acti_fun = acti_fun
        self.sep_conv_size = sep_conv_size

        self.F1 = 8*self.n_Hb
        self.F2 = 16*self.n_Hb
        self.D = 2
        self.padding = int(self.conv1_size/2)
        self.n_linear = int(self.time_point/4/8)*self.n_Hb
        self.actf = nn.ELU()
        # self.pooling1 = nn.AvgPool2d((1, 4))
        # self.pooling2 = nn.AvgPool2d((1, 8))

        if pooling_type == 'avgPool':
          self.pooling1 = nn.AvgPool2d((1, 4))
          self.pooling2 = nn.AvgPool2d((1, 8))
        elif pooling_type == 'maxPool':
          self.pooling1 = nn.MaxPool2d((1, 4))
          self.pooling2 = nn.MaxPool2d((1, 8))

        if acti_fun == 'relu':
          self.actf = nn.ReLU()

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.n_Hb, self.F1, (1, self.conv1_size), padding=(0, self.padding), bias=False),
            nn.BatchNorm2d(self.F1)
        )

        self.conv2_1 = nn.Sequential(
            nn.Conv2d(self.F1, self.D*self.F1, (self.channel_num, 1), groups=self.F1, bias=False),
            nn.BatchNorm2d(self.D*self.F1)
        )
        # self.conv2_2 = nn.Sequential(
        #     # nn.AvgPool2d((1,4)),
        #     self.pooling1,
        #     nn.Dropout(0.5)
        # )
        # self.pooling1
        self.dropout = nn.Dropout(0.5)
        self.Conv3_1 = nn.Sequential(
            nn.Conv2d(self.D*self.F1, self.D*self.F1, (1, self.sep_conv_size), padding=(0, int(self.sep_conv_size/2)), groups=self.D*self.F1, bias=False),
            nn.Conv2d(self.D*self.F1, self.F2, (1, 1), bias=False),
            nn.BatchNorm2d(self.F2)
        )
        # self.Conv3_2 = nn.Sequential(
        #     # nn.AvgPool2d((1, 8)),
        #     self.pooling2,
        #     nn.Dropout(0.5)
        # )

        self.classifier = nn.Linear(16*self.n_linear, 3, bias=True)
        
    def activation_Fun(self,x):
        if self.acti_fun == 'relu' or self.acti_fun == 'elu':
          x = self.actf(x)
        elif self.acti_fun == 'square':
          # print('square')
          x = x**2
        elif self.acti_fun == 'linear':
          # print('linear')
          x=x
        return x
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_1(x)
        x = self.activation_Fun(x)
        x = self.pooling1(x)
        x = self.dropout(x)
        # x = self.conv2_2(x)
        x = self.Conv3_1(x)
        x = self.activation_Fun(x)
        x = self.pooling2(x)
        x = self.dropout(x)
        # x = self.Conv3_2(x)
        x = x.view(-1, 16*self.n_linear)
        x = self.classifier(x)
        # x = self.softmax(x)
        return x
        
        
class EEGNet(nn.Module):
    def __init__(self,conv1_size=64, n_Hb=1):
        super(EEGNet, self).__init__()
        self.conv1_size = conv1_size
        self.F1 = 8*n_Hb
        self.F2 = 16
        self.D = 2
        self.padding = int(conv1_size/2)

        self.conv1 = nn.Sequential(
            nn.Conv2d(n_Hb, self.F1, (1, conv1_size), padding=(0, self.padding), bias=False),
            nn.BatchNorm2d(self.F1)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(self.F1, self.D*self.F1, (20, 1), groups=self.F1, bias=False),
            nn.BatchNorm2d(self.D*self.F1),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.5)
        )

        self.Conv3 = nn.Sequential(
            nn.Conv2d(self.D*self.F1, self.D*self.F1, (1, 16), padding=(0, 8), groups=self.D*self.F1, bias=False),
            nn.Conv2d(self.D*self.F1, self.F2, (1, 1), bias=False),
            nn.BatchNorm2d(self.F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.5)
        )


        self.classifier = nn.Linear(16*4, 3, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.Conv3(x)

        x = x.view(-1, 16*4)
        x = self.classifier(x)
        return x
        
        

class EEGNet_25s(nn.Module):
    def __init__(self,conv1_size=64, n_Hb=1):
        super(EEGNet_25s, self).__init__()
        self.conv1_size = conv1_size
        self.F1 = 8*n_Hb
        self.F2 = 16
        self.D = 2
        self.padding = int(conv1_size/2)

        self.conv1 = nn.Sequential(
            nn.Conv2d(n_Hb, self.F1, (1, conv1_size), padding=(0, self.padding), bias=False),
            nn.BatchNorm2d(self.F1)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(self.F1, self.D*self.F1, (20, 1), groups=self.F1, bias=False),
            nn.BatchNorm2d(self.D*self.F1),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.5)
        )

        self.Conv3 = nn.Sequential(
            nn.Conv2d(self.D*self.F1, self.D*self.F1, (1, 16), padding=(0, 8), groups=self.D*self.F1, bias=False),
            nn.Conv2d(self.D*self.F1, self.F2, (1, 1), bias=False),
            nn.BatchNorm2d(self.F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.5)
        )


        self.classifier = nn.Linear(16*10, 3, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.Conv3(x)

        x = x.view(-1, 16*10)
        x = self.classifier(x)
        return x
	
    
class SCCNet(nn.Module):
    def __init__(self,temporal_filter=12):
        super(SCCNet, self).__init__()
        # bs, 1, channel, sample
        self.conv1 = nn.Conv2d(1, 20, (20, 1)) # 更改
        self.Bn1 = nn.BatchNorm2d(20) # 更改
        # bs, 22, 1, sample
        self.conv2 = nn.Conv2d(20, 20, (1, temporal_filter), padding=(0, int(temporal_filter/2))) #更改
        self.Bn2   = nn.BatchNorm2d(20)
        self.Drop1 = nn.Dropout(0.5)
        self.AvgPool1 = nn.AvgPool2d((1, 62), stride=(1, 12))
        self.classifier = nn.Linear(20*7, 3, bias=True) # 更改


    def forward(self, x):
        x = self.conv1(x)
        x = self.Bn1(x)
        x = self.conv2(x)
        x = self.Bn2(x)
        x = x ** 2
        x = self.Drop1(x)
        x = self.AvgPool1(x)
        x = torch.log(x)
        x = x.view(-1, 20*7) # 更改
        # x = x.view(-1, 840) # 原本的
        x = self.classifier(x)

        #x = self.softmax(x)
        return x    
    
    
class SCCNet_25s(nn.Module):
    def __init__(self,temporal_filter=12):
        super(SCCNet_25s, self).__init__()
        # bs, 1, channel, sample
        self.conv1 = nn.Conv2d(1, 20, (20, 1)) # 更改
        self.Bn1 = nn.BatchNorm2d(20) # 更改
        # bs, 22, 1, sample
        self.conv2 = nn.Conv2d(20, 20, (1, temporal_filter), padding=(0, int(temporal_filter/2))) #更改
        self.Bn2   = nn.BatchNorm2d(20)
        self.Drop1 = nn.Dropout(0.5)
        self.AvgPool1 = nn.AvgPool2d((1, 62), stride=(1, 12))
        self.classifier = nn.Linear(460, 3, bias=True) # 更改


    def forward(self, x):
        x = self.conv1(x)
        x = self.Bn1(x)
        x = self.conv2(x)
        x = self.Bn2(x)
        x = x ** 2
        x = self.Drop1(x)
        x = self.AvgPool1(x)
        x = torch.log(x)
        x = x.view(-1, 460) # 更改
        x = self.classifier(x)

        #x = self.softmax(x)
        return x       
    
	
class ShallowConvNet(nn.Module):
    def __init__(self,conv1_size=13):
        super(ShallowConvNet, self).__init__()

        self.conv1_size = conv1_size
        self.conv1 = nn.Conv2d(1, 40, (1, conv1_size), bias=False) 
        self.conv2 = nn.Conv2d(40, 40, (20, 1), bias=False)
        self.Bn1   = nn.BatchNorm2d(40)
        # self.SquareLayer = square_layer()
        self.AvgPool1 = nn.AvgPool2d((1, 35), stride=(1, 7))
        # self.LogLayer = Log_layer()
        self.Drop1 = nn.Dropout(0.25)
        self.classifier = nn.Linear(40*int((135-conv1_size-35)/7+1), 3, bias=True) # 更改
        # self.classifier = nn.Linear(40*74, 3, bias=True) # 原本
        #self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.conv1(x) #w1 = 134 - conv1_size + 1
        x = self.conv2(x) #w2 = w1
        x = self.Bn1(x)
        x = x ** 2
        x = self.AvgPool1(x) #w3 = (w2-35)/7+1
        x = torch.log(x)
        x = self.Drop1(x)
        x = x.view(-1, 40*int((135-self.conv1_size-35)/7+1)) # 40*w3
        # x = x.view(-1, 40*74) # 原本
        x = self.classifier(x)

        #x = self.softmax(x)
        return x
	
	
	
class ShallowConvNet2(nn.Module):
    def __init__(self,conv1_size=13):
        super(ShallowConvNet2, self).__init__()

        self.classifier_size = 40*int(max(0,(135-conv1_size-35)/7)+1)
        self.conv1 = nn.Conv2d(1, 40, (1, conv1_size), bias=False) 
        self.conv2 = nn.Conv2d(40, 40, (20, 1), bias=False)
        self.Bn1   = nn.BatchNorm2d(40)
        # self.SquareLayer = square_layer()
        self.AvgPool1 = nn.AvgPool2d((1, min(135-conv1_size, 35)), stride=(1, 7))
        # self.LogLayer = Log_layer()
        self.Drop1 = nn.Dropout(0.25)
        self.classifier = nn.Linear(40*int(max(0,(135-conv1_size-35)/7)+1), 3, bias=True)
        # self.classifier = nn.Linear(40*74, 3, bias=True) # 原本
        #self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.conv1(x) #w1 = 134 - conv1_size + 1
        print(x.shape)
        x = self.conv2(x) #w2 = w1
        x = self.Bn1(x)
        x = x ** 2
        print(x.shape)
        x = self.AvgPool1(x) #w3 = (w2-35)/7+1
        print(x.shape)
        x = torch.log(x)
        x = self.Drop1(x)
        #x = x.view(-1, 40*int((135-self.conv1_size-35)/7+1)) # 40*w3
        x = x.view(-1, self.classifier_size) # 40*w3
        # x = x.view(-1, 40*74) # 原本
        x = self.classifier(x)

        #x = self.softmax(x)
        return x