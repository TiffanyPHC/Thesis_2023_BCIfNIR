import torch
from scipy import io
import numpy as np
from torch.utils.data import Dataset, DataLoader
# from termcolor import colored
class fnirs_data:
    # fNIRS類別定義
    def __init__(self, HbO, HbR, HbT, labels):
        self.HbO = HbO
        self.HbR = HbR
        self.HbT = HbT
        self.labels = labels


def read_one_file(s1, s2, id):
    
    # 讀取一個mat格式的fNIRS 資料
    # mat檔案位置為 s1 id s2
    # 回傳fnirs_data格式
    
    if id<10:
        sub = '0'+str(id)
    else:
        sub = str(id)
    data = io.loadmat(s1+sub+s2)
    HbO = data['HbO']
    HbR = data['HbR']
    HbT = HbO + HbR
    labels = data['labels']
    one_data = fnirs_data(HbO=HbO, HbR=HbR, HbT=HbT, labels=labels)
    return one_data
    
def leave_subject_out(s1, s2, data_range, test_id):
    # s1=path_dataset+'S'
    # s2='.mat'
    # data_range=range(1,31)
    # test_id=25
    print("leave subject out")
    test_id=[test_id]
    train_id=list(set(data_range)-set(test_id))

    print("train id:",end="")
    id=train_id[0]
    print(id,end=', ')
    train = read_one_file(s1, s2, id)
    
    for ID in train_id[1:]:
        print(ID,end=', ')
        temp = read_one_file(s1, s2, ID)
        train.HbO = np.concatenate((train.HbO,temp.HbO),axis=0)
        train.HbR = np.concatenate((train.HbR,temp.HbR),axis=0)
        train.HbT = np.concatenate((train.HbT,temp.HbT),axis=0)
        train.labels = np.concatenate((train.labels,temp.labels),axis=0) 
    
    test_id=test_id[0]
    print("test id: ",test_id)
    test = read_one_file(s1, s2, test_id)

    if train.labels.shape[1] == 1 and test.labels.shape[1] == 1 : # no one-hot encoding label
        if min(train.labels)>0 and min(train.labels)==min(test.labels):
            print('modify label')
            test.labels = test.labels-min(train.labels)
            train.labels = train.labels-min(train.labels)
        
        if min(train.labels)!=min(test.labels):
            print('Error: Training set and test set markings may be wrong!!!')
            print('The minimum values of the training set and test set labels are not equal.')

    print("training data dimension: ",train.HbO.shape)
    print("training label dimension: ",train.labels.shape)
    print("testing data dimension: ",test.HbO.shape)
    print("testing label dimension: ",test.labels.shape)

    return test, train


def leave_trial_out(s1, s2, subject_id, range1, range2):
    print("leave trial",len(range1),"out")
    train = read_one_file(s1, s2, subject_id)
    test = read_one_file(s1, s2, subject_id)
    # range1=list(range(0,5,1))
    # range2=list(range(5,75,1))
    
    test.HbO = train.HbO[range1,:,:]
    test.HbR = train.HbR[range1,:,:]
    test.HbT = train.HbT[range1,:,:]
    test.labels = train.labels[range1,:]

    train.HbO = train.HbO[range2,:,:]
    train.HbR = train.HbR[range2,:,:]
    train.HbT = train.HbT[range2,:,:]
    train.labels = train.labels[range2,:]
    
    print("training data dimension: ",train.HbO.shape)
    print("training label dimension: ",train.labels.shape)
    print("testing data dimension: ",test.HbO.shape)
    print("testing label dimension: ",test.labels.shape)
    
    return test, train
    
    

def getDataLoader(x, y, batch_size, onehot_encoding):
    
    # 訓訓練資料轉為可以放進模型的格式
    
    data = torch.from_numpy(x).float()
    if len(data.shape)==3:
        data = data.unsqueeze(1)

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
    
    
