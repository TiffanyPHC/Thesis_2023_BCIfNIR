import torch
from scipy import io
import numpy as np
from torch.utils.data import Dataset, DataLoader
# from termcolor import colored

def read_one_file(s1, s2, id):
    if id<10:
        sub = '0'+str(id)
    else:
        sub = str(id)
    data = io.loadmat(s1+sub+s2)
    HbO = data['HbO']
    HbR = data['HbR']
    HbT = HbO + HbR
    labels = data['labels']
    
    return HbO, HbR, HbT, labels

    
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
    train_HbO, train_HbR, train_HbT, train_labels = read_one_file(s1, s2, id)
    
    for ID in train_id[1:]:
        print(ID,end=', ')
        temp_HbO, temp_HbR, temp_HbT, temp_labels = read_one_file(s1, s2, ID)
        train_HbO = np.concatenate((train_HbO,temp_HbO),axis=0)
        train_HbR = np.concatenate((train_HbR,temp_HbR),axis=0)
        train_HbT = np.concatenate((train_HbT,temp_HbT),axis=0)
        train_labels = np.concatenate((train_labels,temp_labels),axis=0) 
    
    test_id=test_id[0]
    print("test id: ",test_id)
    test_HbO, test_HbR, test_HbT, test_labels = read_one_file(s1, s2, test_id)

    if train_labels.shape[1] == 1 and test_labels.shape[1] == 1 : # no one-hot encoding label
        if min(train_labels)>0 and min(train_labels)==min(test_labels):
            print('modify label')
            test_labels = test_labels-min(train_labels)
            train_labels = train_labels-min(train_labels)
        
        if min(train_labels)!=min(test_labels):
#            print(colored('Training set and test set markings may be wrong!!!', 'red'))
#            print(colored('The minimum values of the training set and test set labels are not equal.', 'red'))
            print('Error: Training set and test set markings may be wrong!!!')
            print('The minimum values of the training set and test set labels are not equal.')

    print("training data dimension: ",train_HbO.shape)
    print("training label dimension: ",train_labels.shape)
    print("testing data dimension: ",test_HbO.shape)
    print("testing label dimension: ",test_labels.shape)

    return test_HbO, test_HbR, test_HbT, test_labels, train_HbO, train_HbR, train_HbT, train_labels


def leave_trial_out(s1, s2, subject_id, range1, range2):
    print("leave trial",len(range1),"out")
    train_HbO, train_HbR, train_HbT, train_labels = read_one_file(s1, s2, subject_id)
    # range1=list(range(0,5,1))
    # range2=list(range(5,75,1))
    
    test_HbO = train_HbO[range1,:,:]
    test_HbR = train_HbR[range1,:,:]
    test_HbT = train_HbT[range1,:,:]
    test_labels = train_labels[range1,:]

    train_HbO = train_HbO[range2,:,:]
    train_HbR = train_HbR[range2,:,:]
    train_HbT = train_HbT[range2,:,:]
    train_labels = train_labels[range2,:]
    
    print("training data dimension: ",train_HbO.shape)
    print("training label dimension: ",train_labels.shape)
    print("testing data dimension: ",test_HbO.shape)
    print("testing label dimension: ",test_labels.shape)
    
    return test_HbO, test_HbR, test_HbT, test_labels, train_HbO, train_HbR, train_HbT, train_labels
    
    

def getDataLoader(x, y, batch_size, onehot_encoding):
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
    

def split_FT(data, FT_num):
  FT_data = data[0:FT_num]
  test_data = data[FT_num:data.shape[0]+1]
  print("fine turning dimension: ", FT_data.shape)
  print("test dimension: ", test_data.shape)
  return FT_data, test_data
  

# def combine_Hb(*Hb):
    # output_data = torch.from_numpy(Hb[0]).float()
    # output_data = output_data.unsqueeze(1)
    # if len(Hb)>1:
        # for i in range(1,len(Hb)):
            # data_temp = torch.from_numpy(Hb[i]).float()
            # data_temp = data_temp.unsqueeze(1)
            # output_data=torch.cat((output_data,data_temp),1)
    # return output_data
    
