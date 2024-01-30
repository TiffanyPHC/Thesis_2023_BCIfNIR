# ------------------import package -------------------------------------------
import os
import sys

path = "/home/nycu813/Tiffany"
os.chdir(path)
os.listdir(path)


path_to_this_work = path
path_to_src = path_to_this_work + '/src'


sys.path.insert(0, path_to_src)
if 'PYTHONPATH' in os.environ:
  os.environ['PYTHONPATH'] += (":"+path_to_src)
  
from BCI_model_class import SCCNet, SCCNet_25s, ShallowConvNet, ShallowConvNet2, fNIRSNet
from other_model import NIRS_CNN,NIRS_ANN, NIRS_LSTM, LSTM_getDataLoader
from transformer import Residual, PreNorm, FeedForward, Attention, Transformer, PreBlock, fNIRS_T, LabelSmoothing, train_transformer
import Dataset_training_schema as dts
import train_model as tm

from torchsummary import summary
import math
import numpy as np
import torch
import wandb
import time
from sklearn.model_selection import KFold, StratifiedKFold

# ------------------ wandb login -------------------------------------------
wandb.login(key="550fa288f18a4938b6519ecd67d11309cfd9d51e")


# GPU check
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# --------------

class parameter:
    def __init__(self, architecture, batch_size, kernel_size, learning_rate, weight_decay):
        self.architecture = architecture  
        self.batch_size = batch_size  
        self.kernel_size = kernel_size  
        self.learning_rate = learning_rate  
        self.weight_decay = weight_decay

parameter_set={
    'SCCNet_h1' : parameter('SCCNet',64,134,0.001,0.00001),
    'SCCNet_h2' : parameter('SCCNet',16,6,0.01,0.00001),
    'SCCNet_h3' : parameter('SCCNet',32,12,0.01,0.00001),
    'EEGNet_h1' : parameter('EEGNet',16,32,0.001,0.001),
    'EEGNet_h2' : parameter('EEGNet',32,134,0.001,0.0001),
    'EEGNet_h3' : parameter('EEGNet',32,24,0.001,0.0001),
    'EEGNet_h4' : parameter('EEGNet',64,64,0.001,0.0001)
}





# The sweep calls this function with each set of hyperparameters
def dataloader_model_train(test, train, config, model=0, rep_ID=""):
    rep_ID = str(rep_ID)
    channel_num = train.HbO.shape[1]
    n_Hb=1

    if config.Hb=='HbO':
        print("Hb type : HbO")
        train_loader = dts.getDataLoader(train.HbO, train.labels, batch_size=config.batch_size, onehot_encoding=False)
        test_loader = dts.getDataLoader(test.HbO, test.labels, batch_size=config.batch_size, onehot_encoding=False)
    elif config.Hb=='HbR':
        print("Hb type : HbR")
        train_loader = dts.getDataLoader(train.HbR, train.labels, batch_size=config.batch_size, onehot_encoding=False)
        test_loader = dts.getDataLoader(test.HbR, test.labels, batch_size=config.batch_size, onehot_encoding=False)
    elif config.Hb=='HbT':
        print("Hb type : HbT")
        train_loader = dts.getDataLoader(train.HbT, train.labels, batch_size=config.batch_size, onehot_encoding=False)
        test_loader = dts.getDataLoader(test.HbT, test.labels, batch_size=config.batch_size, onehot_encoding=False)
    elif config.Hb=='HbO+HbR_40channel':
        print("Hb type : HbO+HbR_40channel")
        Hb_test = np.concatenate((np.expand_dims(test.HbO,1), np.expand_dims(test.HbR,1)), axis=2)
        Hb_train = np.concatenate((np.expand_dims(train.HbO,1), np.expand_dims(train.HbR,1)), axis=2)
        train_loader = dts.getDataLoader(Hb_train, train.labels, batch_size=32, onehot_encoding=False)
        test_loader = dts.getDataLoader(Hb_test, test.labels, batch_size=32, onehot_encoding=False)
        channel_num = channel_num*2
    elif config.Hb=='HbO+HbR_2layer':
        print("Hb type : HbO+HbR_2layer")
        Hb_test = np.concatenate((np.expand_dims(test.HbO,1), np.expand_dims(test.HbR,1)), axis=1)
        Hb_train = np.concatenate((np.expand_dims(train.HbO,1), np.expand_dims(train.HbR,1)), axis=1)
        print(Hb_train.shape)
        train_loader = dts.getDataLoader(Hb_train, train.labels, batch_size=32, onehot_encoding=False)
        test_loader = dts.getDataLoader(Hb_test, test.labels, batch_size=32, onehot_encoding=False)
        n_Hb=2

    my_lr = config.learning_rate if model==0 else config.learning_rate/10
    my_epoch = config.epochs if model==0 else math.ceil(config.epochs/10)
    if model==0:
      # model = fNIRSNet(config.kernel_size)
      if config.architecture=="EEGNet":
          # model = fNIRSNet(conv1_size=config.kernel_size, n_Hb=1, time_point=train_HbO.shape[2], channel_num =20,  acti_fun='elu')
          model = fNIRSNet(conv1_size=config.kernel_size, sep_conv_size=config.sep_conv_size, n_Hb=n_Hb, time_point=train.HbO.shape[2], channel_num=channel_num, acti_fun=config.acti_fun,  pooling_type = config.pooling_type, class_num=test.labels.shape[1], con1_type=config.con1_type)
          model=model.to(device)
          summary(model, (n_Hb, channel_num, train.HbO.shape[2]))
      elif config.architecture=="NIRS_CNN":
          model = NIRS_CNN(Hb_num=n_Hb, ch=train.HbO.shape[1], time_point=train.HbO.shape[2], class_num=train.labels.shape[1])
          model=model.to(device)
          summary(model, (n_Hb, channel_num, train.HbO.shape[2]))
      elif config.architecture=="NIRS_transformer":
          model = fNIRS_T(n_class=train.labels.shape[1], sampling_point= train.HbO.shape[2], dim=128, depth=6, heads=8, mlp_dim=64).to(device)
      elif config.architecture=="NIRS_ANN": 
          model = NIRS_ANN(Hb_num=n_Hb, ch=train.HbO.shape[1], time_point=train.HbO.shape[2], class_num=train.labels.shape[1])
          model=model.to(device)
          summary(model, (n_Hb, train.HbO.shape[1], train.HbO.shape[2]))
          # summary(model, (1, train_HbO.shape[1]))
      elif config.architecture=="NIRS_LSTM":
          # model = FNIRSLSTM(input_size, hidden_size, num_classes)
          model = NIRS_LSTM()
          train_HbO_lstm = train.HbO.transpose(0, 2, 1)
          test_HbO_lstm = test.HbO.transpose(0, 2, 1)
          train_loader = LSTM_getDataLoader(train_HbO_lstm, train.labels, batch_size=8, onehot_encoding=False)
          test_loader = LSTM_getDataLoader(test_HbO_lstm, test.labels, batch_size=8, onehot_encoding=False)
      else:
          if config.duration=='10s':
              if config.architecture=="SCCNet":
                  model = SCCNet(config.kernel_size)
              elif config.architecture=="ShallowConvNet":
                  model = ShallowConvNet()
              model=model.to(device)
              summary(model, (1, 20, 134))
          elif config.duration=='25s':
              if config.architecture=="SCCNet":
                  model = SCCNet_25s(config.kernel_size)
              elif config.architecture=="ShallowConvNet":
                  model = ShallowConvNet()
              model=model.to(device)
              summary(model, (1, 20, 334))



    if config.architecture=="NIRS_transformer":
        return train_transformer(model,train_loader,test_loader,sub=config.sub,epoch=my_epoch,rep_ID = rep_ID)
    else:
        return tm.test_EEG_kernel(train_loader,test_loader,
                model=model,
                optimizer=config.optimizer,
                # kernel_size=config.kernel_size,
                epoch=my_epoch,
                learning_rate=my_lr,
                weight_decay=config.weight_decay,
                wandb_import=True,
                rep_ID = rep_ID)

def train():

  wandb.init()
  # Config is a variable that holds and saves hyperparameters and inputs
  config = wandb.config
  
  path_dataset='./dataset/'+ config.dataset + '_' + config.duration + '/'
  s1=path_dataset+'S'
  s2='.mat'
  

  if config.parameter_tag in parameter_set.keys():
    p = parameter_set[config.parameter_tag]
    config.architecture = p.architecture
    config.batch_size = p.batch_size
    config.kernel_size = p.kernel_size
    config.learning_rate = p.learning_rate
    config.weight_decay = p.weight_decay

  if config.scheme == 'SI' or config.scheme == 'SIFT': #------------------modify
      if (config.dataset=='MA') | (config.dataset=='MI') | (config.dataset=='MA_raw') | (config.dataset=='MI_raw'):
        # test_HbO, test_HbR, test_HbT, test_labels, train_HbO, train_HbR, train_HbT, train_labels = dts.leave_subject_out(s1,s2, data_range=range(1,30), test_id=int(config.sub))
        test, train = dts.leave_subject_out(s1,s2, data_range=range(1,30), test_id=int(config.sub))
      else:
        test, train = dts.leave_subject_out(s1,s2, data_range=range(1,31), test_id=int(config.sub))
  elif config.scheme == 'SD':
      if (config.dataset=='MA') | (config.dataset=='MI') | (config.dataset=='MA_raw') | (config.dataset=='MI_raw'):
        range1 = []
        range2 = list(range(0,60,1))
      else:
        range1 = []
        range2 = list(range(0,75,1))
      test, train = dts.leave_trial_out(s1, s2, int(config.sub), range1, range2)
  print(train.HbO.shape)
  #--------------------------------------------------
  # intput data
  if config.scheme == 'SI' : #------------------modify
      model, criterion, optimizer = dataloader_model_train(test, train, config)
  elif config.scheme == 'SD':
      skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=None)
      skf.get_n_splits()
      print(skf) # StratifiedKFold(n_splits=2, random_state=None, shuffle=False)
      # get train set and test set index
      for rep_ID, (train_index, test_index) in enumerate(skf.split(train.HbO, np.argmax(train.labels, axis=1))):
          #print("train_index: ",train_index)
          print("test_index: ",test_index)
          sd_test = dts.fnirs_data(HbO = train.HbO[test_index], HbR = train.HbR[test_index], HbT = train.HbT[test_index], labels = train.labels[test_index])
          sd_train = dts.fnirs_data(HbO = train.HbO[train_index], HbR = train.HbR[train_index], HbT = train.HbT[train_index], labels = train.labels[train_index])
          dataloader_model_train(sd_test, sd_train, config, rep_ID=rep_ID)
  elif config.scheme == 'SIFT':
      model, criterion, optimizer = dataloader_model_train(test, train, config)

      skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=None)
      skf.get_n_splits()
      print(skf) # StratifiedKFold(n_splits=2, random_state=None, shuffle=False)
      # get train set and test set index
      for rep_ID, (val_index, trainFT_index) in enumerate(skf.split(test.HbO, np.argmax(test.labels, axis=1))):
          print("trainFT_index: ",trainFT_index)
          FT_trainFT = dts.fnirs_data(HbO = train.HbO[trainFT_index], HbR = train.HbR[trainFT_index], HbT = train.HbT[trainFT_index], labels = train.labels[trainFT_index])
          FT_val = dts.fnirs_data(HbO = train.HbO[val_index], HbR = train.HbR[val_index], HbT = train.HbT[val_index], labels = train.labels[val_index])
          dataloader_model_train(FT_val, FT_trainFT, config, model=model, rep_ID=rep_ID)



  with open('finish_time.txt', 'a') as f:
          seconds = time.time()
          local_time = time.ctime(seconds)
          f.write('FINISH!! ')
          f.write(local_time)
          f.write('\n')

if len(sys.argv)>=3:
  projectName =  str(sys.argv[1])
  i = 2
  while i<len(sys.argv):
    sweepID = str(sys.argv[i])
    sweep_agent = 'cphnycu/' + projectName + '/' + sweepID
    wandb.agent(sweep_agent, train)
    # wandb.agent(sweep_agent, train, count=1)
    i=i+1
else:
  print('empty')
