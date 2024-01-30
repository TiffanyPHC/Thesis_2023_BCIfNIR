# Thesis_2023_BCIfNIR
[my readme link](https://hackmd.io/@pinhuaChen/B1wXyX8qT)

# overview

## 運行邏輯
主程式檔案為`sweep.py`、`Demo.ipynb`，其中包含兩個funciton
- dataloader_model_train
副程式
輸入：測試驗證資料、訓練參數設定、預訓練模型(SIFT專用)
輸出：得到最終訓練結果
- train
是程式進入點，根據各項設定進行模型訓練
輸入：訓練參數設定
輸出：訓練結果
![流程](https://hackmd.io/_uploads/H1aBQQIqa.png)


## src

資料夾src中包含的檔案及其功能如下
| Python 檔案名稱             | 描述|
|:------------------------- |:--------:|
|train_model.py |包含方法test_EEG_kernel，用於模型訓練|
|Dataset_training_schema.py|fNIRS類別的定義、數據載入、取得dataloader
|BCI_model_class.py|定義了不同的BCI神經網絡模型|
|other_model.py|定義了一些額外的模型，如 NIRS_ANN。|

#### Dataset_training_schema.py 內容
- 定義fNIRS類別
   - class fnirs_data
   - HbO : $channel*time$
   - HbR : $channel*time$
   - HbT : $channel*time$
   - label: 經過onehot encoding的結果
- 數據載入
    - read_one_file：讀取一個mat資料
    - leave_subject_out：cross subject/ subject independent根據給定的subject ID，輸出訓練集資料和測試集資料
    - leave_trial_out： within subject/ subject dependent根據給定的subject ID，將該受測者的資料分成訓練集和測試集
- 取得dataloader
    - getDataLoader：訓訓練資料轉為可以放進模型的格式



# 神經網路模型程式執行
## 資料夾位置
主程式檔案位置與src和dataset在同一資料夾

```
.
├── dataset/
│   ├── MA_10s/
│   │   ├── S01.mat
│   │   ├── ...
│   │   └── S29.mat
│   ├── MA_15s/
│   ├── MA_25s/
│   ├── MI_10s/
│   │   ├── S01.mat
│   │   ├── ...  
│   │   └── S29.mat
│   ├── MI_15s/
│   ├── MI_25s/
│   ├── preprocess_10s/
│   │   ├── S01.mat
│   │   ├── ... 
│   │   └── S30.mat
│   ├── preprocess_15s/
│   └── preprocess_25s/
├── src
│   ├── BCI_model_class.py
│   ├── Dataset_training_schema.py
│   ├── other_model.py
│   ├── train_model.py
│   └── transformer.py
├── vistualization_demo.ipynb
└── sweep.py

```
## 參數設定選項


### 訓練資料
|參數名稱|描述|參數選項|
|:--|:--|:--|
|dataset|資料集名稱(讀取[dataset]_[duration]資料夾)|MI、MA、preprocess|
|duration|採用的時間長度|10s、15s、25s|
|sub|受測者編號|根據資料集的受測者編號自行設定EX:”1”|
|Hb|定義採用血氧濃度資料的形式|HbO、HbR、HbT、HbO+HbR_2layer、HbO+HbR_40channel|
### 關於模型
|參數名稱|描述|參數選項|
|:-------|:---|:-------|
|architecture|進行訓練的模型(若使用)|EEGNet、NIRS_ANN、NIRS_CNN、NIRS_transformer|

要是使用EEGNet的架構則需要進行以下設定
|參數名稱|描述|參數選項|
|:-------|:---|:-------|
|con1_type|第一個convolution的種類|conv|
|kernel_size|第一個convolution layer大小|數字。EX: 32|
|sep_conv_size|可分離式捲基層大小|數字。EX:32|
|pooling_type|池化方法|avgPool、maxPool|
|acti_fun|激活函數的選項|relu、elu、linear、square、square1、square2|

### 訓練時的參數
|參數名稱|描述|參數選項|
|:-------|:---|:-------|
|batch_size|訓練時單個batch的大小|數字。Ex: 16|
|epoch|進行訓練的回數|數字。EX:200|
|learning_rate|學習率|數字。EX:0.001|
|optimizer|最佳化方法|adam|
|weight_decay|Weight decay設定|數字。EX:0.0001|
|scheme|驗證方式|SI、SD、SIF|

### 其他視情況增加的超參數
完全不影響模型訓練，而是根據需求增加，參數的格式很自由，只要容易閱讀即可。

|參數名稱|描述|參數選項|
|:-------|:---|:-------|
|parameter_tag|根據目前正在進行麼實驗描述|文字描述|
|rep|實驗需要重複的話，可以用此方式重複實驗|可用1、2…進行描述|


## 執行程式碼
### 1. 確認參數設定
    - 可以利用wandb將所有要訓練的參數組合設定好，並用sweep.py執行
        - 可以參考[Create sweeps from existing projects](https://docs.wandb.ai/guides/sweeps/existing-project)
    - 在DEMO.ipynb中進行設定
    範例如下
```python=1
my_config = config(
  Hb = "HbO+HbR_2layer",
  acti_fun = "elu",
  architecture = "EEGNet",
  batch_size = 16,
  con1_type = "conv",
  dataset = "MA",
  duration = "25s",
  epochs = 10,
  kernel_size = 64,
  learning_rate = 0.001,
  optimizer = "adam",
  pooling_type = "avgPool",
  scheme = "SIFT",
  sep_conv_size = 16,
  sub = "1",
  weight_decay = 0.0001
)
```

### 2. 確認sweep.py中的路徑設定
```python=9
path = "/path/to/sweep.py"
os.chdir(path)
os.listdir(path)
```
### 3. 執行
- 在server上執行
```cmd=1
python sweep.py DEMO_code 2wiyn4ri rnd3pt8n
```
如果有多個sweep要測試可以給定多個sweep id，格式如下
`python sweep.py wandb_project_ID sweep1_ID sweep2_ID`
- 若在google colab上執行，則只需要修改**設定環境路徑**區塊的程式碼即可
```python=1
import sys
import os

path_to_this_work = '/path/to/DEMO.ipynb'
os.chdir(path_to_this_work)
os.listdir(path_to_this_work)
path_to_src = path_to_this_work + '/src'

sys.path.insert(0, path_to_src)
os.environ['PYTHONPATH'] += (":"+path_to_src)
```
### 4. 取得結果(optional)
用於下載wandb上的結果，若是資料量太大，沒有辦法由網頁下載，可以參考[download sweep](https://docs.wandb.ai/guides/track/public-api-guide#querying-multiple-runs)將表格資料下載下來

# 數學模型程式執行
## 參數選項
- Demo_SVM_KNN.ipynb

    |參數名稱|描述|參數選項|
    |:-------|:---|:-------|
    |dataset|資料集名稱(讀取dataset_duration資料夾)|MA 、 MI 、 preprocess|
    |duration|採用的時間長度|15s|
    |method|採用的分類方法|SVM 、 KNN|
    |scheme|驗證方式|SI、SD|
    |sub|受測者編號|“1” ~”30”|

- CSP_LDA.ipynb

    |參數名稱|描述|參數選項|
    |:-------|:---|:-------|
    |dataset|資料集名稱(讀取dataset_duration資料夾)|MA 、 MI 、 preprocess|
    |duration|採用的時間長度|15s|
    |scheme|驗證方式|SI、SD|
    |sub|受測者編號|“1” ~”30”|

## 環境設定
```python=4
path_to_this_work = '/path/to/src'
path_to_src = path_to_this_work + '/src'
```

## 微調程式碼
- 單純在colab上進行測試
    ```python=1
    def main(dataset,duration,sub,scheme):                     

      if (int(sub)>30 and dataset=='preprocess') or (int(sub)>29 and dataset in ['MI','MA']):
        wandb.finish()
        return 0


      path_dataset='./dataset/'+ dataset + '_' + duration + '/'
      s1=path_dataset+'S'
      s2='.mat'
      sub_range = 31 if dataset=='preprocess' else 30

      if scheme=="SI":

        # 檔案讀取
        test, train = dts.leave_subject_out(s1,s2, data_range=range(1,sub_range), test_id=int(sub))
        acc =  runClassifyCSP_LDA(train.HbO, np.argmax(train.labels,axis=1), testData=test.HbO, testLabel=np.argmax(test.labels,axis=1))

        print('ACC: ',acc)
        return 0


      elif scheme=="SD":

        # 檔案讀取
        data = dts.read_one_file(s1,s2,sub)

        acc = runClassifyCSP_LDA(data.HbO,np.argmax(data.labels,axis=1))

        print('ACC: ',acc)
        return 0
    ```
- 利用wandb大量計算時，需要修改完以下部分
    - 開頭修改
    ```python=1
    def main():                                         
      wandb.init()
      # Config is a variable that holds and saves hyperparameters and inputs
      config = wandb.config
      dataset = config.dataset
      duration = config.duration
      sub = int(config.sub)
      scheme = config.scheme
      method = config.method
    ```
    - 其他修正
    於程式碼main區塊的33行和50行加入
    ```python=32
    wandb.log({"Accuracy": acc})
    wandb.finish()
    ```
# 視覺化
完成以下設定後執行程式碼即可
## 環境設定
```python=1
path_to_this_work = '/path/to/this/work'
os.chdir(path_to_this_work)
os.listdir(path_to_this_work)
path_to_src = path_to_this_work + '/src_before20230712'
```
## 特定類別視覺化輸出設定
在熱力圖和大腦地形圖的程式碼區塊
- 若要視覺化第0類
    ```python=1
    input_data = data0
    class_id = 0
    ```
- 若要視覺化第1類
    ```python=1
    input_data = data1
    class_id = 1
    ```
- 若要視覺化第2類
    ```python=1
    input_data = data2
    class_id = 2
    ```

