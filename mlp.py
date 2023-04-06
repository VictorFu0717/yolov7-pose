import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
import torch.utils.data as data
from torchvision import datasets
import configparser
import random
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

config = configparser.ConfigParser()    # 注意大小寫
config.read("label.ini")   # 配置檔案的路徑

ran = []
for i in range(200):
    r = random.randint(0,1903)
    if r not in ran:
        ran.append(r)
print(ran)


x = np.array(eval(config['main']['label_x']))
y = np.array(eval(config['main']['label_y']))

x_test = x[ran]
y_test = y[ran]
print(len(x_test))
print(len(y_test))

x_train = np.delete(x, ran, axis=0)
y_train = np.delete(y, ran)

print(len(x_train))
print(x_train.shape)
print(len(y_train))


# 轉換資料為PyTorch Tensor格式
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)
x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
print(x_train)

# 將資料集組合成TensorDataset
train_dataset = TensorDataset(x_train, y_train)

# 定義訓練批次大小和數據加載器
batch_size = 24
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

import torch
import torch.nn as nn
import torch.optim as optim

# 定義 MLP 模型
# class MLP(nn.Module):
#     def __init__(self):
#         super(MLP, self).__init__()
#         self.fc1 = nn.Linear(17*2, 64)
#         self.fc2 = nn.Linear(64, 32)
#         self.fc3 = nn.Linear(32, 1)
#         self.relu = nn.ReLU()
#         self.sigmoid = torch.nn.Sigmoid()
#
#     def forward(self, x):
#         x = x.view(-1, 17*2)
#         x = self.relu(self.fc1(x))
#         x = self.relu(self.fc2(x))
#         x = self.fc3(x)
#         x = self.sigmoid(x)
#         return x

# 定義 MLP 模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(17*2, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.fc5 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1, 17*2)
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.relu(self.bn4(self.fc4(x)))
        x = self.fc5(x)
        x = self.sigmoid(x)
        return x




# # 初始化模型和優化器
# model = MLP()
# model.train()  # model.train()的作用是啟用Batch Normalization 和Dropout。
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#
# # 定義損失函數
# criterion = nn.MSELoss()
#
# # 訓練模型
# min_loss = float('inf')
# for epoch in range(100):
#     running_loss = 0.0
#     for i, data in enumerate(train_loader, 0):
#         inputs, labels = data
#         optimizer.zero_grad()  # 將所有參數的梯度緩衝區（buffer）歸零
#
#         outputs = model(inputs)
#         loss = criterion(outputs.squeeze(), labels)
#         loss.backward()   # 進行反向傳播
#         optimizer.step()  # 更新權重
#
#         running_loss += loss.item()
#
#     epoch_loss = running_loss / len(train_loader)
#     print('Epoch %d loss: %.3f' % (epoch + 1, epoch_loss))
#
#     # 如果當前的loss比最小值還小，就保存當前的模型參數
#     if epoch_loss < min_loss:
#         min_loss = epoch_loss
#         torch.save(model.state_dict(), 'min_loss_model.pt')
#
#
# # 載入已儲存的模型
# model = MLP()
# model.load_state_dict(torch.load('min_loss_model.pt'))
# model.to(0)
# model.eval()
#
# # 輸入新的資料進行推論
# test_data = [[0.46640625, 0.3268229166666667], [0.4765625, 0.3151041666666667], [0.4703125, 0.3098958333333333], [0.50625, 0.3333333333333333], [0.51015625, 0.31640625], [0.4765625, 0.4348958333333333], [0.553125, 0.39453125], [0.4375, 0.5520833333333334], [0.5828125, 0.4322916666666667], [0.378125, 0.5481770833333334], [0.56875, 0.3502604166666667], [0.49921875, 0.6783854166666666], [0.5421875, 0.66015625], [0.5, 0.8333333333333334], [0.521875, 0.7981770833333334], [0.5234375, 0.9596354166666666], [0.5046875, 0.91015625]]
# test_data = torch.FloatTensor(test_data)
#
# prediction = model(test_data.to(0))
# prediction = prediction.cpu()
# print(prediction.detach().numpy())
