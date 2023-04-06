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
for i in range(300):
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


class MyDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.long)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return len(self.X)


# 定義神經網路模型
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(16*2, 10)
#         self.fc2 = nn.Linear(10, 10)
#         self.fc3 = nn.Linear(10, 1)
#
#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = torch.sigmoid(self.fc3(x))
#         return x

# 轉換資料為PyTorch Tensor格式
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)
x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
print(x_train)

# 將資料集組合成TensorDataset
train_dataset = TensorDataset(x_train, y_train)

# 定義訓練批次大小和數據加載器
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 定義模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(2, 17, batch_first=True)
        self.fc1 = nn.Linear(17, 8)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc1(x[:, -1, :])
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x

# 初始化模型
model = Net()

# 將模型放在GPU上
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 0
model.to(device)

# 定義優化器和損失函數
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# 訓練模型
num_epochs = 100
min_loss = float('inf')
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
    # 如果當前的loss比最小值還小，就保存當前的模型參數
    if loss.item() < min_loss:
        min_loss = loss.item()
        torch.save(model.state_dict(), 'min_loss_model.pt')

