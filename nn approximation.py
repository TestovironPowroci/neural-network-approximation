import torch
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader, Dataset
import numpy as np
from matplotlib import pyplot as plt

test_effect = []
validation_effect = []

testing_data = np.loadtxt('testdata.csv', delimiter=',', dtype=np.float32)
validation_data = np.loadtxt('validationdata.csv', delimiter=',', dtype=np.float32)
test_target=(testing_data[:, [-1]])
validation_target=(validation_data[:, [-1]])



epoch = 100
firststart = True #if you want to load previous learning outcomes: state, optimizer --> True

lr=0.01
n_hidden=11

def load_checkpoint(model, optimizer, filename='state.pth'):
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch


class TrainDataset(Dataset):

    def __init__(self):
        xy = np.loadtxt('traindata.csv',
                        delimiter=',', dtype=np.float32)

        self.len = xy.shape[0]
        self.x_datai = torch.from_numpy(xy[:, 0:-1])
        self.y_datai = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_datai[index], self.y_datai[index]

    def __len__(self):
        return self.len


Traindataset = TrainDataset()
train_loader = DataLoader(dataset=Traindataset,
                               batch_size=1,
                               shuffle=True, pin_memory=True)


class TestowDataset(Dataset):
    """ Diabetes dataset."""

    def __init__(self):
        xy = np.loadtxt('testdata.csv',
                        delimiter=',', dtype=np.float32)

        self.len = xy.shape[0]
        self.x_datai = torch.from_numpy(xy[:, 0:-1])
        self.y_datai = torch.from_numpy(xy[:, [-1]])


    def __getitem__(self, index):
        return self.x_datai[index], self.y_datai[index]

    def __len__(self):
        return self.len

Testowdataset = TestowDataset()
testow_loaderTest = DataLoader(dataset=Testowdataset,
                          batch_size=1,
                          shuffle=False, pin_memory=True)

class TestDataset(Dataset):
    """ Diabetes dataset."""

    def __init__(self):
        xy = np.loadtxt('validationdata.csv',
                        delimiter=',', dtype=np.float32)

        self.len = xy.shape[0]
        self.x_datai = torch.from_numpy(xy[:, 0:-1])
        self.y_datai = torch.from_numpy(xy[:, [-1]])


    def __getitem__(self, index):
        return self.x_datai[index], self.y_datai[index]

    def __len__(self):
        return self.len

Testdataset = TestDataset()
test_loader = DataLoader(dataset=Testdataset,
                          batch_size=1,
                          shuffle=False)




class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.sigmoid(self.hidden(x))
        x = self.predict(x)
        return x


net = Net(n_feature=2, n_hidden=n_hidden, n_output=1)
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
loss_func = torch.nn.MSELoss()


if firststart == False:
  load_checkpoint(net, optimizer)
  for g in optimizer.param_groups:
     g['lr']=lr


for t in range(epoch):
  loss_function=0
  loss_sum = 0


  for i, data1 in enumerate(train_loader):
    net.train()
    inputs, target = data1


    prediction = net(inputs)
    loss = loss_func(prediction, target)
    loss_sum+=loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


  loss_function=(loss_sum/(i+1)).item()


  print("loss: ", (round(loss_function,8)),"epoch: ", t+1)








print("------------------------------------uczące----------------------------------------------")
net.eval()
for i, data1 in enumerate(train_loader):
    inputs, target = data1
    prediction = net(inputs)

    loss = loss_func(prediction, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("target: ", round(target.item(),5), "prediction: ", round(prediction.item(),5))
print("------------------------------------test----------------------------------------------")
for i, data1 in enumerate(testow_loaderTest):
    inputs, target = data1
    prediction = net(inputs)

    loss = loss_func(prediction, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    test_effect.append(prediction)
    print("target: ",  round(target.item(),5), "prediction: ",  round(prediction.item(),5))





print("------------------------------------walidacja----------------------------------------------")
for i, data1 in enumerate(test_loader):
    inputs, target = data1
    prediction = net(inputs)

    loss = loss_func(prediction, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    validation_effect.append(prediction)
    print("target: ",  round(target.item(),5), "prediction: ", round(prediction.item(),5))



# the value is multiplied by 10000 due to earlier data normalization for the neural network

plt.title("Wykres dokładności predykcji sieci neuronowej na zbiorze testowym")
plt.xlabel("iteracja")
plt.ylabel("średnica wejściowa[μm]")
plt.plot(np.array(test_effect) * 10000, "ob", label='predykcja', linestyle='-')
plt.plot(test_target * 10000, "go", label='wartość rzeczywista', linestyle='-')
plt.xticks(np.arange(len(test_effect)), np.arange(1, len(test_effect)+1))
plt.xticks(np.arange(len(test_target)), np.arange(1, len(test_target)+1))
leg = plt.legend()
plt.show()


plt.title("Wykres dokładności predykcji sieci neuronowej na zbiorze walidacyjnym")
plt.xlabel("iteracja")
plt.ylabel("średnica wejściowa[μm]")
plt.plot(np.array(validation_effect) * 10000, "ob", label='predykcja', linestyle='-')
plt.plot(validation_target * 10000, "go", label='wartość rzeczywista', linestyle='-')
plt.xticks(np.arange(len(validation_effect)), np.arange(1, len(validation_effect) + 1))
plt.xticks(np.arange(len(validation_target)), np.arange(1, len(validation_target) + 1))
leg = plt.legend()
plt.show()
