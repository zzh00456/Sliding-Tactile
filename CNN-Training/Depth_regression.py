#!/usr/bin/env python
# coding: utf-8

import os
import torch
from torchvision import transforms, datasets,utils
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import random_split,Subset
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from torchsummary import summary

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 100
batch_size = 32
learning_rate = 0.0001
                               
transform = transforms.Compose([
    transforms.Resize(106),    
    transforms.CenterCrop(106),   
#     transforms.Grayscale(),
    transforms.ToTensor(),          
#     transforms.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5]) #[-1,1]
]) 


class MyDataset:
    def __init__(self, img_path, transform=None):
        super(MyDataset, self).__init__()
        self.root = img_path
 
        self.txt_root = self.root + 'label.txt'
        f = open(self.txt_root, 'r')
        data = f.readlines()
 
        imgs = []
        labels = []
        for line in data:
            line = line.rstrip()
            word = line.split()
#             imgs.append(os.path.join(self.root, word[1], word[0]))
            imgs.append(os.path.join(self.root, word[0]))
#             print(imgs)
            labels.append(word[3])
#             print(labels)
        self.img = imgs
        print(imgs[0])
        self.label = labels
        self.transform = transform
 
    def __len__(self):
        return len(self.label)
 
    def __getitem__(self, item):
        img = self.img[item]
        label = self.label[item]
        img = Image.open(img).convert('L')
 
        if transform is not None:
            img = self.transform(img).to(device)
 
        label = np.array(label).astype(np.float32)
        label = torch.from_numpy(label).to(device)
        
        return img, label
    
# In[27]:


path = r'D:/JupyterNotebook/dataset_cut/'
dataset = MyDataset(path, transform=transform)


dataSet_length = len(dataset)
print(dataSet_length)
train_size = int(0.8*len(dataset))
print(str(train_size))
test_size = val_size = (len(dataset)-train_size)      
print(str(val_size))

train_dataset,test_dataset = torch.utils.data.random_split(dataset,[train_size,test_size])
print(len(train_dataset))
print(len(test_dataset))


data_loader = DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)
# test_loader = DataLoader(dataset=test_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=293, shuffle=True)


# Convolutional neural network (two convolutional layers)

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(#Out=(height_in-height_kernel+2*padding)/stride+1
            # (106-6+2*0)//2+1 =51     #padding
            nn.Conv2d(1, 16, kernel_size=6, stride=2, padding=0),#
            nn.BatchNorm2d(16),#
            nn.ReLU(),
            # (51-3+2*1)//2+1 = 26
            nn.MaxPool2d(kernel_size=3, stride=2,padding=1))  #kernel_size, stride 
        self.layer2 = nn.Sequential(
            # (26-4+2*1)//2+1 = 13
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # (13-3)//2+1 = 6
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.layer3 = nn.Sequential(
            # (6-3+2*1)//1+1 = 6
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # (6-2)//2+1 = 3
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Linear(3 * 3 * 64, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


model = ConvNet().to(device)
summary(model, input_size=(1,106,106)) 


# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.SGD(model.parameters(), lr = 0.001)
# Train the model
total_step = len(data_loader)
train_loss_all = []
for epoch in range(num_epochs):
    for i, (img, label) in enumerate(data_loader):
        
        img = img

        label = label.to(torch.float32)
        label = label
        
        train_loss = 0
        train_num = 0
        
        # Forward pass
        outputs = model(img)
        loss = criterion(outputs, label.view([-1, 1]))

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss +=loss.item()*img.size(0)  
        train_num += img.size(0)
        if (i + 1) % 3 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
    train_loss_all.append(train_loss / train_num)
    print(train_loss_all)


plt.figure(figsize = (8, 6))
plt.plot(train_loss_all, 'ro-', label = 'Train loss')
plt.legend()
plt.grid()
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.show()

a = np.array(train_loss_all)
np.savetxt('./dataset_cut_csv/depth_epoch_csv.csv', a, delimiter=",")
# plot_save_csv(train_loss_all,'force_epoch_csv',0)

# Save the model checkpoint
torch.save(model.state_dict(), 'Model_Depth_regression.ckpt')







#load and test

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            # (106-6+2*0)//2+1 =51     
            nn.Conv2d(1, 16, kernel_size=6, stride=2, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # (51-3+2*1)//2+1 = 26
            nn.MaxPool2d(kernel_size=3, stride=2,padding=1))  
        self.layer2 = nn.Sequential(
            # (26-4+2*1)//2+1 = 13
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # (13-3)//2+1 = 6
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.layer3 = nn.Sequential(
            # (6-3+2*1)//1+1 = 6
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # (6-2)//2+1 = 3
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Linear(3 * 3 * 64, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

model = ConvNet().to(device)
model.load_state_dict(torch.load('Model_Depth_regression.ckpt'))
model.eval()



# Test the model

model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)

        labels = labels.to(torch.float32)
#         labels = labels.to(device)
        labels = labels.cpu()

        outputs = model(images)
        outputs = outputs.squeeze()
#         outputs = torch.round(outputs,2)
        outputs = outputs.cpu().data.numpy()
        print(labels)
        print(outputs)
        print(labels - outputs)
#         mae = mean_absolute_error(labels, outputs)
#         print('在测试集上的绝对值误差为:', mae)
        print("Mean squared error(MSE):",mean_squared_error(labels,outputs))
        print("Root Mean Square Error(RMSE):",mean_absolute_error(labels,outputs))
        print("Average absolute value error(MAE):",r2_score(labels,outputs))
#         total += labels.size(0)
#         correct += (outputs == labels).sum().item()

#     print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))


labels_array = np.array(labels)
outputs_array=np.array(outputs)
all_array=(labels_array,outputs_array)
all_array=all_array
print(all_array)
np.savetxt('./Original_Prediction/depth_OP_csv.csv', all_array, delimiter=",")


#Visualize the true set and predicted values
index = np.argsort(labels)
plt.figure(figsize=(8, 6))
# plt.plot(np.arange(len(labels)), labels[index], 'r', label = 'Original Y')
plt.scatter(np.arange(len(labels)), labels[index], s = 5, c = 'r', label = 'Original')
plt.scatter(np.arange(len(outputs)), outputs[index], s = 5, c = 'b', label = 'Prediction')
plt.legend(loc = 'upper left')
plt.grid()
plt.xlabel('Index')
plt.ylabel('Width (mm)')
plt.show()


