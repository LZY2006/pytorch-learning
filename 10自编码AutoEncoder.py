#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import tqdm

EPOCH = 15
BATCH_SIZE = 128
LR = 0.001
DOWNLOAD_MNIST = True

train_data = torchvision.datasets.MNIST(root='./mnist',
                                       train=True,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=DOWNLOAD_MNIST)
train_loader = Data.DataLoader(dataset=train_data, 
                               batch_size=BATCH_SIZE,
                              shuffle=True)


# In[2]:


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.Tanh(),
            
            nn.Linear(128, 64), 
            nn.Tanh(),
            
            nn.Linear(64, 12),
            nn.Tanh(),
            
            nn.Linear(12, 3),
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.Tanh(),
            
            nn.Linear(12, 64), 
            nn.Tanh(),
            
            nn.Linear(64, 128),
            nn.Tanh(),
            
            nn.Linear(128, 28*28),
            nn.Sigmoid(),
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


# In[3]:


autoencoder = AutoEncoder()
autoencoder.cuda()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
loss_func = nn.MSELoss()


# In[6]:


for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        b_x = Variable(x.view(-1, 28*28)).cuda()
        b_y = Variable(x.view(-1, 28*28)).cuda()
        
        encoded, decoded = autoencoder(b_x)
        
        loss = loss_func(decoded, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(loss.data.cpu().numpy())
    


# In[ ]:




