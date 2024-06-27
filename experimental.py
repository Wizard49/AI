"""More detailed approach on dataset use in ANNs"""
"""typical PyTorch pipeline looks like this: """
# + Design model (input, output, forward pass with different layers)
# + Construct loss and optimizer
# + Training loop:
#    - Forward = compute prediction and loss
#    - Backward = compute gradients
#    - Update weights


import torch
import torch.nn as nn
import torch.optim
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
import os

import warnings
warnings.filterwarnings("ignore")

"""Preprocessing"""


PATH = 'archive/games.csv'
traffic_data = pd.read_csv(PATH)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x_data = traffic_data.drop(['Release date','Genres','Score rank', 'Website', 'AppID', 'Name', 'Reviews', 'Header image', 'Supported languages', 'Full audio languages', 'Screenshots', 'Movies', 'Developers', 'Publishers', 'Tags', 'About the game', 'Support url', 'Support email', 'Metacritic url', 'Notes', 'Categories'], axis=1)
x_data = pd.get_dummies(x_data,columns=['Estimated owners'])
x_data = np.array(x_data, dtype=np.float32)
y_data = np.array(traffic_data['Price'], dtype=np.float32)

x_train,x_test,y_train,y_test= train_test_split(x_data,y_data,test_size=0.2, random_state=42)
samples,features = x_train.shape

x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1,1).to(device)
x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1,1).to(device)

# print(x_train.shape)
# print()
# print(y_train.shape)
# print()
# print(x_test.shape)
# print()
# print(y_test.shape)
# a = torch.tensor(np.linspace(-2,3,100))
# a = a.to(device)

"""Model construction"""

class Model(nn.Module):
    def __init__(self,input_dimensions,output_dimensions):
        super(Model,self).__init__()
        self.l1=nn.Linear(input_dimensions,output_dimensions)
        # self.relu=nn.ReLU()
        # self.l2=nn.Linear(output_dimensions,output_dimensions)

    def forward(self, x):
        out=self.l1(x)
        # out=self.relu(out)
        # out=self.l2(out)
        return out

"""Defining optimizer and loss"""

input_size,output_size=features,features

model = Model(input_size,output_size).to(device)
#print(f'Prediction before training: f({x_test.item()}) = {model(x_test).item():.3f}')
print()

lr=0.01
epochs = 1000

loss=nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=lr)

"""Training"""

model.train()
for epoch in range(epochs):
    y_hat = model(x_train)
    ls = loss(y_train,y_hat)

    optimizer.zero_grad()
    ls.backward()
    optimizer.step()
    # optimizer.zero_grad()

    if epoch%200==0:
        lr/=10

    if (epoch + 1) % 10 == 0:
        w = model.parameters()  # unpack parameters
        print('epoch ', epoch + 1,' loss = ', ls.item())


print('Finally')

model.eval()
#print(f'Prediction after training: f({x_test.item()}) = {model(x_test).item():.3f}')
y_pred=model(x_test).detach().numpy()
y_pred=y_pred.reshape(1,-1)
y_test=y_test.detach().numpy()
y_test=y_test.reshape(1,-1)


print(y_pred,y_test)
