"""Simple approach on dataset use on training simple ANNs"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import warnings

torch.set_default_dtype(torch.float32)

# Load your dataset and preprocess it
PATH = 'archive/games.csv'
file = pd.read_csv(PATH)
file = file.drop(['Score rank', 'Website', 'AppID', 'Name', 'Reviews', 'Header image', 'Supported languages', 'Full audio languages', 'Screenshots', 'Movies', 'Developers', 'Publishers', 'Tags', 'About the game', 'Support url', 'Support email', 'Metacritic url', 'Notes', 'Categories'], axis=1)
file = pd.get_dummies(file, columns=['Estimated owners'])
file = file.drop(['Release date', 'Genres'], axis=1)

# Normalize the target variable (Price)
max_price = max(file['Price'])
y = torch.tensor(file['Price'] / max_price, dtype=torch.float32)

# print(file['Price'].dtype)

x_aux = file.drop(['Price'],axis=1)
x_aux = x_aux.astype(float)
x_aux = x_aux.values
X = torch.tensor(x_aux, dtype=torch.float32)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define your model
class Model(nn.Module):
    def __init__(self, in_dims, out_dims=1):
        super(Model, self).__init__()
        self.l1 = nn.Linear(in_features=in_dims, out_features=out_dims)

    def forward(self, x):
        return self.l1(x)

# Initialize the model
samps, feats = X.shape
model = Model(in_dims=feats).to(device)

# Loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(500):
    optimizer.zero_grad()
    pred_y = torch.tensor(model(X), dtype=torch.float16).to(device)
    ls = loss_fn(pred_y, y)
    ls.backward()
    optimizer.step()
    print(f'Epoch {epoch}, Loss: {ls.item()}')
