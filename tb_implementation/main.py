#%%
# LimB: Learning-driven mmWave Beamforming
# Arav Sharma, Ari Gebhardt, Raymond Chi
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm 
from mlp import doaMLP

def get_beams(n, spread=49):
    return np.linspace(32-(spread/2), 32+(spread/2), n, dtype=int)

print(get_beams(3))

csv_file = "../data/aggregated_training_data/agg_data.csv"
df = pd.read_csv(csv_file, header=None)

# Extract angles and gain profiles
angles = df[0].to_numpy(dtype=np.float32)  # First column: angles
gain_profiles = df.iloc[:, 2:65]

print(gain_profiles)

scaler = MinMaxScaler()
gain_profs = scaler.fit_transform(gain_profiles.iloc[:, get_beams(16)])

print(gain_profs.shape)

X_train, X_test, y_train, y_test = train_test_split(
    gain_profs, angles, test_size=0.05, random_state=42
)

dmlp = doaMLP(X_train.shape[1])

dmlp.train_model(X_train, y_train, epochs=600)

dmlp.eval_model(X_test, y_test)