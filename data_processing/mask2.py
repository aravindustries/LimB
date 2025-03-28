#%%
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

scaler = MinMaxScaler()
pows = [316, 700, 800, 1778, 10000, 20000]

avg_gains = np.zeros((91, 63))

for pow in pows:
    csv_file = 'plot3d/filtered_data_power_' + str(pow) + '.csv'
    df = pd.read_csv(csv_file)
    print(csv_file)
    print(df.shape)
    print(df.head)
    print()
    print()