#%%
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.pyplot as plt
#import utils
import pandas as pd
import mlp
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def get_beams(n, spread=49):
    return np.linspace(32-(spread/2), 32+(spread/2), n, dtype=int)

n = 8

device = torch.device("cuda")

df_train = pd.read_csv('../data_processing/power_800_data.csv')

df_test = pd.read_csv('../data_processing/power_700_data.csv')

dmlp = mlp.doaMLP(n)

y_train = df_train['Angle'].to_numpy()
y_test = df_test['Angle'].to_numpy()

beta = get_beams(n, 50)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(df_train.iloc[:, 2:65].iloc[:, beta])
X_test = scaler.fit_transform(df_test.iloc[:, 2:65].iloc[:,beta])
#print(beta)
#print(gain_profs.shape)
#
#ind = np.arange(0, n)
#print(ind)
#
#for i in range(n):
#    plt.plot(ind, gain_profs[beta[i]], label=str(i))
#plt.legend()
#plt.savefig('single gain prof')

dmlp.train_model(X_train, y_train, epochs=100)

dmlp.eval_model(X_test, y_test)

# %%
