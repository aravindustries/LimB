#%%
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.pyplot as plt
import utils2
import pandas as pd
import mlp
import numpy as np
from sklearn.preprocessing import MinMaxScaler

df_train = pd.read_csv('../data_processing/train_gain_prof.csv')
df_test = pd.read_csv('../data/spacial_profiles/gp_10.csv')

n_beams = np.array([3, 4, 6, 8, 12, 16])

for n in n_beams:
    dmlp = mlp.doaMLP(n)
    y_test = df_test['Angle'].to_numpy()
    beta = mlp.get_beams(n, 50)
    scaler = MinMaxScaler()
    X_test = scaler.fit_transform(df_test.iloc[:, 2:65].iloc[:,beta])

    dmlp.iterative_train(df_train, scaler, 100)

    dmlp.eval_model(X_test, y_test)