
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

def get_beams(n, spread=49):
    return np.linspace(32-(spread/2), 32+(spread/2), n, dtype=int)

n = 4

device = torch.device("cuda")

df_train = pd.read_csv('../data_processing/train_gain_prof.csv')

df_test = pd.read_csv('../data/spacial_profiles/gp_10.csv')

dmlp = mlp.doaMLP(n)

y_test = df_test['Angle'].to_numpy()

beta = get_beams(n, 50)

scaler = MinMaxScaler()
X_test = scaler.fit_transform(df_test.iloc[:, 2:65].iloc[:,beta])

#def iterative_train(df_train, scaler, N):
#    #print(df_train.head())
#    for k in range(N):
#        ndf, snr = utils2.adjust_noise_to_target_snr(df_train, np.random.uniform(-5, 20))
#        y_train = ndf['Angle'].to_numpy()
#        X_train = scaler.fit_transform(ndf.iloc[:, 2:65].iloc[:, beta])
#        loss = dmlp.train_model(X_train, y_train, epochs=300)
#        print(f"LOSS {loss} FOR ITERATION {k}")

dmlp.iterative_train(df_train, scaler, 100)
dmlp.eval_model(X_test, y_test)
