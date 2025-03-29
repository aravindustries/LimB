#%%
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import utils2
import pandas as pd
import mlp2
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

df_test = pd.read_csv('../data/spacial_profiles/mid_snr/gp_9.01.csv')

n_beams = np.array([8])
num_classes = 91  # Adjust based on desired resolution

snr_ranges = ['ultra_low_snr', 'low_snr', 'mid_low_snr', 'mid_snr', 'mid_high_snr']

def avg_error_eval(dmlp, scaler, beta, snr_range):
    directory = '../data/spacial_profiles/' + snr_range + '/'
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    avg_err = np.zeros(91)
    count = 0
    for csv_file in csv_files:
        file_path = os.path.join(directory, csv_file)
        df_test = pd.read_csv(file_path)
        X_test = scaler.fit_transform(df_test.iloc[:, 2:65].iloc[:, beta])
        y_actual, y_pred =  dmlp.eval_model(X_test, y_test)
        y_hat = np.subtract(y_pred, 45)
        err = np.abs(y_hat-np.arange(-45,46))
        avg_err = avg_err + err
        count = count + 1

    avg_err = np.divide(avg_err, count)

    return avg_err

for n in n_beams:
    dmlp = mlp2.doaMLPClassifier(n, num_classes)
    y_test = df_test['Angle'].to_numpy()
    y_test = np.digitize(y_test, bins=np.linspace(-45, 45, num_classes)) - 1 
    beta = mlp2.get_beams(n, 60)
    scaler = MinMaxScaler()
    X_test = scaler.fit_transform(df_test.iloc[:, 2:65].iloc[:, beta])

    dmlp.iterative_train(scaler=scaler, N=10)

    plt.figure()

    for snr in snr_ranges: 
        err = avg_error_eval(dmlp, scaler, beta, snr)
        plt.plot(np.arange(-45, 46), err, label=snr)

    plt.legend()
    plt.show()
    plt.savefig('Error{n}.png')
    

# %%
