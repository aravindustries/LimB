#%%
import os

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.pyplot as plt
import utils2
import pandas as pd
import mlp
import numpy as np
from sklearn.preprocessing import MinMaxScaler

df_train = pd.read_csv('../../data_processing/train_gain_prof.csv')


beam_champ = [3, 4, 6, 8, 12, 16]

#n_beams = 3

for n_beams in beam_champ:

    dmlp = mlp.doaMLP(n_beams)
    beta = mlp.get_beams(n_beams, 50)
    scaler = MinMaxScaler()

    dmlp.iterative_train(df_train, scaler, 10)

    plt.figure(figsize=(8, 6))

    for folder in ['ultra_low_snr', 'low_snr', 'mid_low_snr', 'mid_snr', 'mid_high_snr']:
    # for folder in ['mid_high_snr']:
        y_err = np.zeros(91)

        for file in os.listdir(f'../../data/spacial_profiles/{folder}'):
            fp = os.path.join(f'../../data/spacial_profiles/{folder}', file)

            df_test = pd.read_csv(fp)
            X_test = scaler.fit_transform(df_test.iloc[:, 2:65].iloc[:,beta])
            y_test = df_test['Angle'].to_numpy()

            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

            dmlp.model.eval()
            with torch.no_grad():
                y_pred = dmlp.model(X_test_tensor)
                test_loss = dmlp.criterion(y_pred, y_test_tensor).item()
                mae = torch.mean(torch.abs(y_pred - y_test_tensor)).item()

            # print(f"Test Loss (MSE): {test_loss:.4f}")
            # print(f"Test MAE: {mae:.4f}")

            # Convert predictions to numpy for visualization
            y_pred_numpy = y_pred.numpy().flatten()
            y_test_numpy = y_test_tensor.numpy().flatten()
            # plt.plot(y_pred_numpy)
            # plt.title(file)
            # plt.savefig(f'a_lot_of_plots/{folder}_{file}.png')
            # plt.cla()
            y_err += np.abs(y_pred_numpy - y_test_numpy)

        y_err /= len(os.listdir(f'../../data/spacial_profiles/{folder}'))

        # Plot actual vs predicted angles
        plt.plot(np.arange(-45, 46), y_err, label=folder)

    plt.xlabel("Angle of Arrival (°)")
    plt.ylabel("Error (°)")
    title = "Angle of Arrival Estimation Error for " + str(n_beams) + " Scan Beams"
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(title)

# %%
