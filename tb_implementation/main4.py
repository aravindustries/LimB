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

from agile import Agile

df_train = pd.read_csv('../data_processing/train_gain_prof.csv')


boolean = False

n_beams = 4 if boolean else 12
dmlp = mlp.doaMLP(n_beams)
beta = mlp.get_beams(n_beams, 50)
scaler = MinMaxScaler()


dmlp.iterative_train(df_train, scaler, 10)

angles = np.hstack([[0], np.linspace(-45, 45, 62)])
indexes = np.round(np.linspace(11, 52, n_beams)).astype(int)
agile = Agile(angles[indexes])

corr_results = {}
mlp_results = {}

dict = {
    'low_snr': 'Low SNR',
    'mid_high_snr': 'Mid-High SNR',
    'mid_low_snr': 'Mid-Low SNR',
    'mid_snr': 'Mid SNR',
    'ultra_low_snr': 'Ultra-Low SNR',
}

colors = {
    'low_snr': 'blue',
    'mid_high_snr': 'green',
    'mid_low_snr': 'red',
    'mid_snr': 'Mid SNR',
    'ultra_low_snr': 'Ultra-Low SNR',
}

# plt.figure(figsize=(16, 6))
# plt.plot([-45, 45], [0, 0], 'k--')

# print(f"B={n_beams}")

for folder in ['low_snr', 'mid_low_snr', 'mid_high_snr']:
# for folder in ['mid_low_snr']:
    y_err = np.zeros(91)
    y_err_agile = np.zeros(91)

    for file in os.listdir(f'../data/spacial_profiles/{folder}'):
        fp = os.path.join(f'../data/spacial_profiles/{folder}', file)

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
        y_err += (y_pred_numpy - y_test_numpy)

        y_err_agile += (agile.call(X_test) - y_test_numpy)

    y_err /= len(os.listdir(f'../data/spacial_profiles/{folder}'))
    # print(f"{folder}: {y_err[15:-15].mean()}")
    y_err_agile /= len(os.listdir(f'../data/spacial_profiles/{folder}'))

    mlp_results[folder] = y_err
    corr_results[folder] = y_err_agile


# breakpoint()

title_size = 17
other_size = 15

if boolean:
    fig, axes = plt.subplots(2, 1, figsize=(10, 16))

    axes[0].plot([-45, 45], [0, 0], 'k--')
    for plot in ['low_snr', 'mid_low_snr', 'mid_high_snr']:
        axes[0].plot(np.arange(-45, 46), mlp_results[plot], label=dict[plot], linewidth=2, color=colors[plot])
    axes[0].tick_params(labelsize=other_size)
    axes[0].set_xlabel("Direction of Arrival (°)", fontsize=other_size)
    axes[0].set_ylabel("Error (°)", fontsize=other_size)
    axes[0].set_ylim(-65, 65)
    axes[0].set_title("BeamSeek MLP", fontsize=title_size)
    axes[0].legend(fontsize=other_size)
    axes[0].grid(True)


    axes[1].plot([-45, 45], [0, 0], 'k--')
    for plot in ['low_snr', 'mid_low_snr','mid_high_snr']:
        axes[1].plot(np.arange(-45, 46), corr_results[plot], label=dict[plot], linewidth=2, color=colors[plot])
    axes[1].tick_params(labelsize=other_size)
    axes[1].set_xlabel("Direction of Arrival (°)", fontsize=other_size)
    axes[1].set_ylabel("Error (°)", fontsize=other_size)
    axes[1].set_ylim(-65, 65)
    axes[1].set_title("Correlation-Based Method", fontsize=title_size)
    axes[1].legend(fontsize=other_size)
    axes[1].grid(True)

    plt.tight_layout()
    fig.subplots_adjust(top=0.93, hspace=0.17, left=0.1, right=0.95, bottom=0.1)
    # fig.suptitle(f'Direction of Arrival Estimation Error With {n_beams} Beams', fontsize=17)

    plt.savefig(f"Figure_4.png")
else:
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.plot([-45, 45], [0, 0], 'k--')

    ax.plot(np.arange(-45, 46), mlp_results['mid_low_snr'], label='MLP', linewidth=2, color='red')
    ax.plot(np.arange(-45, 46), corr_results['mid_low_snr'], label='Correlation', linewidth=2, color='blue')

    ax.tick_params(labelsize=other_size)
    ax.set_xlabel("Direction of Arrival (°)", fontsize=other_size)
    ax.set_ylabel("Error (°)", fontsize=other_size)
    ax.set_ylim(-65, 65)
    # fig.suptitle(f"Direction of Arrival Estimation Error with {n_beams} Beams and Mid-Low SNR", fontsize=17)
    ax.legend(fontsize=other_size)
    ax.grid(True)

    plt.tight_layout()
    fig.subplots_adjust(top=0.93, hspace=0.17, left=0.1, right=0.95, bottom=0.1)

    plt.savefig("Figure_5.png")
