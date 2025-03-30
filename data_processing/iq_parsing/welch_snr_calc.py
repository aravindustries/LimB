#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch

# Load and process the data as before
df = pd.read_csv('../../../beamtracking17/combined_beam_iq.csv', header=None)
df.drop(df.columns[1], axis=1, inplace=True)

angles = df.iloc[:, 0].values
beams = df.iloc[:, 1].values
complex_data = df.iloc[:, 2:].values

unique_angles = np.unique(angles)
unique_beams = np.unique(beams)
num_values = complex_data.shape[1]

angle_map = {angle: idx for idx, angle in enumerate(unique_angles)}
result = np.zeros((len(unique_angles), len(unique_beams), num_values), dtype=complex)

for i, angle in enumerate(unique_angles):
    for j, beam in enumerate(unique_beams):
        mask = (angles == angle) & (beams == beam)
        
        if np.sum(mask) == 1:
            result[i, j, :] = complex_data[mask, :].flatten()
        else:
            result[i, j, :] = complex_data[mask, :][0]

def get_psd(iq_data): 

    f, psd = welch(iq_data, fs=1, nperseg=256)

    return f, psd


f, psd = get_psd(result[10, 10, :])
plt.semilogy(f, psd)
plt.show()



# %%
