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

def get_psd(iq_data):
    f, psd = welch(iq_data, fs=1, nperseg=256)
    return f, psd

psds = []

for i, angle in enumerate(unique_angles):
    for j, beam in enumerate(unique_beams):
        mask = (angles == angle) & (beams == beam)
        
        if np.sum(mask) == 1:
            result[i, j, :] = complex_data[mask, :].flatten()
        else:
            result[i, j, :] = complex_data[mask, :][0]
        
        f, psd = get_psd(result[i, j, :]) 
        psds.append(psd)

# Calculate the average PSD
avg_psd = np.mean(psds, axis=0)

# Find the index corresponding to the signal tone frequency (0.4 Hz)
signal_freq_idx = np.argmin(np.abs(f - 0.4))  # Find the index closest to 0.4 Hz

# Estimate the signal power (value at the tone frequency)
signal_power = avg_psd[signal_freq_idx]

# Estimate the noise power (average PSD in a frequency range away from the signal, e.g., below 0.2 Hz or above 0.6 Hz)
# Here, I'm assuming the noise is flat in the region outside the tone
noise_region = np.concatenate([avg_psd[f < 0.2], avg_psd[f > 0.6]])  # Adjust the frequency ranges if needed
noise_power = np.mean(noise_region)

# Compute SNR in dB
snr = 10 * np.log10(signal_power / noise_power)

# Plot the average PSD
plt.semilogy(f, avg_psd)
plt.title('Average PSD')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density (dB/Hz)')
plt.grid(True)
plt.show()

# Print the calculated SNR
print(f"SNR: {snr:.2f} dB")
