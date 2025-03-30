#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch

df = pd.read_csv('../../../beamtracking17/combined_beam_iq.csv', header=None)

print(df.head(14))

df.drop(df.columns[1], axis=1, inplace=True)

print(df.iloc[45])

angles = df.iloc[:, 0].values  # First column (angle)
beams = df.iloc[:, 1].values   # Second column (beam)
complex_data = df.iloc[:, 2:].values  # Complex data values starting from the third column

print(angles)
print(beams)
print(complex_data.shape)

# Get the unique angles and beams
unique_angles = np.unique(angles)
unique_beams = np.unique(beams)
num_values = complex_data.shape[1]

# Create a mapping for angles to indices
angle_map = {angle: idx for idx, angle in enumerate(unique_angles)}

# Initialize a 3D array with the shape (num_angles, num_beams, num_values)
result = np.zeros((len(unique_angles), len(unique_beams), num_values), dtype=complex)

# Fill the array with the complex numbers
for i, angle in enumerate(unique_angles):
    for j, beam in enumerate(unique_beams):
        # Mask to select rows where angle and beam match
        mask = (angles == angle) & (beams == beam)
        
        # Check if more than one row matches the (angle, beam) pair
        if np.sum(mask) == 1:
            result[i, j, :] = complex_data[mask, :].flatten()
        else:
            # If multiple rows match, you can average them or take the first one
            result[i, j, :] = complex_data[mask, :][0]  # Take the first row that matches

print(result[0, 45, :])

fs = 1  # Adjust if necessary (sampling rate)
nfft = 1024  # High resolution for better frequency bin separation
frequencies, psd = welch(iq_data, fs=fs, nperseg=256, nfft=nfft, window='hann', return_onesided=True)

# Find the peak frequency (single-tone signal)
peak_idx = np.argmax(psd)
peak_freq = frequencies[peak_idx]

# Define a small bandwidth around the peak for signal power estimation
num_bins = 5  # Adjust for better peak isolation
low_idx = max(0, peak_idx - num_bins)
high_idx = min(len(frequencies) - 1, peak_idx + num_bins)

# Create a mask for the signal power region
signal_mask = np.zeros_like(psd, dtype=bool)
signal_mask[low_idx:high_idx + 1] = True

# Compute signal power
signal_power = np.trapz(psd[signal_mask], frequencies[signal_mask])

# Estimate noise power using the median noise floor
noise_floor = np.median(psd[~signal_mask])
noise_power = noise_floor * (frequencies[-1] - frequencies[0])

# Compute SNR in dB
snr_db = 10 * np.log10(signal_power / noise_power)

print(f"SNR (Fixed PSD Method): {snr_db:.2f} dB")

# Plot PSD with corrections
plt.figure(figsize=(10, 5))
plt.semilogy(frequencies, psd, label="PSD", linewidth=2)
plt.axvline(x=peak_freq, color='r', linestyle='--', label="Tone Frequency")
plt.fill_between(frequencies, psd, where=signal_mask, color='g', alpha=0.5, label="Signal Power")
plt.fill_between(frequencies, psd, where=~signal_mask, color='b', alpha=0.3, label="Noise Power")
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density (dB)')
plt.title('PSD of IQ Data (Fixed Wrap-Around Issue)')
plt.legend()
plt.grid()
plt.show()# Extract the IQ data for the selected angle and beam