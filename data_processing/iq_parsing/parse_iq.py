#%%%
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch

df = pd.read_csv('../../../beamtracking18/beam_iq.csv', header=None)

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
# Example function to compute PSD (Power Spectral Density) for a given IQ data
def compute_psd(iq_data, fs=1e6, nfft=1024):
    f, Pxx = welch(iq_data, fs=fs, nperseg=nfft)
    return f, Pxx

# Assuming result is a 3D array (angles, beams, complex data)
# Initialize variables for signal and noise power
signal_power = []
noise_power = []

# Define a small constant to avoid division by zero
epsilon = 1e-10

# Define noise and signal bands based on expected frequency ranges
noise_band = (10, 20)  # Example: select frequencies between 10-20 Hz as noise band
signal_band = (30, 40)  # Example: select frequencies between 30-40 Hz as signal band

# Iterate over angles and beams
for i in range(result.shape[0]):  # Iterate over angles
    for j in range(result.shape[1]):  # Iterate over beams
        iq_data = result[i, j, :]  # Get the IQ data for the (angle, beam) combination
        f, Pxx = compute_psd(iq_data)  # Compute the PSD
        
        # Check the frequency range to ensure the bands are within the PSD range
        if np.max(f) < signal_band[1] or np.min(f) > noise_band[0]:
            print(f"Frequency bands are out of bounds for Angle {unique_angles[i]}, Beam {unique_beams[j]}")
            continue  # Skip this iteration if bands are out of bounds

        # Find the indices of the signal and noise bands in the frequency array
        signal_indices = np.where((f >= signal_band[0]) & (f <= signal_band[1]))[0]
        noise_indices = np.where((f >= noise_band[0]) & (f <= noise_band[1]))[0]
        
        # Check if valid indices exist for signal/noise
        if len(signal_indices) == 0 or len(noise_indices) == 0:
            print(f"No valid indices found for Signal/Noise bands for Angle {unique_angles[i]}, Beam {unique_beams[j]}")
            continue  # Skip this iteration if no valid indices found

        # Compute the signal power (mean power within the signal band)
        signal_power_value = np.mean(Pxx[signal_indices])
        noise_power_value = np.mean(Pxx[noise_indices])

        # Ensure non-zero noise power to avoid NaN SNR
        if noise_power_value < epsilon:
            print(f"Noise power is too small for Angle {unique_angles[i]}, Beam {unique_beams[j]}, using epsilon for SNR")
            noise_power_value = epsilon  # Set to a small value to avoid divide by zero
        
        # Append power values to lists
        signal_power.append(signal_power_value)
        noise_power.append(noise_power_value)

# Compute SNR for each (angle, beam) combination
snr_values = np.array(signal_power) / np.array(noise_power)

# Print the SNR values for each (angle, beam) combination
for i in range(len(snr_values)):
    angle = unique_angles[i % result.shape[0]]
    beam = unique_beams[i // result.shape[0]]
    print(f"SNR for Angle {angle}, Beam {beam}: {snr_values[i]}")

# Plot the PSD of a specific (angle, beam) for inspection
angle_to_plot = 0  # Example: plot the first angle
beam_to_plot = 0   # Example: plot the first beam
iq_data = result[angle_to_plot, beam_to_plot, :]
f, Pxx = compute_psd(iq_data)

plt.plot(f, 10 * np.log10(Pxx))
plt.title(f'Power Spectral Density (PSD) - Angle {unique_angles[angle_to_plot]}, Beam {unique_beams[beam_to_plot]}')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Power [dB/Hz]')
plt.grid(True)
plt.show()